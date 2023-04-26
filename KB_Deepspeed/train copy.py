import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

#from datasets import load_dataset
import evaluate
import accelerate
from accelerate import (Accelerator, DeepSpeedPlugin, infer_auto_device_map, init_empty_weights,
                    load_checkpoint_and_dispatch, DistributedType, dispatch_model)

from transformers import (MT5Config, MT5ForConditionalGeneration, MT5Tokenizer, T5Tokenizer,
                        get_linear_schedule_with_warmup, set_seed, )
from transformers.deepspeed import (HfTrainerDeepSpeedConfig, deepspeed_init) 
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer

import argparse
import numpy as np
import os
import gc
from tqdm import tqdm
import socket
from contextlib import closing

from data import FinanceData
#from utils import get_device, get_gpu_free_memory, save_checkpoint

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

def get_args():
    ## config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root_dir', default='./data')
    
    parser.add_argument('--pretrained_model', type=str, default='google/mt5-base', choices=['google/mt5-base', 'google/mt5-small'])
    parser.add_argument('--save_dir', type=str, default='/home/jiyun/KB_Deepspeed/checkpoint', help='A directory checkpoints will be saved in')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', default=0.0009)
    parser.add_argument('--seed', default=1234)
    parser.add_argument('--epochs', default=5)
    parser.add_argument('--max_len', default=1024)
    
    # Accelerator
    parser.add_argument('--cpu', action='store_true', help="If passed, will train on the CPU.")
    parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--with_tracking', action='store_true', 
        help="Whether to load in all available experiment trackers from the environment and use them for logging.")
    parser.add_argument('--checkpointing_steps', type=str, default=None, 
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help="A directory of a checkpoint folder")
    parser.add_argument('--output_dir', type=str, default='.', 
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.")
    parser.add_argument('--logging_dir', type=str, default='logs',
        help="Location on where to store experiment tracking logs")


    args = parser.parse_args() 
    return args


def train(args):
    # Accelerator and DeepSpeed
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=2)
    #accelerator = Accelerator(fp16=True, deepspeed_plugin=deepspeed_plugin)
    #accelerator = Accelerator()

    # Initialize accelerator
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=args.cpu, mixed_precision=args.mixed_precision, deepspeed_plugin=deepspeed_plugin, 
            log_with="all", logging_dir=args.logging_dir
        )
    else:
        accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision, deepspeed_plugin=deepspeed_plugin)

    print("Load Accelerator")

    args.accelerator = accelerator
    # args.device = accelerator.device

    if hasattr(args.checkpointing_steps, "isdigit"):
        if args.checkpointing_steps == "epoch":
            checkpointing_steps = args.checkpointing_steps
        elif args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
        else:
            raise ValueError(
                f"Argument `checkpointing_steps` must be either a number or `epoch`. `{args.checkpointing_steps}` passed."
            )
    else:
        checkpointing_steps = None
    
    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if args.batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
        gradient_accumulation_steps = args.batch_size // MAX_GPU_BATCH_SIZE
        args.batch_size = MAX_GPU_BATCH_SIZE
    args.gradient_accumulation_steps = gradient_accumulation_steps

    set_seed(args.seed)

    # Load Data
    # data = FinanceData(args, tokenizer)
    # train_dataloader = data.get_dataloaders('train')
    # val_dataloader = data.get_dataloaders('val')
    
    train_data = FinanceData(args, tokenizer, 'train')
    val_data = FinanceData(args, tokenizer, 'val')
    collate_fn = train_data.collate_fn
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)  # TODO collate_fn Dataloader
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=args.batch_size)

    print("Load DataLoader")

    # Load Model
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model)
    model = MT5ForConditionalGeneration.from_pretrained(args.pretrained_model)

    print("Load Model")

    # Optimizer and Metrics
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    metrics = {
        'bertscore':
        evaluate.load('bertscore'),
        'meteor':
        evaluate.load('meteor'),
    }

    # Scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * args.epochs) // gradient_accumulation_steps,
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the stating epoch so files are named properly
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(args.epochs):
        args.epoch = epoch
        model.train()
        train_epoch(args, model, train_dataloader, optimizer, lr_scheduler)
        model.eval()
        val_epoch(args, model, val_dataloader)


def train_epoch(args, model, dataloader, optimizer, lr_scheduler):
    if args.with_tracking:
        total_loss = 0

    # deepspeed needs to know your gradient accumulation steps before hand, so don't forget to pass it
    # Remember you still need to do gradient accumulation by yourself, just like you would have done without deepspeed
    for i, batch in enumerate(tqdm(dataloader)):
        # We need to skip steps until we reach the resumed step
        if args.resume_from_checkpoint and epoch == starting_epoch:
            if resume_step is not None and step < resume_step:
                overall_step += 1
                continue
        input, label = batch          
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        input.to(args.accelerator.device)
        label.to(args.accelerator.device)
        
        #input = {'input_ids':batch.input_ids, 'attention_mask':batch.attention_mask, }  # if using dataloader by huggingface
        #label.to(args.device_map['lm_head'])

        outputs = model(**input, labels=label.input_ids)
        loss = outputs.loss
        loss = loss / args.gradient_accumulation_steps
        print(loss)

        # We keep track of the loss at each epoch
        if args.with_tracking:
            total_loss += loss.detach().float()
        args.accelerator.backward(loss)

        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        overall_step += 1
        
        if isinstance(checkpointing_steps, int):
            output_dir = f"step_{overall_step}"
            if overall_step % checkpointing_steps == 0:
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)
        
        # How to save your Transformer?
        args.accelerator.wait_for_everyone()
        unwrapped_model = args.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.save_dir, save_function=args.accelerator.save, state_dict=args.accelerator.get_state_dict(model))


def val_epoch(args, model, dataloader):
    for i, batch in enumerate(dataloader):
        input, label = batch
        input.to(args.accelerator.device)
        label.to(args.accelerator.device)

        with torch.no_grad():
            outputs = model(**input, labels=label.input_ids)
        preds = outputs.logits.argmax(dim=-1)
        
        predict, refere, input_t = args.accelerator.gather((
            preds.to(args.device),
            label["input_ids"].to(args.device),  # Should load the tensors to gpu:0
            input['input_ids'].to(args.device)))  # Should load the tensors to gpu:0
        
        results = {}
        for key, metric in metrics.items():
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
            if key == 'bertscore':
                results[key] = metric.compute(lang='others')
            else:
                results[key] = metric.compute()
        # Use accelerator.print to print only on the main process.
        args.accelerator.print(f"epoch {args.epoch}:", results)
        if args.with_tracking:
            args.accelerator.log(
                {
                    "accuracy": results["accuracy"],
                    "f1": results["f1"],
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": args.epoch,
                },
                step = args.epoch,
            )

        if checkpointing_steps == "epoch":  # TODO checkpoing_step
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            args.accelerator.save_state(output_dir)

    if args.with_tracking:
        args.accelerator.end_training()
        

if __name__ == '__main__':
    args = get_args()

    # sock = socket.socket()
    # sock.bind(('', 0))
    # s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # sock.getsockname()[1]

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('',0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        free_port = s.getsockname()[1]

    # os.environ['MASTER_ADDR'] = socket.gethostbyname(socket.gethostname())
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ['MASTER_PORT'] = str(free_port)
    # os.environ["MASTER_PORT"] = "9994"
    os.environ['RANK'] = '0'
    os.environ["LOCAL_RANK"] = '0'  # rank of the process during distributed training
    os.environ['WORLD_SIZE'] = '1'  # the num of process
    
    print("Set OSVariables")

    output_dir = './'

    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model)
    model = MT5ForConditionalGeneration.from_pretrained(args.pretrained_model)

    train_data = FinanceData(args, tokenizer, 'train')
    val_data = FinanceData(args, tokenizer, 'val')

    training_args = TrainingArguments(output_dir=output_dir,
                                      deepspeed="ds_config_zero3.json")  # json file -> should find on deepspeed MS tutorial site / should I create a new json file of deepspeed options?
    
    print(training_args)
    exit()

    trainer = Trainer(model=model, 
                      args=training_args, 
                      train_dataset=train_data, 
                      eval_datset=val_data, 
                      tokenizer=tokenizer,
                      )
    trainer.train()  # args: resume_from_checkpoint, trial, ignore_keys_for_eval,


    train(args)
    
