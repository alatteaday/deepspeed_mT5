import argparse
import os

import torch
# from datasets import load_dataset
from torch.optim import AdamW
# from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup,  
                          MT5ForConditionalGeneration, T5Tokenizer,
                          BartForConditionalGeneration, BartTokenizer, 
                          MBartForConditionalGeneration, MBart50TokenizerFast,
                          get_scheduler)

from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed, DummyOptim, DummyScheduler
import evaluate 
import math

from dataset import FinanceDataset


# MAX_GPU_BATCH_SIZE = 16
# EVAL_BATCH_SIZE = 32

# accelerate launch --config_file deepspeed_config_zero2.yaml accel_ds.py

def get_args():
    parser = argparse.ArgumentParser(description='')

    # ACCELERATOR
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--max_gpu_batch_size', type=int, default=1)

    parser.add_argument('--mixed_precision', type=str, default=None, choices=['no', 'fp16', 'bf16', 'fp8'],
                        help="Whether to use mixed precision. Choose"
                            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
                            "and an Nvidia Ampere GPU.")
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--ckpt_steps', type=str, default=None, 
                        help="Whether the various states should be saved at the end of every n steps,"
                          "or 'epoch' for each epoch.")
    parser.add_argument('--with_tracking', action='store_true', 
                        help="Whether to load in all available experiment trackers from the env and use them for logging")
    parser.add_argument('--output_dir', type=str, default=".", 
                        help="Optional save directiory where all checkpoint folders will be stored")
    parser.add_argument('--logging_dir', type=str, default="logs",
                        help="Location on where to store experiment tracking logs")
    
    # TRAIN
    parser.add_argument('--data_root_dir', type=str, default='./data')
    parser.add_argument('--max_len', type=int, default=1024)

    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_train_steps', type=int, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_warmup_steps', type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--pretrained_model', type=str, default='google/mt5-small', choices=['facebook/mbart-large-50', 'google/mt5-small'])
    parser.add_argument('--resume_from_ckpt', type=str, default=None)
    # parser.add_argument('--args_to_my_script', default='./deepspeed_config_zero2.yaml')
    args = parser.parse_args()

    config = {'lr': args.lr, 
              'num_epochs': args.epochs, 
              'seed': args.seed, 
              'batch_size': args.batch_size,
              }
    
    return args, config

def train(config, args):
    if args.with_tracking:
        accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision, 
                                  log_with="all", logging_di=args.logging_dir)
        run = os.path.split(__file__)[-1].split('.')  # initialize the trackers and store the config
        accelerator.init_trackers(run, config)
    else:
        accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)

    if hasattr(args.ckpt_steps, "isdigit"):
        if args.ckpt_steps == 'epoch':
            ckpt_steps = args.ckpt_steps
        elif args.ckpt_steps.isdigit():
            ckpt_steps = int(args.ckpt_steps)
        else:
            raise ValueError(
                "Argument 'ckpt_steps' must be either a number of 'epoch'. {} passed.".format(args.ckpt_steps)
            )
    else:
        ckpt_steps = None

    # batch_size = args.batch_size

    # if args.with_tracking:
    #     run = os.path.split(__file__)[-1].split(".")[0]
    #     accelerator.init_trackers(run, config)

    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model)
    # tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="ko_KR", tgt_lang="ko_KR")


    fin_dataset = FinanceDataset(args, tokenizer, accelerator)
    train_dataloader = fin_dataset.get_dataloaders('train')
    val_dataloader = fin_dataset.get_dataloaders('val')

    gradient_accumulation_steps = 1
    if args.batch_size > args.max_gpu_batch_size and accelerator.distributed_type != DistributedType.TPU:
        gradient_accumulation_steps = args.batch_size // args.max_gpu_batch_size
        args.batch_size = args.max_gpu_batch_size
    args.gradient_accumulation_steps = gradient_accumulation_steps

    set_seed(args.seed)

    model = MT5ForConditionalGeneration.from_pretrained(args.pretrained_model)
    # model = MBartForConditionalGeneration.from_pretrained(args.pretrained_model)
    model = model.to(accelerator.device)

    # optimizer = AdamW(params=model.parameters(), lr=args.lr)

    #####
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.epochs * num_update_steps_per_epoch
    else:
        args.epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # DS Optim + DS Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # Creates Dummy Optimizer if `optimizer` was spcified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.lr)

    # Creates Dummy Scheduler if `scheduler` was spcified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type, optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps, num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
        )

    #####

    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer, num_warmup_steps=100, num_training_steps=(len(train_dataloader)*args.epochs) // gradient_accumulation_steps
    # )

    metrics = {
        # 'bertscore': evaluate.load('bertscore'),
        'meteor': evaluate.load('meteor')
    }

    # Prepare everything
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    args.overall_step = 0  # to keep track of how many total steps we have iterated over
    args.starting_epoch = 0  # to keep track of the stating epoch so files are named properly 

    resume_step = None
    if args.resume_from_ckpt:
        if args.resume_from_ckpt != "":
            accelerator.print("Resumed from the checkpoint: {}".format(args.resume_from_ckpt))
            accelerator.load_state(args.resume_from_ckpt)
            path = os.path.basename(args.resume_from_ckpt)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last

        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            args.starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            args.starting_epoch = resume_step // len(train_dataloader)
            resume_step -= args.starting_epoch * len(train_dataloader)

    args.ckpt_steps = ckpt_steps
    args.resume_step = resume_step
    args.accelerator = accelerator

    for epoch in range(args.starting_epoch, args.epochs):
        args.epoch = epoch
        train_epoch(args, model, train_dataloader, optimizer, lr_scheduler)
        val_epoch(args, model, val_dataloader, metrics)

    if args.with_tracking:
        accelerator.end_training()
  

def train_epoch(args, model, dataloader, optimizer, lr_scheduler):
    accelerator = args.accelerator
    model.train()
    if args.with_tracking:
        total_loss = 0
    if args.resume_from_ckpt and args.epoch == args.starting_epoch and args.resume_step is not None:
        dataloader = accelerator.skip_first_batches(dataloader, args.resume_step)
        args.overall_step += args.resume_step

    for step, batch in enumerate(dataloader):
        batch.to(accelerator.device)
        print(batch.input_ids.shape)
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / args.gradient_accumulation_steps

        if args.with_tracking:
            total_loss += loss.detach().float()
        accelerator.backward(loss)

        if step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        args.overall_step += 1

        if isinstance(args.ckpt_steps, int):
            output_dir = "step_{}".format(args.overall_step)
            if args.overall_step %args.ckpt_steps == 0:
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)


def val_epoch(args, model, dataloader, metrics):
    accelerator = args.accelerator
    model.eval()
    for step, batch in enumerate(dataloader):
        batch.to(accelerator.device)
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        preds, refers = accelerator.gather_for_metrics((preds, batch['labels']))

        results = {}
        for key, metric in metrics.items():
            metrics.add_batch(
                predictions=preds,
                references=refers
            )
            if key == 'bertscore':
                results[key] = metric.compute(lang='others')
            else:
                results[key] = metric.compute()

        accelerator.print("epoch {}: ".format(results))
        if args.with_tracking:
            accelerator.log(
                {
                    "bertscore": results["bertscore"],
                    "meteor": results["meteor"],
                    "val_loss": loss.item() / len(dataloader),
                    "epoch": args.epoch,
                },
                step=args.epoch,
            )

        if args.ckpt_steps == 'epoch':
            output_dir = "epoch_{}".format(args.epoch)
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)



        # New Code #
        # Evaluates using the best checkpoint
        perplexity, eval_loss = evaluate(args, model, eval_dataloader, accelerator, eval_dataset)
        logger.info(f"Best model metrics: perplexity: {perplexity} eval_loss: {eval_loss}")
        if perplexity != best_metric:
            raise AssertionError(
                f"Best metric {best_metric} does not match the metric {perplexity} of the loaded best model."
            )

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            # New Code #
            # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
            # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
            # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
            # For Zero Stages 1 and 2, models are saved as usual in the output directory.
            # The model name saved is `pytorch_model.bin`
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                if args.push_to_hub:
                    repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity, "eval_loss": eval_loss.item()}, f)


if __name__ == "__main__":
    args, config = get_args()
    train(config, args)