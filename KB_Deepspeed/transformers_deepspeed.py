#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Referenced Huggingface Transformers Source code 

from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer

import argparse
import numpy as p
import os
import socket
from contextlib import closing

from dataset import FinanceDataset2


with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
    s.bind(('', 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    free_port = s.getsockname()[1]

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(free_port)  
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

print("Set OSVariables")    

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root_dir', default='./data')
    parser.add_argument('--max_len', default=1024)
    parser.add_argument('--pretrained_model', type=str, default='google/mt5-base')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=int, default=1e-6)

    parser.add_argument('--output_dir', type=str, default='./')

    parser.add_argument('--local_rank', default=0)
    parser.add_argument('--world_size', default=1) 

    args = parser.parse_args()
    return args

args = get_args()
tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model)
model = MT5ForConditionalGeneration.from_pretrained(args.pretrained_model)  

fin_dataset = FinanceDataset2(args, tokenizer)
train_data = fin_dataset.tokenized_dataset['train']
val_data = fin_dataset.tokenized_dataset['val']

training_args = TrainingArguments(output_dir=args.output_dir,
                                  deepspeed="ds_config_zero3.json")  

trainer = Trainer(model=model, 
                  args=training_args, 
                  train_dataset=train_data, 
                  eval_dataset=val_data, 
                  tokenizer=tokenizer,
                  )
trainer.train()  # args: resume_from_checkpoint, trial, ignore_keys_for_eval,
