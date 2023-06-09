from torch.utils.data import Dataset, DataLoader

import torch
from transformers import T5Tokenizer
from datasets import load_dataset, Dataset
from accelerate import DistributedType, Accelerator

import pandas as pd
import argparse
import os

        
class FinanceDataset():
    def __init__(self, args, tokenizer, accelerator):
        self.args = args
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        
        self.dataset = load_dataset('csv',
                                    data_files={
                                        'train': os.path.join(args.data_root_dir, 'train.csv'),
                                        'val': os.path.join(args.data_root_dir, 'val.csv'),
                                        'test': os.path.join(args.data_root_dir, 'test.csv')
                                    })
        
        with self.accelerator.main_process_first(): 
            self.tokenized_dataset = self.dataset.map(
                        self._tokenize_fn,
                        # batched=True,
                        remove_columns=['article_idx', 'txt', 'txt_noise'])

    def _tokenize_fn(self, examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = self.tokenizer(examples['txt'],
                                truncation=True,
                                padding='max_length',
                                max_length=self.args.max_len, )
                                #return_tensors="pt")
        outputs['labels'] = self.tokenizer(examples['txt_noise'], 
                                truncation=True, 
                                padding='max_length', 
                                max_length=self.args.max_len, 
                                return_tensors="pt")['input_ids']
        """
        max_length = max(len(outputs['input_ids']), len(outputs['labels']))
        for key, value in outputs.items():  # [0] list is appended to keys which will remove
            outputs[key] = value + [self.tokenizer.pad_token_id] * (max_length - len(outputs[key]))
        """
        return outputs

    def _collate_fn(self, examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        max_length = args.max_len if self.accelerator.distributed_type == DistributedType.TPU else None

        # When using mixed precision we want round multiples of 8/16
        if self.accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif self.accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        # if self.args.accelerator.distributed_type == DistributedType.TPU:
        #     return self.tokenizer.pad(examples,
        #                                 padding="max_length",
        #                                 max_length=self.args.max_len,
        #                                 return_tensors="pt")
        return self.tokenizer.pad(examples,
                                    padding="longest",
                                    max_length=max_length,
                                    pad_to_multiple_of=pad_to_multiple_of,
                                    return_tensors="pt")

    def get_dataloaders(self, split):
        assert split in ['train', 'val', 'test']

        # Build DataLoader and return it
        if split == 'train':
            loader = DataLoader(self.tokenized_dataset['train'],
                                collate_fn=self._collate_fn,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=True)
        elif split == 'val':
            loader = DataLoader(self.tokenized_dataset['val'],
                                collate_fn=self._collate_fn,
                                batch_size=self.args.eval_batch_size,
                                shuffle=False,
                                pin_memory=True)
        else:
            loader = DataLoader(self.tokenized_dataset['test'],
                                collate_fn=self._collate_fn,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                pin_memory=True)

        return loader


class FinanceDataset2():
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        
        self.dataset = load_dataset('csv',
                                    data_files={
                                        'train': os.path.join(args.data_root_dir, 'train.csv'),
                                        'val': os.path.join(args.data_root_dir, 'val.csv'),
                                        'test': os.path.join(args.data_root_dir, 'test.csv')
                                    })
        
        self.tokenized_dataset = self.dataset.map(
                        self._tokenize_fn,
                        # batched=True,
                        remove_columns=['article_idx', 'txt', 'txt_noise'])

    def _tokenize_fn(self, examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = self.tokenizer(examples['txt'],
                                truncation=True,
                                padding='max_length',
                                max_length=self.args.max_len, )
                                #return_tensors="pt")
        outputs['labels'] = self.tokenizer(examples['txt_noise'], 
                                truncation=True, 
                                padding='max_length', 
                                max_length=self.args.max_len, 
                                return_tensors="pt")['input_ids']
        """
        max_length = max(len(outputs['input_ids']), len(outputs['labels']))
        for key, value in outputs.items():  # [0] list is appended to keys which will remove
            outputs[key] = value + [self.tokenizer.pad_token_id] * (max_length - len(outputs[key]))
        """
        return outputs

    def _collate_fn(self, examples):
        return self.tokenizer.pad(examples,
                                    padding="longest",
                                    max_length=args.max_len,
                                    return_tensors="pt")

    def get_dataloaders(self, split):
        assert split in ['train', 'val', 'test']

        # Build DataLoader and return it
        if split == 'train':
            loader = DataLoader(self.tokenized_dataset['train'],
                                collate_fn=self._collate_fn,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=True)
        elif split == 'val':
            loader = DataLoader(self.tokenized_dataset['val'],
                                collate_fn=self._collate_fn,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                pin_memory=True)
        else:
            loader = DataLoader(self.tokenized_dataset['test'],
                                collate_fn=self._collate_fn,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                pin_memory=True)

        return loader



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root_dir', default='data')
    parser.add_argument('--batch_size', default=3)
    parser.add_argument('--eval_batch_size', default=3)

    parser.add_argument('--max_len', default=1024)
    args = parser.parse_args()   
    tokenizer = tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")

    t = ['10년 내로 식물성 단백질 등으로 만든 고기인 대체육 소비가 전 세계 육고기 소비 시장의 10%를 차지하게 될 것이란 관측이 제기됐다.', '나는 그렇다.']
    t_g = ['십년 내로 식물성 단백질 등으로 만든 고기인 대체육 소비가 전 세계 육고기 소비 시장의 십퍼센트를 차지하게 될 것이란 관측이 제기됐다.', '나는 그러타.']

    # input = tokenizer(t, truncation=True, max_length=2048, padding=True)
    # with tokenizer.as_target_tokenizer():
    #     label = tokenizer(t_g, truncation=True, max_length=2048, padding=True)
    
    # data = FinanceData(args, tokenizer, split='train')
    # print(len(data))
    # print(data[300][1])
    # print(data.tokenized_dataset['train'][0])
    # dataloader = data.get_dataloaders('train')
    # print(dataloader)
    # print(data.tokenized_inputs['train'][0])

    accelerator = Accelerator(cpu=False, mixed_precision=None)

    fin_dataset = FinanceDataset(args, tokenizer, accelerator)
    train_dataloader = fin_dataset.get_dataloaders('train')
    val_dataloader = fin_dataset.get_dataloaders('val')

    for i, t in enumerate(train_dataloader):
        print(t['input_ids'].shape)
        if t['input_ids'].shape == 3:
            print(i)
            print(t['input_ids'].shape)
            print(t['input_ids'])
            print('\n')

    print('VAL')
    for i, t in enumerate(val_dataloader):
        print(t['input_ids'].shape)
        if t['input_ids'].shape == 3:
            print(i)
            print(t['input_ids'].shape)
            print(t['input_ids'])
            print('\n')


   


