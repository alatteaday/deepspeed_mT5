from torch.utils.data import Dataset, DataLoader

import torch
from transformers import T5Tokenizer
#from datasets import load_dataset, Dataset
from accelerate import DistributedType

import pandas as pd
import argparse
import os


class FinanceData(Dataset):
    def __init__(self, args, tokenizer, split='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        
        assert split in ['train', 'val', 'test'], '[! data.py] split not in [train/val/test]'
        data_dir = os.path.join(args.data_root_dir, split+'.csv')
        df = pd.read_csv(data_dir)
        self.inputs = list(df.txt_noise)  
        self.labels = list(df.txt)  

        """
        # max_len_input = max([len(self.tokenizer(i).input_ids) for i in self.inputs])
        # max_len_label = max([len(self.tokenizer(i).input_ids) for i in self.labels])
        
        # the max length of input sentences = 1039
        # the max length of label sentences = 789
        """
        
        self.max_len_input = 1039
        self.max_len_label = 789
                
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.split == 'test':
            input = self.tokenizer(self.inputs[idx], truncation=True, max_length=self.max_len_input, return_tensors='pt')
            with self.tokenizer.as_target_tokenizer():
                label = self.tokenizer(self.labels[idx], truncation=True, max_length=self.max_len_label, return_tensors='pt')
            # input['input_ids'] = input.input_ids.squeeze(0)
            # input['attention_mask'] = input.attention_mask.squeeze(0)
            # label['input_ids'] = label.input_ids.squeeze(0)
            # label['attention_mask'] = label.attention_mask.squeeze(0)
            
        else:
            input = self.tokenizer(self.inputs[idx], max_length=self.max_len_input, padding='max_length', return_tensors='pt')
            with self.tokenizer.as_target_tokenizer():
                label = self.tokenizer(self.labels[idx], max_length=self.max_len_input, padding='max_length', return_tensors='pt') 
            
        input['input_ids'] = input.input_ids.squeeze(0)
        input['attention_mask'] = input.attention_mask.squeeze(0)
        label['input_ids'] = label.input_ids.squeeze(0)
        label['attention_mask'] = label.attention_mask.squeeze(0)
        
        return (input, label)

    def collate_fn(self, batch):
        print(batch)
        
        max_len = 0
        for data in batch:
            max_len_data = max([len(data[0].input_ids), len(data[1].input_ids)])
            if max_len < max_len_data:
                max_len = max_len_data

        new_batch = []
        for data in batch:
            new_data = []
            for sen in data:
                sen = self.tokenizer.pad(sen, padding="max_length", max_length=max_len, return_tensors="pt")
                new_data.append(sen)
            print(type(new_data))
            new_data = set(new_data)
            new_batch.append(new_data)
        new_batch = set(new_batch)
                
        print(new_batch)
        exit()
        # input = self.tokenizer.pad(input, padding="max_length", max_length=max_len, return_tensors="pt")
        # label = self.tokenizer.pad(label, padding="max_length", max_length=max_len, return_tensors="pt")

        # return (input, label)
        return batch
            
        
        
        
class FinanceDataHuggingface():  # did not work
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
                    remove_columns=['article_idx', 'txt', 'txt_noise'], load_from_cache_file=False)

    def _tokenize_fn(self, examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = self.tokenizer(examples['txt'],
                                truncation=True,
                                padding='max_length',
                                max_length=self.args.max_len, )
                                #return_tensors="pt")
        # outputs['labels'] = self.tokenizer(examples['txt_noise'], 
        #                         truncation=True, 
        #                         padding='max_length', 
        #                         max_length=self.args.max_len, 
        #                         return_tensors="pt")['input_ids']
        """
        max_length = max(len(outputs['input_ids']), len(outputs['labels']))
        for key, value in outputs.items():  # [0] list is appended to keys which will remove
            outputs[key] = value + [self.tokenizer.pad_token_id] * (max_length - len(outputs[key]))
        """
        return outputs

    def _collate_fn(self, examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if self.args.accelerator.distributed_type == DistributedType.TPU:
            return self.tokenizer.pad(examples,
                                        padding="max_length",
                                        max_length=self.args.max_len,
                                        return_tensors="pt")
        return self.tokenizer.pad(examples,
                                    padding="longest",
                                    return_tensors="pt")

    def get_dataloaders(self, split):
        assert split in ['train', 'val', 'test']

        # Build DataLoader and return it
        if split == 'train':
            loader = DataLoader(self.tokenized_dataset['train'],
                                # collate_fn=self._collate_fn,
                                batch_size=self.args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)
        elif split == 'val':
            loader = DataLoader(self.tokenized_dataset['val'],
                                # collate_fn=self._collate_fn,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                pin_memory=True)
        else:
            loader = DataLoader(self.tokenized_dataset['test'],
                                # collate_fn=self._collate_fn,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                pin_memory=True)

        return loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root_dir', default='data')
    parser.add_argument('--batch_size', default=3)
    args = parser.parse_args()   
    tokenizer = tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")

    t = ['10년 내로 식물성 단백질 등으로 만든 고기인 대체육 소비가 전 세계 육고기 소비 시장의 10%를 차지하게 될 것이란 관측이 제기됐다.', '나는 그렇다.']
    t_g = ['십년 내로 식물성 단백질 등으로 만든 고기인 대체육 소비가 전 세계 육고기 소비 시장의 십퍼센트를 차지하게 될 것이란 관측이 제기됐다.', '나는 그러타.']

    input = tokenizer(t, truncation=True, max_length=2048, padding=True)
    with tokenizer.as_target_tokenizer():
        label = tokenizer(t_g, truncation=True, max_length=2048, padding=True)
    
    data = FinanceData(args, tokenizer, split='train')
    print(len(data))
    print(data[300][1])
    # print(data.tokenized_dataset['train'][0])
    # dataloader = data.get_dataloaders('train')
    # print(dataloader)
    # print(data.tokenized_inputs['train'][0])
   


