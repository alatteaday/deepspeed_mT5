from torch.utils.data import Dataset, DataLoader

from transformers import T5Tokenizer
from datasets import load_dataset
import pandas as pd
import argparse
import os


class FinanceData(Dataset):
    def __init__(self, args, tokenizer, split='train'):
        if split == 'train':
            data_file = os.path.join(args.data_dir, 'train_data_splited_half_kor_2.csv')
        elif split == 'val':
            data_file = os.path.join(args.data_dir, 'val_data_splited_half_kor_2.csv')
        if split == 'test':
            data_file = os.path.join(args.data_dir, 'test_data_splited_half_kor_2.csv')
        self.split = split
        ## max len
        ## train_data_splited_kor: inputs = 796 / labels = 698

        df = pd.read_csv(data_file)
        #self.inputs = list(df.content_noise)
        #self.labels = list(df.content_prep)

        self.inputs = list(df.txt_noise)  
        self.labels = list(df.txt)  

        self.tokenizer = tokenizer

        max_len_input = 0
        max_len_label = 0
        for i, l in zip(self.inputs, self.labels):
            input = self.tokenizer(i)
            if len(input.input_ids) > max_len_input:
                max_len_input = len(input.input_ids)
            
            label = self.tokenizer(l)
            if len(label.input_ids) > max_len_label:
                max_len_label = len(label.input_ids)

        self.max_len_input = max_len_input
        self.max_len_label = max_len_label
        
        # ips_encoded = []
        # gts_encoded = []
        # max = 0
        # for ip, gt in zip(ips, gts):
        #     #ip_e = self.tokenizer(ip)
        #     ip_e = self._tokenize(ip)
        #     gt_e = self._tokenize(gt)
            
        #     if len(ip_e.input_ids) > 2048:
        #        continue
        #     gt_e = self.tokenizer(gt)
        #     if len(gt_e.input_ids) > 2048:
        #        continue
            
        #     ips_encoded.append(ip_e)
        #     gts_encoded.append(gt_e)
            
        #     if len(ip_e.input_ids) > max:
        #         max = len(ip_e.input_ids)
        # print(max)
        """
        self.ips_encoded = ips_encoded
        self.gts_encoded = gts_encoded

        self.tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
        self.args = args
        self.dataset = load_dataset('csv', data_files={
                                        'train': os.path.join(args.data_root_dir, 'train_data.csv'),
                                        'val': os.path.join(args.data_root_dir, 'val_data.csv'),
                                        'test': os.path.join(args.data_root_dir, 'test_data.csv')
                                    })
        #print(self.dataset['train'])

        self.tokenized_dataset = self.dataset.map(
            self._tokenize_input,
            batched=True,
            remove_columns=['date', 'link', 'content_kor', 'title', 'content', 'eng_idx', 'num_idx']
            )
        
        self.tokenized_dataset = self.dataset.map(
            self._tokenize_label,
            batched=True,
            remove_columns=['date', 'link', 'content_kor', 'title', 'content', 'eng_idx', 'num_idx']
            )
        """
        #print(self.tokenized_dataset['train'])
        
        #print(len(self.tokenized_dataset['train']))

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        # Tokenizing
        
        # in_tensor = tokenizer(self.inputs[idx], truncation=True, max_length=2048, padding="longest", return_tensors='pt')
        # with tokenizer.as_target_tokenizer():
        #     gt_tensor = tokenizer(self.labels[idx], truncation=True, max_length=2048, padding="longest", return_tensors='pt')
        
        if self.split == 'test':
            input = self.tokenizer(self.inputs[idx], truncation=True, max_length=self.max_len_input, return_tensors='pt')
            with self.tokenizer.as_target_tokenizer():
                label = self.tokenizer(self.labels[idx], truncation=True, max_length=self.max_len_label, return_tensors='pt')
        else:
            input = self.tokenizer(self.inputs[idx], truncation=True, max_length=self.max_len_input, padding='max_length', return_tensors='pt')
            with self.tokenizer.as_target_tokenizer():
                label = self.tokenizer(self.labels[idx], truncation=True, max_length=self.max_len_label, padding='max_length', return_tensors='pt')
            

        # print(input, type(input))
        # print(label, type(label))
        # print(type(input.input_ids))

        
        input['input_ids'] = input.input_ids.squeeze(0)
        input['attention_mask'] = input.attention_mask.squeeze(0)

        label['input_ids'] = label.input_ids.squeeze(0)
        label['attention_mask'] = label.attention_mask.squeeze(0)
        
        # print(input, type(input))
        # print(label, type(label))

        return (input, label)

    def _tokenize_input(self, data):
        # max_length=None => use the model max length (it's actually the default)
        input = self.tokenizer(data['content_noise'], truncation=True, max_length=2048)
        return input     

    def _tokenize_label(self, data):
        # max_length=None => use the model max length (it's actually the default)
        label = self.tokenizer(data['content_prep'], truncation=True, max_length=2048)
        return label    

    def _collate_fn(self, examples):
        print(examples)
        # On TPU it's best to pad everything to the same length or training will be very slow.
        return self.tokenizer.pad(examples,
                                padding="longest",
                                return_tensors="pt")


def get_dataloaders(args, tokenizer, split='train'):
    dataset = FinanceData(args, tokenizer, split='train')
    
    assert split in ['train', 'val', 'test']
    # Build DataLoader and return it
    if split == 'train':
        loader = DataLoader(dataset,
                            collate_fn=dataset._collate_fn,
                            batch_size=args.batch_size,
                            shuffle=True,
                            pin_memory=True)
    elif split == 'val':
        loader = DataLoader(dataset,
                            collate_fn=dataset._collate_fn,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True)
    else:
        loader = DataLoader(dataset,
                            collate_fn=dataset._collate_fn,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True)
    return loader



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--batch_size', default=3)
    args = parser.parse_args()    
    tokenizer = tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")

    t = ['10년 내로 식물성 단백질 등으로 만든 고기인 대체육 소비가 전 세계 육고기 소비 시장의 10%를 차지하게 될 것이란 관측이 제기됐다.', '나는 그렇다.']
    t_g = ['십년 내로 식물성 단백질 등으로 만든 고기인 대체육 소비가 전 세계 육고기 소비 시장의 십퍼센트를 차지하게 될 것이란 관측이 제기됐다.', '나는 그러타.']

    input = tokenizer(t, truncation=True, max_length=2048, padding=True)
    with tokenizer.as_target_tokenizer():
        label = tokenizer(t_g, truncation=True, max_length=2048, padding=True)
    
    data = FinanceData(args, tokenizer, 'train')
    print(data[0])
   






import numpy as np
>>> from datasets import Dataset 
>>> from torch.utils.data import DataLoader
>>> data = np.random.rand(16)
>>> label = np.random.randint(0, 2, size=16)
>>> ds = Dataset.from_dict({"data": data, "label": label}).with_format("torch")
>>> dataloader = DataLoader(ds, batch_size=4)
>>> for batch in dataloader:
...     print(batch)    