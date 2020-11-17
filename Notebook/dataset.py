import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import glob
import re
import pandas as pd
import tqdm
from transformers import BertTokenizer
import json

class NewsDataset(Dataset):
    def __init__(self, real_news_path='data/True.csv', fake_news_path='data/Fake.csv', bert_pretrained_name='bert-base-uncased', max_seq_len=128):

        self.max_seq_len = max_seq_len

        real_news = pd.read_csv(real_news_path)
        fake_news = pd.read_csv(fake_news_path)

        self.data = []
        self.label = []

        for index, row in tqdm.tqdm(real_news.iterrows()):
            title, text = row['title'], row['text']  # Take out the required value like a dictionary

            self.data.append(text)
            self.label.append(1)

        for index, row in tqdm.tqdm(fake_news.iterrows()):
            title, text = row['title'], row['text']  # Take out the required value like a dictionary

            self.data.append(text)
            self.label.append(0)

        self.label = torch.tensor(self.label).long()

        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sentence = self.data[idx]
        label = self.label[idx]
        indexed_tokens, segments_ids = self.preprocess_sentence(sentence)

        return indexed_tokens, segments_ids, label

    def preprocess_sentence(self, sentence):
        """Process sentence to get input in bert format（indexed_tokens, segments_ids）
        """
        # Process title, get index
        tokenized_text = self.tokenizer.tokenize(sentence)[:self.max_seq_len - 2]
        # get [PAD]
        if len(tokenized_text) < self.max_seq_len - 1:
            tokenized_text.extend(['[PAD]' for _ in range(self.max_seq_len - len(tokenized_text) - 1)])
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        
        segments_ids = [0 for _ in range(len(indexed_tokens))]

        return torch.tensor(indexed_tokens), torch.tensor(segments_ids)


if __name__ == '__main__':
    dataset = NewsDataset()
    a = dataset[0]
    c = dataset[2]
    print(a)
    print(c)



