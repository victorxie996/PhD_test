import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

from transformers import BertTokenizer, BertModel


class Model(nn.Module):

    def __init__(self, title_embedding_size=768, max_seq_len=128, output_size=2, bert_pretrained_name='bert-base-uncased'):
        super(Model, self).__init__()

        self.title_embedding = TitleEmbedding(title_embedding_size=title_embedding_size,
                                                  max_seq_len=max_seq_len,
                                                  bert_pretrained_name=bert_pretrained_name)

        self.fc = nn.Linear(title_embedding_size, output_size)


    def forward(self, indexed_tokens, segments_ids):

        x = self.title_embedding(indexed_tokens=indexed_tokens, segments_ids=segments_ids)

        x = self.fc(x)

        return x

# title embedding
# Input as title sentence without preprocessing
# use bert here
class TitleEmbedding(nn.Module):

    def __init__(self, title_embedding_size=768, max_seq_len=512, bert_pretrained_name='bert-base-uncased'):
        super(TitleEmbedding, self).__init__()

        self.bert = BertModel.from_pretrained(bert_pretrained_name)


    def forward(self, indexed_tokens, segments_ids):
        # encoded_layers.shape: (batchsize, max_seq_len, 768)
        encoded_layers, pooler_embedding = self.bert(indexed_tokens.detach(), segments_ids.detach())

        # token embedding的平均值作为sentence embedding
        # TODO 这里可调，先用[CLS]的embedding
        #embedding = encoded_layers.mean(dim=1)
        embedding = encoded_layers[:, 0, :]

        return embedding