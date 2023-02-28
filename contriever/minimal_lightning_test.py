import json
import os
import random
import re
import sys
from argparse import ArgumentParser
from codecs import EncodedFile
from datetime import datetime
from shutil import copyfile
from typing import Optional

import datasets
import numpy as np
import pandas as pd
import torch
import tqdm
from datasets import load_dataset
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer, seed_everything)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import transformers
from transformers import (AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup, get_scheduler)

print(" ------------- WARNING ONLY USING CPU FOR NOW ---------------------")
dev = 'cpu'
device = torch.device(dev)


class MSMARCOData(LightningDataModule):

  def __init__(
      self,
      model_name: str,
      langs,
      max_seq_length: int = 250,
      train_batch_size: int = 32,
      eval_batch_size: int = 32,
      num_negs: int = 3,
      **kwargs,
  ):
    super().__init__()
    self.model_name = model_name
    self.max_seq_length = max_seq_length
    self.train_batch_size = train_batch_size
    self.eval_batch_size = eval_batch_size
    self.langs = langs
    self.num_negs = num_negs
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    #def setup(self, stage: str):
    print(f"!!!!!!!!!!!!!!!!!! SETUP {os.getpid()}  !!!!!!!!!!!!!!!")

    #Get the queries
    self.queries = {lang: {} for lang in self.langs}

    # for lang in self.langs:
    #     for row in tqdm.tqdm(load_dataset('unicamp-dl/mmarco', f'queries-{lang}')['train'], desc=lang):
    #         self.queries[lang][row['id']] = row['text']

    #Get the passages
    # self.collections = {lang: load_dataset('unicamp-dl/mmarco', f'collection-{lang}')['collection'] for lang in self.langs}

    # asmita's paths
    s = open(
        "/mnt/project/chatbotai/kastan/info-retrieval/contriever/textbook_embeddings/fine_tune_cleaned_training_data.json"
    )
    data = json.load(s)
    df = pd.DataFrame.from_dict(data, orient="index")
    self.triplets = df[['query', 'pos_a', 'neg_a1', 'neg_a2', 'neg_a3']]

    # self.bad_data = []
    # for dataset in [third, fourth]:
    #     for row in dataset:
    #         self.bad_data.append(row['text'])

    # # create tirplets+ of <question, good answer (pos), and 3 bad answers (neg1, neg2, neg3)>
    # self.triplets = []
    # for dataset in [first, second]:
    #     for row in dataset:
    #         itr_counter = 0
    #         # Ensure our negative samples are not the same as each other (and that neg not == pos sample)

    #         neg1, neg2, neg3, = random.choice(self.bad_data), random.choice(self.bad_data), random.choice(self.bad_data)
    #         while ( neg1 == neg2 or neg1 == neg3 or neg2 == neg3 ) and ( any(neg_ex in row['GPT-3-Semantic-Search-Generations']['answer'] for neg_ex in [neg1, neg2, neg3]) ) and itr_counter < 50:
    #             neg1, neg2, neg3, = random.choice(self.bad_data), random.choice(self.bad_data), random.choice(self.bad_data)
    #             itr_counter += 1
    #         if itr_counter == 50:
    #             print("❌❌❌ WARNING: 50 iterations reached, negs may be equal ❌❌❌")
    #         self.triplets.append([row['GPT-3-Semantic-Search-Generations']['question'], row['GPT-3-Semantic-Search-Generations']['answer'],[neg1, neg2, neg3]])

  def collate_fn(self, batch):
    print("INSIDE COLLATE FUNCTION!")
    '''
        # EXPECED DATA FORMAT BEFORE TOKENIZATION
        query_doc_pairs_OUR_INTERPRETATION = [
              [('query1', 'pos1'), ('query2', 'po2')],
                [('query1', 'neg1'), ('query2', 'neg2')],
                [],
                [],
                []
            ]
        '''
    #Create data for list-rank-loss
    query_doc_pairs = [[] for _ in range(1 + 3)]

    #example_train_data = [['query', 'pos', 'neg'],['query2', 'po2', 'neg2']]

    # create a list of lists
    query_doc_pairs = []
    for row in batch.iterrows():
      tmp_row = []
      tmp_row.append(row[1]['query'])
      tmp_row.append(row[1]['pos_a'])
      tmp_row.append(row[1]['neg_a1'])
      tmp_row.append(row[1]['neg_a2'])
      tmp_row.append(row[1]['neg_a3'])

      query_doc_pairs.append(tmp_row)
      ''' 
                  future refernece for multiple negs
                  # for num_neg, neg_id in enumerate(neg_ids):
                      # query_doc_pairs[1+num_neg].append((query_text, row[2]))
                  '''
    ''' ORIGINAL CODE
              query_doc_pairs = [[] for _ in  range(1+self.num_negs)]
              cross_lingual_batch = random.random() < self.cross_lingual_chance 
              for row in batch:
                  qid = row['qid']
                  print('qid', qid)
                  pos_id = random.choice(row['pos'])
                  query_lang = random.choice(self.langs)
                  query_text = self.queries[query_lang][qid]
                      
                  doc_lang = random.choice(self.langs) if cross_lingual_batch else query_lang 
                  query_doc_pairs[0].append((query_text, self.collections[doc_lang][pos_id]['text']))
                  dense_bm25_neg = list(set(row['dense_neg'] + row['bm25_neg']))
                  neg_ids = random.sample(dense_bm25_neg, self.num_negs)
                  for num_neg, neg_id in enumerate(neg_ids):
                      doc_lang = random.choice(self.langs) if cross_lingual_batch else query_lang
                      query_doc_pairs[1+num_neg].append((query_text, self.collections[doc_lang][neg_id]['text']))
              '''
    # print("query_doc_pairs", query_doc_pairs)

    #Now tokenize the data
    features = [
        self.tokenizer(qd_pair,
                       max_length=self.max_seq_length,
                       padding=True,
                       truncation='only_second',
                       return_tensors="pt") for qd_pair in query_doc_pairs
    ]
    # print(self.triplets)

    return features

  def train_dataloader(self):
    print("IN FUNC RETURNING DATALOADERS!!!!")
    return DataLoader(self.triplets,
                      shuffle=True,
                      batch_size=self.train_batch_size,
                      num_workers=32,
                      pin_memory=True,
                      collate_fn=self.collate_fn)


# write if name == main
if __name__ == "__main__":
  dm = MSMARCOData(model_name='facebook/contriever-msmarco', langs='args.langs', train_batch_size=1, num_negs=3)

  my_dl = dm.train_dataloader()

  for ex in my_dl:
    print(ex.keys())
    # print(ex)
    break
