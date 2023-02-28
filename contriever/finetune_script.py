#importing required libraries

import gzip
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

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

print(" ------------- WARNING ONLY USING CPU FOR NOW ---------------------")
#dev = 'cpu'
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
      cross_lingual_chance: float = 0.0,
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
    self.cross_lingual_chance = cross_lingual_chance  #Probability for cross-lingual batches

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
    s = open("info-retrieval/contriever/textbook_embeddings/fine_tune_cleaned_training_data.json")
    data = json.load(s)
    
    df = pd.DataFrame.from_dict(data, orient="index")
    self.triplets = []
    for row in df.iterrows():
      tmp = []
      tmp2 = []
      tmp.append(row[1]['query'])
      tmp.append(row[1]['pos_a'])
      tmp2.append(row[1]['neg_a1'])
      tmp2.append(row[1]['neg_a2'])
      tmp2.append(row[1]['neg_a3'])

      tmp.append(tmp2)
      self.triplets.append(tmp)

    
  def collate_fn(self, batch):
    #Create data for list-rank-loss
    query_doc_pairs = [[] for _ in range(1 + 3)]

    for row in batch:
      query_text = row[0]
      #pos
      query_doc_pairs[0].append((query_text, row[1]))
      #neg
      for neg_id, neg in enumerate(row[2]):
        query_doc_pairs[1+neg_id].append((query_text, neg))


    #Now tokenize the data
    features = [
        self.tokenizer(qd_pair,
                       max_length=self.max_seq_length,
                       padding=True,
                       truncation='only_second',
                       return_tensors="pt") for qd_pair in query_doc_pairs
    ]
    

    return features

  def train_dataloader(self):
    return DataLoader(self.triplets,
                      shuffle=True,
                      batch_size=self.train_batch_size,
                      num_workers=1,
                      pin_memory=True,
                      collate_fn=self.collate_fn)


class ListRankLoss(LightningModule):

  def __init__(
      self,
      model_name: str,
      learning_rate: float = 2e-5,
      warmup_steps: int = 1000,
      weight_decay: float = 0.01,
      train_batch_size: int = 32,
      eval_batch_size: int = 32,
      **kwargs,
  ):
    super().__init__()

    self.save_hyperparameters()
    print(self.hparams)

    self.config = AutoConfig.from_pretrained(model_name, num_labels=1)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
    self.loss_fct = torch.nn.CrossEntropyLoss()
    self.global_train_step = 0

  def forward(self, **inputs):
    return self.model(**inputs)

  def training_step(self, batch, batch_idx):
    pred_scores = []
    print("batch", len(batch))
    #print("batch a 0", batch[0])
    scores = torch.tensor(0 * len(batch[0]['input_ids']), device=self.model.device)

    print("SCORES")
    print(scores.shape)
    print(len(batch[0]['input_ids']))

    for feature in batch:
      pred_scores.append(self(**feature).logits.squeeze())

    pred_scores = torch.stack(pred_scores, dim=0)
    
    print("SHAPE")
    print(pred_scores)
    print(scores)

    loss_value = self.loss_fct(pred_scores, scores)
    self.global_train_step += 1
    self.log('global_train_step', self.global_train_step)
    self.log("train/loss", loss_value)

    return loss_value

  def setup(self, stage=None) -> None:
    print("IN SETUP")
    if stage != "fit":
      return
    print("IN SETUP w/ FIT")
    # Get dataloader by calling it - train_dataloader() is called after setup() by default
    train_loader = self.trainer.datamodule.train_dataloader()

    # Calculate total steps
    tb_size = self.hparams.train_batch_size * max(1, self.trainer.num_devices)
    ab_size = self.trainer.accumulate_grad_batches
    self.total_steps = (len(train_loader) // ab_size) * self.trainer.max_epochs

    print("{tb_size=}")
    print("{ab_size=}")
    print("{len(train_loader)=}")
    print("{self.total_steps=}")

  def configure_optimizers(self):
    """Prepare optimizer and schedule (linear warmup and decay)"""
    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.hparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=self.hparams.warmup_steps,
        num_training_steps=self.total_steps,
    )

    scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
    return [optimizer], [scheduler]


def main(args):
  dm = MSMARCOData(model_name=args.model,
                   langs=args.langs,
                   train_batch_size=args.batch_size,
                   cross_lingual_chance=args.cross_lingual_chance,
                   num_negs=args.num_negs)

  output_dir = f"output/{args.model.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
  print("Output_dir:", output_dir)

  os.makedirs(output_dir, exist_ok=True)

  wandb_logger = WandbLogger(project="multilingual-cross-encoder", name=output_dir.split("/")[-1])

  train_script_path = os.path.join(output_dir, 'finetune_script.py')

  print("TRAIN SCRIPT PATH: ", train_script_path)

  copyfile(__file__, train_script_path)
  with open(train_script_path, 'a') as fOut:
    fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

  # saves top-K checkpoints based on "val_loss" metric
  checkpoint_callback = ModelCheckpoint(
      every_n_train_steps=25000,
      save_top_k=5,
      monitor="global_train_step",
      mode="max",
      dirpath=output_dir,
      filename="ckpt-{global_train_step}",
  )

  print("CHECKPOINT CALLBACK: ", checkpoint_callback)

  model = ListRankLoss(model_name=args.model)
  print("MODEL: ", model)

  # train_dataloader = dm.train_dataloader()
  # val_dataloader = dm.train_dataloader()
  print("DATALOADER OP: ", dm)

  trainer = Trainer(max_epochs=args.epochs,
                    accelerator="gpu",
                    devices=args.num_gpus,
                    precision=args.precision,
                    strategy=args.strategy,
                    default_root_dir=output_dir,
                    callbacks=[checkpoint_callback],
                    logger=wandb_logger)
  print("TRAINER: ", trainer)

  trainer.fit(model, datamodule=dm)

  #Save final HF model
  final_path = os.path.join(output_dir, "final")
  print("FINAL PATH: ", final_path)
  dm.tokenizer.save_pretrained(final_path)
  model.model.save_pretrained(final_path)


def eval(args):
  import ir_datasets

  model = ListRankLoss.load_from_checkpoint(args.ckpt)
  hf_model = model.model.cuda()
  tokenizer = AutoTokenizer.from_pretrained(model.hparams.model_name)

  dev_qids = set()

  dev_queries = {}
  dev_rel_docs = {}
  needed_pids = set()
  needed_qids = set()

  corpus = {}
  retrieved_docs = {}

  dataset = ir_datasets.load("msmarco-passage/dev/small")
  for query in dataset.queries_iter():
    dev_qids.add(query.query_id)

  with open('data/qrels.dev.tsv') as fIn:
    for line in fIn:
      qid, _, pid, _ = line.strip().split('\t')

      if qid not in dev_qids:
        continue

      if qid not in dev_rel_docs:
        dev_rel_docs[qid] = set()
      dev_rel_docs[qid].add(pid)

      retrieved_docs[qid] = set()
      needed_qids.add(qid)
      needed_pids.add(pid)

  for query in dataset.queries_iter():
    qid = query.query_id
    if qid in needed_qids:
      dev_queries[qid] = query.text

  with open('data/top1000.dev', 'rt') as fIn:
    for line in fIn:
      qid, pid, query, passage = line.strip().split("\t")
      corpus[pid] = passage
      retrieved_docs[qid].add(pid)

  ## Run evaluator
  print("Queries: {}".format(len(dev_queries)))

  mrr_scores = []
  hf_model.eval()

  with torch.no_grad():
    for qid in tqdm.tqdm(dev_queries, total=len(dev_queries)):
      query = dev_queries[qid]
      top_pids = list(retrieved_docs[qid])
      cross_inp = [[query, corpus[pid]] for pid in top_pids]

      encoded = tokenizer(cross_inp, padding=True, truncation=True, return_tensors="pt").to('cuda')
      output = model(**encoded)
      bert_score = output.logits.detach().cpu().numpy()
      bert_score = np.squeeze(bert_score)

      argsort = np.argsort(-bert_score)

      rank_score = 0
      for rank, idx in enumerate(argsort[0:10]):
        pid = top_pids[idx]
        if pid in dev_rel_docs[qid]:
          rank_score = 1 / (rank + 1)
          break

      mrr_scores.append(rank_score)

      if len(mrr_scores) % 10 == 0:
        print("{} MRR@10: {:.2f}".format(len(mrr_scores), 100 * np.mean(mrr_scores)))

  print("MRR@10: {:.2f}".format(np.mean(mrr_scores) * 100))


if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument("--num_gpus", type=int, default=1)
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--strategy", default=None)
  parser.add_argument("--model", default='facebook/contriever-msmarco')
  parser.add_argument("--eval", action="store_true")
  parser.add_argument("--ckpt")
  parser.add_argument("--cross_lingual_chance", type=float, default=0.0)
  parser.add_argument("--precision", type=int, default=16)
  parser.add_argument("--num_negs", type=int, default=3)
  parser.add_argument(
      "--langs", nargs="+", default=['english']
  )  #, 'chinese', 'french', 'german', 'indonesian', 'italian', 'portuguese', 'russian', 'spanish', 'arabic', 'dutch', 'hindi', 'japanese', 'vietnamese'

  args = parser.parse_args()

  if args.eval:
    eval(args)
  else:
    print("MAIN")
    main(args)
