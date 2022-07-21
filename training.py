import os
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from models.JointBert_model import JointBERT


class JointTrainer:
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, dataset="SNIPS"):
        self.args = args

        # datasets
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        # labels
        self.intent_labels = [label.strip() for label in
                              open(f"./data/{dataset}/intent_labels.txt", 'r', encoding='utf-8')]
        self.slot_labels = [label.strip() for label in
                            open(f"./data/{dataset}/slot_labels.txt", 'r', encoding='utf-8')]

        # model and configuration
        self.config = BertConfig
        self.config = self.config.from_pretrained('bert-base-uncased', finetuning_task=dataset)

        self.model = JointBERT
        self.model = self.model.from_pretrained('bert-base-uncased',
                                                config=self.config,
                                                intent_labels=self.intent_labels,
                                                slot_labels=self.slot_labels)


t = JointTrainer(args="")
