import os
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from models.JointBert_model import JointBERT
from dataloader import Features, load_dataset


class JointTrainer:
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, dataset="SNIPS"):
        self.args = args

        # datasets
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.dataset = dataset

        # labels
        self.intent_labels = [label.strip() for label in
                              open(f"./data/{dataset}/intent_labels.txt", 'r', encoding='utf-8')]
        self.slot_labels = [label.strip() for label in
                            open(f"./data/{dataset}/slot_labels.txt", 'r', encoding='utf-8')]

        # model and configuration
        self.config_type = BertConfig
        self.config = self.config_type.from_pretrained('bert-base-uncased', finetuning_task=dataset)

        self.model_type = JointBERT
        self.model = self.model_type.from_pretrained('bert-base-uncased',
                                                     config=self.config,
                                                     intent_labels=self.intent_labels,
                                                     slot_labels=self.slot_labels)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"device: {self.device}")

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler,
                                      batch_size=32)  # batch size to be tuned
        gradient_accumulation_steps = 1  # to be tuned
        train_epochs = 10.0  # to be tuned
        t_steps = len(train_dataloader) // gradient_accumulation_steps * train_epochs

        # linear warm-up and decay ( preparing Optimizer and scheduler)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_params = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},  # to be tuned, now is useless
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}  # NOT TO BE TUNED
        ]

        lr, eps = 5e-5, 1e-8  # learning rate and eps to be tuned
        optimizer = AdamW(optimizer_params, lr=lr, eps=eps)
        warmup_steps = 0  # to be tuned
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_steps)

        # training phase

        print("start training phase")

        global_steps = 0
        train_loss = 0.0
        self.model.zero_grad()

        train_iter = trange(int(train_epochs), desc="Epoch")

        for _ in train_iter:
            epoch_iter = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iter):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # moving the tensors of the dataset to the proper
                # device

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}
                outputs = self.model(**inputs)

                loss = outputs[0]

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                train_loss += loss.item()
                max_grad_norm = 1  # to be tuned
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_steps += 1

                    if global_steps % 200 == 0:
                        self.save_model()

        return global_steps, train_loss / global_steps

    def save_model(self):
        if not os.path.exists("./trained_models/"):
            os.makedirs("./trained_models/", exist_ok=True)
        if not os.path.exists(f"./trained_models/{self.dataset}"):
            os.makedirs(f"./trained_models/{self.dataset}", exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.save_pretrained(f"./trained_models/{self.dataset}")

    def load_model(self):
        # check the existence of the model
        if not os.path.exists(f"./trained_models/{self.dataset}"):
            raise Exception(f"Train the model first, no pretrained model for {self.dataset}")

        try:

            self.model = self.model_type.from_pretrained(f"./trained_models/{self.dataset}",
                                                         intent_labels=self.intent_labels,
                                                         slot_labels=self.slot_labels)
            self.model.to(self.device)
            print(f"{self.dataset} model loaded succesfully")
        except:

            raise Exception(f"Something went wrong when loading {self.dataset} model")


if __name__ == '__main__':
    train_data = load_dataset()
    t = JointTrainer(args="", train_dataset=train_data)
