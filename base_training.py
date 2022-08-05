import torch.optim as optim
import torch.nn as nn
from models.baseline_model import *
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

class BaselineTrainer:

    def __init__(self, dataset="SNIPS"):
        self.hid_size = 200
        self.emb_size = 300
        PAD_TOKEN = 0

        self.lr = 0.0001  # learning rate
        self.clip = 5  # Clip the gradient

        # labels
        self.intent_labels = [label.strip() for label in
                              open(f"./data/{dataset}/intent_labels.txt", 'r', encoding='utf-8')]
        self.slot_labels = [label.strip() for label in
                            open(f"./data/{dataset}/slot_labels.txt", 'r', encoding='utf-8')]

        self.out_slot = len(self.slot_labels)
        self.out_int = len(self.intent_labels)
        self.vocab_len = 30522  # bert-base-uncased tokenizer vocab len
        # BertTokenizer.from_pretrained('bert-base-uncased').vocab_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {self.device}")

        self.model = ModelIAS(self.hid_size, self.out_slot, self.out_int, self.emb_size, self.vocab_len,
                              pad_index=PAD_TOKEN).to(self.device)
        self.model.apply(init_weights)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        self.criterion_intents = nn.CrossEntropyLoss()  # Because we do not have the pad token

    def train(self, data):
        self.model.train()
        loss_array = []
        dataloader = DataLoader(dataset=data, batch_size=32)
        epoch_iter = tqdm(dataloader, desc="Iteration")
        for sample in epoch_iter:
            sample = tuple(t.to(self.device) for t in sample)
            self.optimizer.zero_grad()  # Zeroing the gradient
            slots_len = torch.LongTensor([t.count_nonzero() for t in sample[0]]) # adaptation of the model seen in class
            max_slot_len = int(torch.max(slots_len)) # needed to slice away the excessive padding in the target slots
            slots, intent = self.model(sample[0], slots_len)
            loss_intent = self.criterion_intents(intent, sample[3])
            loss_slot = self.criterion_slots(slots, sample[4][:,:max_slot_len])
            loss = loss_intent + loss_slot  # In joint training we sum the losses.
            # Is there another way to do that?
            loss_array.append(loss.item())
            loss.backward()  # Compute the gradient, deleting the computational graph
            # clip the gradient to avoid explosioning gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            self.optimizer.step()  # Update the weights

        return loss_array
