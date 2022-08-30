import pandas as pd
import torch.optim as optim
import torch.nn as nn
from models.baseline_model import *
from models.baseline_extended_model import *
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
import os
from tqdm import tqdm, trange
import matplotlib.pyplot as plt


# if "drive" in os.getcwd():
#     from tqdm.notebook import tqdm, trange
# else:


# print(f"Running on colab: {'drive' in os.getcwd()}")

class BaselineTrainer:

    def __init__(self, model_type = ModelIAS, dataset="SNIPS"):
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

        self.model = model_type(self.hid_size, self.out_slot, self.out_int, self.emb_size, self.vocab_len,
                              pad_index=PAD_TOKEN).to(self.device)
        self.model.apply(init_weights)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        self.criterion_intents = nn.CrossEntropyLoss()  # Because we do not have the pad token

    def train(self, data):
        self.model.train()
        loss_array = []
        intent_loss_array = []
        slot_loss_array = []
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
            intent_loss_array.append(loss_intent.item())
            slot_loss_array.append(loss_slot.item())
            loss = loss_intent + loss_slot  # In joint training we sum the losses.
            # Is there another way to do that?
            loss_array.append(loss.item())
            loss.backward()  # Compute the gradient, deleting the computational graph
            # clip the gradient to avoid explosioning gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            self.optimizer.step()  # Update the weights

        return loss_array, (intent_loss_array, slot_loss_array)

    def eval(self, data, dict_out = True):
        self.model.eval()
        loss_array = []
        intent_loss_array = []
        slot_loss_array = []

        ref_intents = []
        hyp_intents = []

        ref_slots = []
        hyp_slots = []
        # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
        dataloader = DataLoader(dataset=data, batch_size=32)
        eval_iter = tqdm(dataloader, position=0, desc="Evaluation", leave=True)
        with torch.no_grad():  # It used to avoid the creation of computational graph
            for sample in eval_iter:
                sample = tuple(t.to(self.device) for t in sample)
                length = torch.LongTensor(
                    [t.count_nonzero() for t in sample[0]])  # adaptation of the model seen in class
                max_slot_len = int(
                    torch.max(length))  # needed to slice away the excessive padding in the target slots
                slots, intent = self.model(sample[0], length)
                loss_intent = self.criterion_intents(intent, sample[3])
                loss_slot = self.criterion_slots(slots, sample[4][:, :max_slot_len])
                loss = loss_intent + loss_slot # In joint training we sum the losses.
                intent_loss_array.append(loss_intent.item())
                slot_loss_array.append(loss_slot.item())

                loss_array.append(loss.item())
                # Intent inference
                # Get the highest probable class
                out_intents = [self.intent_labels[x]
                               for x in torch.argmax(intent, dim=1).tolist()]
                gt_intents = [self.intent_labels[x] for x in sample[3].tolist()]
                ref_intents.extend(gt_intents)
                hyp_intents.extend(out_intents)

                # Slot inference
                output_slots = torch.argmax(slots, dim=1)
                for id_seq, seq in enumerate(output_slots):

                    gt_ids = sample[4][id_seq].tolist()
                    msk = sample[4][id_seq].ne(0)

                    gt_slots = [self.slot_labels[elem] for elem in sample[4][id_seq][msk].tolist()] # fix in case delete

                    to_decode = seq[msk[:len(seq)]].tolist() # fix in case delete
                    ref_slots.append([elem for elem in gt_slots])
                    tmp_seq = []
                    for id_el, elem in enumerate(to_decode):
                        tmp_seq.append(self.slot_labels[elem])
                    hyp_slots.append(tmp_seq)

        flat_ref_slots = [s for u in ref_slots for s in u]  # flattening out for the scikit classification report
        flat_hyp_slots = [s for u in hyp_slots for s in u]

        report_slots = classification_report(flat_ref_slots, flat_hyp_slots, zero_division=False, output_dict=dict_out)

        report_intent = classification_report(ref_intents, hyp_intents,
                                              zero_division=False, output_dict=dict_out)
        return report_slots, report_intent, loss_array, (intent_loss_array, slot_loss_array)

    def epoch_trainer(self, train_data, dev_data):
        n_epochs = 200
        patience = 5
        losses_train = []
        losses_train_intent = []
        losses_train_slot = []

        losses_dev = []
        losses_dev_intent = []
        losses_dev_slot = []
        sampled_epochs = []
        best_f1 = 0
        for x in trange(1, n_epochs, position = 0, desc = "Epoch", leave=True,  colour='green'):
            loss, (intent_loss, slot_loss) = self.train(train_data)
            if x % 2 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                losses_train_intent.append(np.asarray(intent_loss).mean())
                losses_train_slot.append(np.asarray(slot_loss).mean())

                results_dev, intent_res, loss_dev, (loss_dev_intent, loss_dev_slot) = self.eval(dev_data)
                losses_dev.append(np.asarray(loss_dev).mean())
                losses_dev_intent.append(np.asarray(loss_dev_intent).mean())
                losses_dev_slot.append(np.asarray(loss_dev_slot).mean())

                print(results_dev['weighted avg'])
                f1 = results_dev['weighted avg']['f1-score']

                if f1 > best_f1:
                    best_f1 = f1
                else:
                    patience -= 1
                if patience <= 0:  # Early stoping with patient
                    print("patience is over")
                    break  # Not nice but it keeps the code clean


        results_test, intent_test, _, __ = self.eval(dev_data)
        print(results_test,"\n______\n", intent_test)



        losses = pd.DataFrame({"train loss":losses_train, "dev loss": losses_dev})
        intent_losses = pd.DataFrame({"train loss":losses_train_intent, "dev loss": losses_dev_intent})
        slot_loss = pd.DataFrame({"train loss": losses_train_slot, "dev loss": losses_dev_slot})
        print(losses)
        losses.plot(use_index=True, y=["train loss", "dev loss"], kind="line")
        losses.to_csv(f"losses.csv")
        plt.savefig("losses.png")
        plt.show()

        intent_losses.plot(use_index=True, y=["train loss", "dev loss"], kind="line")
        plt.savefig("intent_losses.png")
        plt.show()

        slot_loss.plot(use_index=True, y=["train loss", "dev loss"], kind="line")
        plt.savefig("slot_losses.png")
        plt.show()

        return losses_train, losses_dev
        #intent_acc.append(intent_test['accuracy'])
        #slot_f1s.append(results_test['weighted_avg']['f1-score'])

