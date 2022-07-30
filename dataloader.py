import os
import logging
from transformers import BertTokenizer
import torch
import pandas as pd

from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class Example:

    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels


class Features:

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids


class JoinProcessor:

    def __init__(self, dataset: str):
        datapath = "./data/"
        self.dataset = dataset
        self.intent_labels = [label.strip() for label in
                              open(f"{datapath}{dataset}/intent_labels.txt", 'r', encoding='utf-8')]
        self.slot_labels = [label.strip() for label in
                            open(f"{datapath}{dataset}/slot_labels.txt", 'r', encoding='utf-8')]

    def read_examples(self, mode: str):
        """
        :param mode: "train" "test" or "valid"
        :return:
        """
        data = pd.read_json(f"./data/{self.dataset}/{mode}.json")
        # print(data)
        return self._make_example(data, mode)

    def _make_example(self, data, mode):
        examples = list()
        for l in data.iterrows():
            # 1 guid
            guid = f"{l[0]}-{mode}"
            # 2 input text
            words = l[1]["utterance"].split()
            # 3 intent label
            intent = l[1]["intent"]
            intent_label = self.intent_labels.index(
                intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            # 4 slots labels
            slots_labels = [self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK") for s
                            in l[1]["slots"].split()]

            assert len(words) == len(slots_labels)
            examples.append(Example(guid, words, intent_label, slots_labels))
        return examples


def convert_examples_to_features(examples, max_seq_len=50, tokenizer=BertTokenizer, pad_token_label_id=0,
                                 cls_token_segment_id=0, pad_token_segment_id=0, sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id
    # print(cls_token, sep_token, unk_token, pad_token_id)
    features = []
    tokenizer = tokenizer.from_pretrained('bert-base-uncased')
    for (ex_index, example) in enumerate(examples):

        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(text=word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(
            len(slot_labels_ids), max_seq_len)

        intent_label_id = int(example.intent_label)

        # print(tokens, input_ids, slot_labels_ids)
        features.append(
            Features(input_ids=input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids,
                     intent_label_id=intent_label_id,
                     slot_labels_ids=slot_labels_ids
                     ))
    return features


def load_dataset(mode="train", dataset ="SNIPS"):
    if not os.path.exists("cache"):
        print("cache directoy is created")
        os.makedirs("cache")

    if os.path.exists(f"./cache/cache_{mode}_{dataset}"):

        features = torch.load(f"./cache/cache_{mode}_{dataset}")

    else:
        examples = JoinProcessor(dataset=dataset).read_examples(mode)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        pad_token_label_id = 0
        features = convert_examples_to_features(examples, tokenizer=tokenizer,
                                                pad_token_label_id=pad_token_label_id)

        torch.save(features, f"./cache/cache_{mode}_{dataset}")

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_intent_label_ids, all_slot_labels_ids)
    return dataset




#load_dataset()
# jp = JoinProcessor("SNIPS")
# data = jp.read_examples("train")
# convert_examples_to_features(data)
