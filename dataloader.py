import os

import pandas as pd


class Example:

    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels


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
        :param mode: "train" "test" or "dev"
        :return:
        """
        data = pd.read_json(f"./data/{self.dataset}/{mode}.json")
        #print(data)
        self.make_example(data, mode)

    def make_example(self, data, mode):
        examples = list()
        for l in data.iterrows():
            #1 guid
            guid = f"{l[0]}-{mode}"
            #2 input text
            words = l[1]["utterance"].split()
            #3 intent label
            intent =  l[1]["intent"]
            intent_label = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            #4 slots labels
            slots_labels = [ self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK") for s in l[1]["slots"].split()]

            assert len(words) == len(slots_labels)
            examples.append(Example(guid, words, intent_label, slots_labels ))
        return examples




def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                     pad_token_label_id=-100,
                                     cls_token_segment_id=0,
                                     pad_token_segment_id=0,
                                     sequence_a_segment_id=0,
                                     mask_padding_with_zero=True):




jp = JoinProcessor("SNIPS")
jp.read_examples("train")
