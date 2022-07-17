class Trainer:
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, dataset = "SNIPS"):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset


        self.intent_labels = [label.strip() for label in
                              open(f"./data/{dataset}/intent_labels.txt", 'r', encoding='utf-8')]
        self.slot_labels = [label.strip() for label in
                            open(f"./data/{dataset}/slot_labels.txt", 'r', encoding='utf-8')]
        
    pass
