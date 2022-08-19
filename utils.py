import training
from dataloader import load_dataset
import base_training
from models.baseline_extended_model import ExtendedModel
import numpy as np

def full_evaluation(iter = 5):
    for dataset in ["ATIS","SNIPS"]:
        train_data = load_dataset(mode="train", dataset=dataset)
        dev_data = load_dataset(mode="valid", dataset=dataset)
        test_data = load_dataset(mode="test", dataset=dataset)
        res = {"slot f1":{} ,"intent accuracy":{}}
        for i in range(iter):
            t = base_training.BaselineTrainer(dataset=dataset)
            t.epoch_trainer(train_data=train_data, dev_data=dev_data)
            rs, ri, _ = t.eval(data=test_data)
            print(rs["weighted avg"]['f1-score'],"\n", ri["accuracy"])
            res["slot f1"][str(i)] = rs["weighted avg"]['f1-score']
            res["intent accuracy"][str(i)] = ri["accuracy"]
            res[str(i)] = {"slot f1":rs["weighted avg"]['f1-score'] ,"intent accuracy":ri["accuracy"]}

        slot_f1s = np.fromiter(res["slot f1"].values(), dtype=float)
        intent_acc = np.fromiter(res["intent accuracy"].values(), dtype=float)
        print('Slot F1', round(slot_f1s.mean(), 3), '+-', round(slot_f1s.std(), 3))
        print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))
        print(res)
        return

