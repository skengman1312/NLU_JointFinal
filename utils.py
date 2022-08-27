import training
from dataloader import load_dataset
import base_training
from models.baseline_extended_model import ExtendedModel
from models.baseline_model import ModelIAS
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def full_evaluation(iter = 5, modeltype ="baseline"):
    modelmap = {"baseline": ModelIAS, "extended_baseline": ExtendedModel}
    for dataset in ["ATIS","SNIPS"]:
        train_data = load_dataset(mode="train", dataset=dataset)
        dev_data = load_dataset(mode="valid", dataset=dataset)
        test_data = load_dataset(mode="test", dataset=dataset)
        res = {"slot f1":{} ,"intent accuracy":{}}
        for i in range(iter):
            t = base_training.BaselineTrainer(dataset=dataset, model_type=modelmap[modeltype])
            t.epoch_trainer(train_data=train_data, dev_data=dev_data)
            rs, ri, _ = t.eval(data=test_data)
            print(rs["weighted avg"]['f1-score'],"\n", ri["accuracy"])
            res["slot f1"][str(i)] = rs["weighted avg"]['f1-score']
            res["intent accuracy"][str(i)] = ri["accuracy"]
            res[str(i)] = {"slot f1":rs["weighted avg"]['f1-score'] ,"intent accuracy":ri["accuracy"]}

        slot_f1s = np.fromiter(res["slot f1"].values(), dtype=float)
        intent_acc = np.fromiter(res["intent accuracy"].values(), dtype=float)
        now = datetime.now()
        slotline = f"Slot F1 {round(slot_f1s.mean(), 3)} +- {round(slot_f1s.std(), 3)} {now}\n"
        intentline = f"Intent Accuracy {round(intent_acc.mean(), 3)} +- {round(slot_f1s.std(), 3)} {now}\n"
        with open("full_evaluation_result.txt","a")as f:
            f.write(f"{dataset}--{modeltype}\n")
            f.write(slotline)
            f.write(intentline)
        print(slotline)
        print(intentline)
        print(res)

    return


def dataset_label_summary(dataset="SNIPS"):
    if dataset == "ATIS":
        data = pd.read_json("./data/ATIS/train.json")
        train, dev = train_test_split(data, test_size=0.12, random_state=1312)  # we set a random state in order
        # to have alwasy the same split
        test = pd.read_json("./data/ATIS/test.json")
    elif dataset == "SNIPS":
        train = pd.read_json("./data/SNIPS/train.json")
        test = pd.read_json("./data/SNIPS/test.json")
        dev = pd.read_json("./data/SNIPS/valid.json")
    else:
        raise Exception(f"Wrong dataset name: {dataset}\nOnly ATIS and SNIPS are allowed")

    intent_info = pd.DataFrame({"train": train["intent"].value_counts(normalize=True),
                                "test": test["intent"].value_counts(normalize=True),
                                "dev": dev["intent"].value_counts(normalize=True)
                                })
    print(intent_info)
    print(intent_info["train"].argmax())
    print(intent_info["train"][ intent_info["train"].isna()])

    intent_info.plot( use_index=True, y=["train", "test", "dev"], kind="bar")
    plt.ylabel("percentage")
    plt.tight_layout()
    plt.show()
    #print(train["slots"].str.split())
    #print(pd.DataFrame([l for u in train["slots"].str.split() for l in u]).value_counts(normalize=True))
    slots_info = pd.DataFrame({"train": pd.DataFrame([l for u in train["slots"].str.split() for l in u])
                              .replace(to_replace = "O", value = np.nan).value_counts(normalize=True),
                               "test": pd.DataFrame([l for u in test["slots"].str.split() for l in u])
                              .replace(to_replace = "O", value = np.nan).value_counts(normalize=True),
                               "dev": pd.DataFrame([l for u in dev["slots"].str.split() for l in u])
                              .replace(to_replace = "O", value = np.nan).value_counts(normalize=True)

    })
    print(slots_info)
    slots_info.plot(use_index=True, y=["train", "test", "dev"], kind="bar")
    plt.ylabel("percentage")
    #plt.tight_layout()
    #plt.xlabel("")
    plt.show()

    print(slots_info.fillna(0).std(axis = 1).sort_values())

def dataset_utterance_summary():
    common_path, datasets = "./data/",  ["SNIPS", "ATIS"]
    all_utterances_lenght = dict()
    for d in datasets:
        dfs = [pd.read_json(f) for f in os.scandir(common_path+d) if f.path.endswith(".json")]

        full_dataset = pd.concat(dfs, ignore_index=True)
        print(full_dataset["utterance"].str.split())
        utt_len = pd.DataFrame([len(u) for u in full_dataset["utterance"].str.split()])
        print(full_dataset["utterance"][ utt_len[0].argmin()])
        all_utterances_lenght[d] = utt_len[0]
    print(all_utterances_lenght)
    all_utt_df = pd.DataFrame(all_utterances_lenght)
    all_utt_df.plot(kind="box", title = "Utterance lenght distrubution")
    plt.show()
    print(all_utt_df)



