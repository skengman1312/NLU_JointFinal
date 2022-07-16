import pandas as pd
import logging
logger = logging.getLogger(__name__)

def vocab_processor(dataset: str):
    """

    :param dataset: name of the dataset. either ATIS or SNIP
    :return:
    """
    dirpath = f"./data/{dataset.upper()}/"
    traindf = pd.read_json(dirpath + "train.json")
    intents = traindf["intent"].drop_duplicates()
    slots = traindf["slots"]
    slots = pd.DataFrame([s for line in slots for s in line.split()]).drop_duplicates()
    slots = pd.concat([pd.DataFrame(["UNK", "PAD"]), slots])
    intents = pd.concat([pd.DataFrame(["UNK"]), intents])
    slots.to_csv(dirpath + "slot_labels.txt", header= False, index= False)
    intents.to_csv(dirpath + "intent_labels.txt", header= False, index= False)
    logger.info("done")
    #print(slots)





if __name__ == '__main__':
    vocab_processor("ATIS")
    vocab_processor("SNIPS")
    print("yoo")