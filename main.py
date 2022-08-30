import training
from dataloader import load_dataset
import base_training
from models.baseline_extended_model import ExtendedModel
from utils import full_evaluation
import os
import argparse


RunningInCOLAB = 'google.colab' in str(get_ipython()) if hasattr(__builtins__, '__IPYTHON__') else False

if "drive" in os.getcwd():
    print("Running in colab notebook")


def main(args):
    """
    main parsing function
    """
    model_trainer_map = {"JointBert": training.JointTrainer,
                         "baseline": base_training.BaselineTrainer,
                         "extended_baseline": base_training.BaselineTrainer}
    if args.full_evaluation == True:
        print("Full eval started")
        if args.model_type != "JointBert":
            full_evaluation(modeltype=args.model_type)
        else:
            full_evaluation()
        return

    train_data = load_dataset(mode="train", dataset=args.dataset) if args.train else None
    dev_data = load_dataset(mode="valid", dataset=args.dataset) if args.train else None
    test_data = load_dataset(mode="test", dataset=args.dataset) if args.test else None

    train_handler = model_trainer_map[args.model_type]

    if args.model_type == "JointBert":
        t = train_handler("", train_dataset=train_data, dev_dataset=dev_data, test_dataset=test_data,
                          dataset=args.dataset)
        if args.train:
            t.train()
        if args.test:
            t.eval(mode="test")
        return
    elif args.model_type == "baseline":
        t = train_handler(dataset=args.dataset)
    elif args.model_type == "extended_baseline":
        t = train_handler(model_type =ExtendedModel ,  dataset=args.dataset)
    else:
        raise Exception("Ivalid model type choosen, available model types are JointBert, baseline and extended_baseline")
    if args.train:
        t.epoch_trainer(train_data=train_data, dev_data=dev_data)
    if args.test:
        if args.model_type == "JointBert":
            rs, ri, _ = t.eval(data=test_data, dict_out=False)
        else:
            rs, ri, _, __ = t.eval(data=test_data, dict_out=False)
        print(rs)
        print(ri)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="SNIPS", type=str, help="The name of the dataset to train on")
    parser.add_argument("--model_type", default="JointBert", type=str, help="Type of model to use fo the task")
    parser.add_argument("--full_evaluation", default=False, action='store_true', help="Whether or not to "
                                                                                      "perform training "
                                                                                      "and evaluation of "
                                                                                      "all three models, "
                                                                                      "this argument "
                                                                                      "overrides all the "
                                                                                      "others ")
    parser.add_argument("--train", default=True, type=bool,
                        help="Whether to train or not the model, if false there must already be a pretrained model")
    parser.add_argument("--test", default=True, type=bool, help="Whether to test or not the model")

    args = parser.parse_args()
    main(args)

    # train_data = load_dataset(dataset="ATIS")
    # dev_data = load_dataset(mode="valid", dataset= "ATIS")
    # test_data = load_dataset(mode = "test", dataset="ATIS")
    # bt = base_training.BaselineTrainer(model_type= ExtendedModel ,dataset="ATIS")
    # bt.epoch_trainer(train_data, dev_data)
    #
    # print("test eval:\n")
    # print(bt.eval(train_data))
    # t = training.JointTrainer(args="", dataset="ATIS", train_dataset=train_data, dev_dataset=dev_data, test_dataset=test_data)
    # t.train()
    # t.eval("test")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
