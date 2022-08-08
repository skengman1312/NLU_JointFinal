import training
from dataloader import load_dataset
import base_training
from models.baseline_extended_model import ExtendedModel
import os
RunningInCOLAB = 'google.colab' in str(get_ipython()) if hasattr(__builtins__,'__IPYTHON__') else False

if "drive" in os.getcwd():
    print("Running in colab notebook")

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_data = load_dataset(dataset="ATIS")
    dev_data = load_dataset(mode="valid", dataset= "ATIS")
    test_data = load_dataset(mode = "test", dataset="ATIS")
    bt = base_training.BaselineTrainer(model_type= ExtendedModel ,dataset="ATIS")
    bt.epoch_trainer(train_data, dev_data)

    print("test eval:\n")
    print(bt.eval(train_data))
    # t = training.JointTrainer(args="", dataset="ATIS", train_dataset=train_data, dev_dataset=dev_data, test_dataset=test_data)
    # t.train()
    # t.eval("test")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
