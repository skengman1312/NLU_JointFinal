import training
from dataloader import load_dataset

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_data = load_dataset(dataset="ATIS")
    dev_data = load_dataset(mode="valid", dataset= "ATIS")
    test_data = load_dataset(mode = "test", dataset="ATIS")
    t = training.JointTrainer(args="", dataset="ATIS", train_dataset=train_data, dev_dataset=dev_data, test_dataset=test_data)
    t.train()
    t.eval("test")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
