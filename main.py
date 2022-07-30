import training
from dataloader import load_dataset

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_data = load_dataset()
    dev_data = load_dataset(mode="valid")
    # test_data = load_dataset(mode = "test")
    t = training.JointTrainer(args="", train_dataset=train_data, dev_dataset=dev_data)
    t.load_model()
    t.eval("dev")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
