import training
from dataloader import load_dataset

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train = load_dataset()
    tr = training.JointTrainer("", train_dataset= train)
    tr.train()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
