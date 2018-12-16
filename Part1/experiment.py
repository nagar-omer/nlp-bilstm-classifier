import pickle
import sys
sys.path.insert(0, "..")
from utils.part1_data_loader import Part1DataLoader
from utils.part1_activator import Part1ModelActivator


if __name__ == "__main__":
    args = sys.argv
    epochs = 30
    lr = 0.1
    if len(args) > 1:
        epochs = int(args[1])
    if len(args) > 2:
        lr = float(args[2])

    train_size = 0.8
    train_size = int(train_size * 1000)
    dev_size = 1000 - train_size
    # create data
    dl_train = Part1DataLoader(size=train_size)
    dl_dev = Part1DataLoader(size=dev_size)

    lstm_part1 = Part1ModelActivator(dl_train, dl_dev, lr=lr)
    loss_vec_dev, accuracy_vec_dev, loss_vec_train, accuracy_vec_train = lstm_part1.train(epochs)
    pickle.dump((loss_vec_dev, accuracy_vec_dev, loss_vec_train, accuracy_vec_train),
                open("sigma_6_mue_3_res.pkl", "wb"))


