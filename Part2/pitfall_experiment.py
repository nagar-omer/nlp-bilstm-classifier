import pickle
import sys
sys.path.insert(0, "..")

from Part2.part2_pitfall_data import PitfallGenExamples
from utils.part2_pitfall_data_loaders import Part2DataLoader
from utils.part1_activator import Part1ModelActivator


if __name__ == "__main__":
    args = sys.argv

    pitfall_type = "long_end"
    # pitfall_type = "length"
    # pitfall_type = "order"
    epochs = 100
    lr = 0.1
    if len(args) > 1:
        pitfall_type = int(args[1])
    if len(args) > 2:
        epochs = int(args[2])
    if len(args) > 3:
        lr = float(args[3])

    train_size = 0.8
    train_size = int(train_size * 1000)
    dev_size = 1000 - train_size
    # create data

    train = PitfallGenExamples(total_size=train_size, to_shuffle=False, pitfall_type=pitfall_type)
    dev = PitfallGenExamples(total_size=dev_size, to_shuffle=False, pitfall_type=pitfall_type)
    dl_train = Part2DataLoader(train)
    dl_dev = Part2DataLoader(dev)

    lstm_part1 = Part1ModelActivator(dl_train, dl_dev, lr=lr)
    loss_vec_dev, accuracy_vec_dev, loss_vec_train, accuracy_vec_train = lstm_part1.train(epochs)
    pickle.dump((loss_vec_dev, accuracy_vec_dev, loss_vec_train, accuracy_vec_train), open(pitfall_type + "_res.pkl", "wb"))
