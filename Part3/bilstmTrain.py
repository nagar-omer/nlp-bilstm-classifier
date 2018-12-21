import pickle
import sys
sys.path.insert(0, "..")
from Part3.part3_bi_model import Part3RnnBiLSTMB, Part3RnnBiLSTMD, Part3RnnBiLSTMA, Part3RnnBiLSTMC
from utils.part3_activator import Part3Activator
from utils.part3_data_loader import Part3DataLoader, Part3LetterDataLoader
import os


def _get_data_loader(repr_, src_file):
    if repr_ in ["a", "c", "d"]:
        return Part3DataLoader(src_file, vocab=os.path.join("..", "data", "word_embed", "embed_map"))
    elif repr_ in ["b"]:
        return Part3LetterDataLoader(src_file)


def _get_model(repr_, train_dl):
    if repr_ == "a":
        return Part3RnnBiLSTMA((100, 60, train_dl.pos_dim), train_dl.vocab_size, embedding_dim=50, lr=0.01,
                               pre_trained=os.path.join("..", "data", "word_embed", "wordVectors"))
    if repr_ == "b":
        return Part3RnnBiLSTMB((50, 100, train_dl.pos_dim), vocab_size=128, embedding_dim=10, lr=0.01)
    if repr_ == "c":
        return Part3RnnBiLSTMC((100, 60, train_dl.pos_dim), train_dl.vocab_size, train_dl.vocabulary.len_pref()
                               , train_dl.vocabulary.len_suf(), embedding_dim=50, lr=0.01,
                               pre_trained=os.path.join("..", "data", "word_embed", "wordVectors"))
    if repr_ == "d":
        return Part3RnnBiLSTMD((100, 60, train_dl.pos_dim), train_dl.vocab_size, train_dl.vocabulary.len_pref()
                               , train_dl.vocabulary.len_suf(), embedding_dim=50, lr=0.01,
                               pre_trained=os.path.join("..", "data", "word_embed", "wordVectors"))


def train(repr_, train_file, model_name, dev_src=None):
    # create data_loader, model, and activator
    train_dl = _get_data_loader(repr_, train_file)
    dev_dl = _get_data_loader(repr_, train_file) if dev_src else None
    rnn_model = _get_model(repr_, train_dl)
    model_activator = Part3Activator(rnn_model, train_dl, dev_dl)

    # train model
    loss_vec_dev, accuracy_vec_dev, loss_vec_train, accuracy_vec_train = model_activator.train(5)
    pickle.dump((loss_vec_dev, accuracy_vec_dev, loss_vec_train, accuracy_vec_train),
                open("res_" + repr_ + "_" + model_name, "wb"))
    pickle.dump((model_activator, train_dl.vocabulary, train_dl.pos_map), open(model_name, "wb"))


if __name__ == "__main__":
    args = sys.argv
    file_name = args[0]
    repr_ = args[1]
    train_src = args[2]
    model_name = args[3]
    dev_src = None
    if len(args) > 4:
        dev_src = args[4]

    train(repr_, train_src, model_name, dev_src)
