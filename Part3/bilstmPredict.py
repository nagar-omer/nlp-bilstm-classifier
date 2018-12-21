import os
import pickle
import sys
sys.path.insert(0, "..")
from utils.part1_data_loader import END
from utils.part3_data_loader import Part3DataLoader, Part3LetterDataLoader


def _get_data_loader(repr_, src_file):
    if repr_ in ["a", "c", "d"]:
        return Part3DataLoader(src_file, vocab=os.path.join("..", "data", "word_embed", "embed_map"), labeled=False)
    elif repr_ in ["b"]:
        return Part3LetterDataLoader(src_file, labeled=False)


def _predict_existing_model(repr_, model_src, test_src):
    model_activator, vocabulary, pos_map = pickle.load(open(model_src, "rb"))

    dl_test = _get_data_loader(repr_, test_src)
    dl_test.load_pos_map(pos_map)
    res = model_activator.predict(dl_test)
    create_out_file(res, test_src + ".pred")


def create_out_file(res, file_name):
    out_file = open("test4.pos", "wt")
    for word, pos in res:
        if word == END:
            continue
        out_file.write(word + " " + pos + "\n")
    out_file.close()


if __name__ == "__main__":
    args = sys.argv
    file_name = args[0]
    repr_ = args[1]
    model_src = args[2]
    test_src = args[3]

    _predict_existing_model(repr_, model_src, test_src)
