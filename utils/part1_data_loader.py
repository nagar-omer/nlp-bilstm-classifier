import torch

from torch.utils.data import Dataset
from Part1.gen_examples import LangGenExamples
UNKNOWN = "UUNNKK"
START = "<s>"
END = "</s>"


class Part1DataLoader(Dataset):
    def __init__(self, size=1000):
        self._len = size
        self._vocab = self._build_vocab()
        self._lang = LangGenExamples(total_size=size, to_shuffle=True)
        self._data = self._lang.examples

    def _build_vocab(self):
        v = {}
        for i in range(9):
            v[str(i+1)] = i
        v["a"] = 9
        v["b"] = 10
        v["c"] = 11
        v["d"] = 12
        v[UNKNOWN] = 13
        v[START] = 14
        v[END] = 15

        return v

    def len_vocab(self):
        return len(self._vocab)

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        label, example = self._data[item]
        vec = []
        for letter in example:
            vec.append(self._vocab[letter])
        vec = [self._vocab[START]] + vec + [self._vocab[END]]
        return torch.Tensor([label]),  torch.Tensor(vec).long()


if __name__ == "__main__":
    dl = Part1DataLoader()
    print(dl._vocab)
    print(dl.len_vocab())
    print(dl.__getitem__(0))
    print(dl.__getitem__(2))
    print(dl.__getitem__(999))
    print([(label, data) for batch_index, (label, data) in enumerate(dl)])

