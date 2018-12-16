import random
import sys
import numpy as np
SIGMA = 4
MUE = 3


class LangGenExamples:
    def __init__(self, total_size=1000, to_shuffle=False):
        self._positive_data = []
        self._negative_data = []
        self._build(size=total_size)
        self._shuffle = to_shuffle

    @property
    def positive_examples(self):
        return self._positive_data

    @property
    def negative_examples(self):
        return self._negative_data

    @property
    def examples(self):
        data = self._positive_data + self._negative_data
        if self._shuffle:
            random.shuffle(data)
        return data

    def _create_single_positive_examples(self):
        return self._rand_seq() + self._rand_seq(letter="a") + \
               self._rand_seq() + self._rand_seq(letter="b") + \
               self._rand_seq() + self._rand_seq(letter="c") + \
               self._rand_seq() + self._rand_seq(letter="d") +\
               self._rand_seq()

    def _create_single_negative_examples(self):
        return self._rand_seq() + self._rand_seq(letter="a") + \
               self._rand_seq() + self._rand_seq(letter="c") + \
               self._rand_seq() + self._rand_seq(letter="b") + \
               self._rand_seq() + self._rand_seq(letter="d") +\
               self._rand_seq()

    def _rand_seq(self, letter=None):
        size = abs(int(np.random.normal(SIGMA, MUE))) + 1
        if letter:
            return "".join([letter] * size)
        return "".join([str(i) for i in np.random.randint(1, 10, size=size)])

    def _build(self, size=1000):
        half = int(size / 2)
        for _ in range(half):
            self._positive_data.append((1, self._create_single_positive_examples()))
            self.negative_examples.append((0, self._create_single_negative_examples()))

    def to_file(self, neg_out_name="neg_examples", pos_out_name="pos_examples"):
        out_file = open(neg_out_name, "wt")
        for label, example in self._negative_data:
            out_file.write(str(example) + "\n")
        out_file.close()
        out_file = open(pos_out_name, "wt")
        for label, example in self._positive_data:
            out_file.write(str(example) + "\n")
        out_file.close()


if __name__ == "__main__":
    args = sys.argv
    size = 1000
    shuffle = False
    if "-s" in args:
        shuffle = True
        args.remove("-s")
    if len(args) > 1:
        size = int(args[1])

    lang = LangGenExamples(total_size=size, to_shuffle=shuffle)
    lang.to_file()
    for label, e in lang.examples:
        print(label, e)
    e = 0
