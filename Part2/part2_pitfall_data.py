import random
import sys

import numpy as np


class PitfallGenExamples:
    def __init__(self, total_size=1000, to_shuffle=False, pitfall_type="long_end"):
        self._type = pitfall_type
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

    def _create_single_positive_long_end(self):
        return self._rand_seq(sigma=1, mue=0) + self._rand_seq(letter="a", sigma=1, mue=0) + \
               self._rand_seq(sigma=1, mue=0) + self._rand_seq(letter="b", sigma=1, mue=0) + \
               self._rand_seq(sigma=1, mue=0) + self._rand_seq(letter="c", sigma=1, mue=0) + \
               self._rand_seq(sigma=1, mue=0) + self._rand_seq(letter="d", sigma=1, mue=0) +\
               self._rand_seq(sigma=20, mue=0)

    def _create_single_negative_long_end(self):
        return self._rand_seq(sigma=1, mue=0) + self._rand_seq(letter="a", sigma=1, mue=0) + \
               self._rand_seq(sigma=1, mue=0) + self._rand_seq(letter="b", sigma=1, mue=0) + \
               self._rand_seq(sigma=1, mue=0) + self._rand_seq(letter="c", sigma=1, mue=0) + \
               self._rand_seq(sigma=1, mue=0) + self._rand_seq(letter="d", sigma=1, mue=0) +\
               self._rand_seq(sigma=20, mue=0)

    def _create_single_positive_length(self):
        return self._rand_seq(sigma=20, mue=0)

    def _create_single_negative_length(self):
        return self._rand_seq(sigma=21, mue=0)

    def _create_single_positive_order(self):
        out_str = self._rand_seq(letter="a", sigma=2, mue=1)
        for i in range(4):
            out_str += self._rand_seq(letter="b", sigma=2, mue=1) + self._rand_seq(letter="a", sigma=2, mue=1)
        return out_str

    def _create_single_negative_order(self):
        out_str = self._rand_seq(letter="b", sigma=2, mue=0)
        for i in range(4):
            out_str += self._rand_seq(letter="a", sigma=2, mue=1) + self._rand_seq(letter="b", sigma=2, mue=1)
        return out_str

    def _rand_seq(self, letter=None, sigma=5, mue=3):
        size = abs(int(np.random.normal(sigma, mue))) + 1
        if letter:
            return "".join([letter] * size)
        return "".join([str(i) for i in np.random.randint(1, 10, size=size)])

    def _build(self, size=1000):
        half = int(size / 2)
        for _ in range(half):
            if self._type == "long_end":
                self._positive_data.append((1, self._create_single_positive_long_end()))
                self.negative_examples.append((0, self._create_single_negative_long_end()))
            elif self._type == "length":
                self._positive_data.append((1, self._create_single_positive_length()))
                self.negative_examples.append((0, self._create_single_negative_length()))
            elif self._type == "order":
                self._positive_data.append((1, self._create_single_positive_order()))
                self.negative_examples.append((0, self._create_single_negative_order()))

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

    lang = PitfallGenExamples(total_size=size, to_shuffle=shuffle)
    lang.to_file()
    for label, e in lang.examples:
        print(label, e)
    e = 0