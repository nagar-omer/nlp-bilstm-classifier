from utils.part3_params import START, END, UNKNOWN, SUF, PREF, PAD
import numpy as np


class Vocabulary:
    def __init__(self, src_file, labeled=True, vocab_file=None):
        if vocab_file:
            self._vocab_to_idx, self._idx_to_vocab = self._read_map_file(vocab_file)
            junk, junk, self._pref_to_idx, self._idx_to_pref, self._suf_to_idx, \
                self._idx_to_suf = self._vocab_from_train(src_file, labeled)
        elif src_file:
            self._vocab_to_idx, self._idx_to_vocab, self._pref_to_idx, self._idx_to_pref, self._suf_to_idx, \
                self._idx_to_suf = self._vocab_from_train(src_file, labeled)

    def __len__(self):
        return len(self._idx_to_vocab)

    def len_suf(self):
        return len(self._idx_to_suf)

    def len_pref(self):
        return len(self._idx_to_pref)

    @staticmethod
    def _vocab_from_train(src_file, labeled):
        all_words = [PAD]
        all_suf = [PAD]
        all_pref = [PAD]

        src = open(src_file, "rt")
        for i, row in enumerate(src):
            if row == "\n":
                continue
            word, pos = row.split() if labeled else (row.strip(), None)
            all_words.append(word)
            if len(word) >= SUF:
                all_suf.append(word[-SUF:])
                all_pref.append(word[:PREF])

        # vocabulary words
        idx_to_vocab = [START, END, UNKNOWN] + [w.lower() for w in set(all_words)]
        vocab_to_idx = {word: i for i, word in enumerate(idx_to_vocab)}

        # vocabulary prefixes
        # idx_to_pref = [UNKNOWN] + [w.lower() for w, s in sorted(all_pref.items(), key=lambda x: -x[1])]
        idx_to_pref = [UNKNOWN] + [w.lower() for w in set(all_pref)]
        pref_to_idx = {pref: i for i, pref in enumerate(idx_to_pref)}

        # vocabulary suffixes
        # idx_to_suf = [UNKNOWN] + [w.lower() for w, s in sorted(all_suf.items(), key=lambda x: -x[1])]
        idx_to_suf = [UNKNOWN] + [w.lower() for w in set(all_suf)]
        suf_to_idx = {suf: i for i, suf in enumerate(idx_to_suf)}

        return vocab_to_idx, idx_to_vocab, pref_to_idx, idx_to_pref, suf_to_idx, idx_to_suf

    @staticmethod
    def _read_map_file(map_file):
        list_words = [PAD]
        map_file = open(map_file, "rt")
        for row in map_file:
            list_words.append(row.strip())
        word_to_idx = {word: i for i, word in enumerate(list_words)}
        return word_to_idx, list_words

    def pref_vocab(self, word):
        # case word
        if word.lower()[:PREF] in self._pref_to_idx:
            return self._pref_to_idx[word.lower()[:PREF]]  # case word is embedded

        return self._pref_to_idx[UNKNOWN]  # no matching word

    def suf_vocab(self, word):
        # case word
        if word.lower()[-SUF:] in self._suf_to_idx:
            return self._suf_to_idx[word.lower()[-SUF:]]  # case word is embedded

        return self._suf_to_idx[UNKNOWN]  # no matching word

    def vocab(self, word_idx):
        if type(word_idx) == int:
            return self._idx_to_vocab[word_idx]  # case int - idx

        word = word_idx.lower() if word_idx not in [START, END, UNKNOWN] else word_idx
        # case word
        if word in self._vocab_to_idx:
            return self._vocab_to_idx[word]  # case word is embedded

        return -1    # no matching word


if __name__ == "__main__":
    import os
    vc = Vocabulary(src_file=os.path.join("..", "data", "pos", "train"))
    e = 1
