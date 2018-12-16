from sys import stdout
from torch.utils.data import Dataset, DataLoader
import os
import torch
from utils.part3_params import START, END, UNKNOWN
from utils.part3_vocab import Vocabulary


class Part3DataLoader(Dataset):
    # src_file:         data source file - each row consist of <word POS> or <\n> for end of sentence
    # embed_map_file:   file withe ordered words separated by \n
    def __init__(self, src_file, vocab=None, labeled=True):                                                          # fix 5
        self._labeled = labeled
        if not vocab:                                                               # vocabulary for any word
            self._vocab = Vocabulary(src_file, labeled=True)
        else:
            self._vocab = Vocabulary(src_file, vocab_file=vocab) if type(vocab) == str else vocab  # learn prob's for unknowns
        self._data, self._idx_to_pos = self._read_file(src_file, labeled=labeled)                    # read file to structured data
        self._pos_to_idx = {pos: i for i, pos in enumerate(self._idx_to_pos)}       # pos to index

    def __len__(self):
        return len(self._data)

    def pos_to_idx(self, pos):
        if pos not in self._pos_to_idx:
            return -1
        return self._pos_to_idx[pos]

    def idx_to_pos(self, idx):
        return self._idx_to_pos[idx]

    @property
    def vocabulary(self):
        return self._vocab

    @property
    def pos_map(self):
        return self._idx_to_pos, self._pos_to_idx

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def pos_dim(self):
        return len(self._idx_to_pos)

    def load_pos_map(self, pos_map):
        self._idx_to_pos = pos_map[0]
        self._pos_to_idx = pos_map[1]

    @staticmethod
    def _read_file(src_file, labeled=True):
        data = []
        all_pos = []

        # current sample
        curr_pos = []
        curr_words = []
        src = open(src_file, "rt")
        for i, row in enumerate(src):
            if row == "\n":
                data.append((curr_words, curr_pos))
                curr_pos = []
                curr_words = []
                continue
            word, pos = row.split() if labeled else (row.strip(), 0)
            curr_words.append(word)
            curr_pos.append(pos)

            all_pos.append(pos)
        return data, list(set(sorted(all_pos)))

    def __getitem__(self, item):
        words, pos = self._data[item]
        embed_vec = []
        for i, (word, p) in enumerate(zip(words, pos)):
            embed_i = self._vocab.vocab(word)
            if embed_i < 0:
                embed_i = self._vocab.vocab(UNKNOWN)
            embed_vec.append(embed_i)
        list_words = torch.Tensor(embed_vec).long()
        label = torch.Tensor([self._pos_to_idx[i] for i in pos]).long() if self._labeled else pos

        pref_vec = []
        for word in words:
            embed_i = self._vocab.pref_vocab(word)
            pref_vec.append(embed_i)

        suf_vec = []
        for word in words:
            embed_i = self._vocab.suf_vocab(word)
            suf_vec.append(embed_i)

        return words, (list_words, torch.Tensor(pref_vec).long(), torch.Tensor(suf_vec).long()), label


class Part3LetterDataLoader(Dataset):
    # src_file:         data source file - each row consist of <word POS> or <\n> for end of sentence
    # embed_map_file:   file withe ordered words separated by \n
    def __init__(self, src_file, labeled=True):
        self._labeled = labeled
        self._vocab = {chr(i): i for i in range(128)}                               # learn prob's for unknowns
        self._data, self._idx_to_pos = self._read_file(src_file, labeled=labeled)   # read file to structured data
        self._pos_to_idx = {pos: i for i, pos in enumerate(self._idx_to_pos)}       # pos to index

    def __len__(self):
        return len(self._data)

    def pos_to_idx(self, pos):
        if pos not in self._pos_to_idx:
            return -1
        return self._pos_to_idx[pos]

    def idx_to_pos(self, idx):
        return self._idx_to_pos[idx]

    @property
    def vocabulary(self):
        return self._vocab

    @property
    def pos_map(self):
        return self._idx_to_pos, self._pos_to_idx

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def pos_dim(self):
        return len(self._idx_to_pos)

    def load_pos_map(self, pos_map):
        self._idx_to_pos = pos_map[0]
        self._pos_to_idx = pos_map[1]

    @staticmethod
    def _read_file(src_file, labeled=True):
        data = []
        all_pos = []

        # current sample
        curr_pos = []
        curr_words = []
        src = open(src_file, "rt")
        for i, row in enumerate(src):
            if row == "\n":
                data.append((curr_words, curr_pos))
                curr_pos = []
                curr_words = []
                continue
            word, pos = row.split() if labeled else (row.strip(), 0)
            curr_words.append(word)
            curr_pos.append(pos)

            all_pos.append(pos)
        return data, list(set(sorted(all_pos)))

    def __getitem__(self, item):
        words, pos = self._data[item]
        embed_vec = []
        for i, word in enumerate(words):
            embed_i = []
            for c in word:
                embed_i.append(self._vocab[c])
            embed_vec.append(torch.Tensor(embed_i).long())
        label = torch.Tensor([self._pos_to_idx[i] for i in pos]).long() if self._labeled else pos
        return words, embed_vec, label


if __name__ == "__main__":
    dl_train = Part3DataLoader(os.path.join("..", "data", "pos", "train"), vocab=os.path.join("..", "data", "word_embed"
                                                                                              , "embed_map"))
    data_loader = DataLoader(
            dl_train,
            batch_size=1, shuffle=True
        )

    for batch_index, (words_, pref, suf, label_) in enumerate(data_loader):
        # stdout.write("\r\r\r%d" % int(100 * ((batch_index + 1) / len(data_loader))) + "%")
        if words_.shape[1] == 1:
            print(words_.shape[1], words_)
        stdout.flush()

