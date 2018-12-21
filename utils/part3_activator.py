import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch import nn
from Part3.part3_bi_model import Part3RnnBiLSTMA, Part3RnnBiLSTMC, Part3RnnBiLSTMD, Part3RnnBiLSTMB
from utils.loggers import PrintLogger
from utils.part3_data_loader import Part3DataLoader, Part3LetterDataLoader
from utils.part3_params import END, DEBUG, PAD_POS_TAG
from sys import stdout


class Part3Activator:
    def __init__(self, model: nn.Module, train_loader, dev_loader=None, lr=0.1, gpu=True):
        self._len_data = 0
        self._gpu = gpu
        # init models with current models
        self._model = model
        self._batch_size = 124
        self._model.set_optimizer(lr=lr)
        if self._gpu:
            self._model.cuda()
        # empty list for every model - y axis for plotting loss by epochs
        self._dev_loader = None
        self._train_loader = None
        self._train = train_loader
        self._dev = dev_loader
        self._load_data()

    # load dataset
    def _load_data(self):
        # set train loader
        self._train_loader = DataLoader(
            self._train,
            batch_size=self._batch_size, shuffle=True
        )
        if self._dev:
            # set validation loader
            self._dev_loader = DataLoader(
                self._dev,
                batch_size=self._batch_size, shuffle=True
            )

    # train a model, input is the enum of the model type
    def train(self, total_epoch, validation_rate=5):
        logger = PrintLogger("NN_train")
        logger.info("start_train")
        loss_vec_dev = []
        loss_vec_train = []
        accuracy_vec_dev = []
        accuracy_vec_train = []

        for epoch_num in range(total_epoch):
            logger.info("epoch:" + str(epoch_num))
            # set model to train mode
            self._model.train()

            # calc number of iteration in current epoch
            for batch_index, (str_w, x, label) in enumerate(self._train_loader):
                stdout.write("\r\r\r%d" % int(100 * (batch_index + 1) / len(self._train_loader)) + "%")
                stdout.flush()

                self._model.zero_grad()               # zero gradients
                output = self._model(x)               # calc output of current model on the current batch
                loss = F.nll_loss(output.view(label.shape[0] * label.shape[1], -1), label.view(-1).long().cuda(),
                                  ignore_index=self._train_loader.dataset.pos_to_idx(PAD_POS_TAG))
                loss.backward()                                 # back propagation
                self._model.optimizer.step()                    # update weights

                if validation_rate and batch_index % validation_rate == 0:
                    print("")
                    logger.info("validating dev...    epoch:" + "\t" + str(epoch_num) + "/" + str(total_epoch))
                    # validate Train
                    loss, accuracy = self._validate(self._train_loader, job="Train")
                    loss_vec_train.append((epoch_num, loss))
                    accuracy_vec_train.append((epoch_num, accuracy))
                    # validate Dev
                    if self._dev:
                        loss, accuracy = self._validate(self._dev_loader, job="Dev")
                        loss_vec_dev.append((epoch_num, loss))
                        accuracy_vec_dev.append((epoch_num, accuracy))
                    self._model.train()
        return loss_vec_dev, accuracy_vec_dev, loss_vec_train, accuracy_vec_train

    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self, data_loader, job=""):
        logger = PrintLogger(job + "_NN_validation")
        loss_count = 0
        self._model.eval()

        good_preds = 0
        all_preds = 0
        tag_O = [data_loader.dataset.pos_to_idx('O'), data_loader.dataset.pos_to_idx(PAD_POS_TAG)]
        # run though all examples in validation set (from input)
        for batch_index, (str_w, x, label) in enumerate(self._train_loader):
            if batch_index > 3:
                break
            output = self._model(x)                                                         # calc output of the model
            loss_count += F.nll_loss(output.view(label.shape[0] * label.shape[1], -1), label.view(-1).long().cuda(),
                                     ignore_index=self._train_loader.dataset.pos_to_idx(PAD_POS_TAG))
            for p, l in zip(torch.argmax(output.view(label.shape[0] * label.shape[1], -1), dim=1), label.view(-1).long()):
                if l.item() not in [tag_O]:
                    all_preds += 1
                    if p.item() == l.item():
                        good_preds += 1

        loss = float(loss_count / len(data_loader.dataset))
        accuracy = good_preds / all_preds
        logger.info("loss=" + str(loss) + "  ------  accuracy=" + str(accuracy))
        return loss, accuracy

    def predict(self, data):
        test_data_loader = DataLoader(
            data,
            batch_size=1, shuffle=False
        )

        results = []
        for batch_index, (str_w, x, label) in enumerate(test_data_loader):
            self._model.zero_grad()  # zero gradients
            output = self._model(x)
            for word, pred in zip(str_w[0], torch.argmax(output, dim=2).squeeze(dim=0)):
                if data.idx_to_pos(pred.item()) == PAD_POS_TAG:
                    results.append((word, "O"))
                else:
                    results.append((word, data.idx_to_pos(pred.item())))
            results.append((END, END))
        if DEBUG:
            print(results)
        return results


if __name__ == "__main__":
    import os

    dl_train = Part3DataLoader(os.path.join("..", "data", "pos", "train"),
                               vocab=os.path.join("..", "data", "word_embed", "embed_map"))
    dl_dev = Part3DataLoader(os.path.join("..", "data", "pos", "dev"), vocab=dl_train.vocabulary)

    NN = Part3RnnBiLSTMD((100, dl_train.pos_dim), dl_train.vocab_size, dl_train.vocabulary.len_pref()
                         , dl_train.vocabulary.len_suf(), embedding_dim=50, lr=1,
                         pre_trained=os.path.join("..", "data", "word_embed", "wordVectors"))
    ma = Part3Activator(NN, dl_train, dl_dev)
    loss_vec_dev, accuracy_vec_dev, loss_vec_train, accuracy_vec_train = ma.train(5)

    dl_test = Part3DataLoader(os.path.join("..", "data", "pos", "test"), dl_train.vocabulary, labeled=False)
    dl_test.load_pos_map(dl_train.pos_map)
    r = ma.predict(dl_test)
    #
    # dl_train = Part3LetterDataLoader(os.path.join("..", "data", "pos", "train"))
    # dl_dev = Part3LetterDataLoader(os.path.join("..", "data", "pos", "dev"))
    #
    # NN = Part3RnnBiLSTMB((50, 100, dl_train.pos_dim), vocab_size=128, embedding_dim=10, lr=1)
    # ma = Part3Activator(NN, dl_train, dl_dev)
    # loss_vec_dev, accuracy_vec_dev, loss_vec_train, accuracy_vec_train = ma.train(5)
    #
    # dl_test = Part3LetterDataLoader(os.path.join("..", "data", "pos", "test"), labeled=False)
    # dl_test.load_pos_map(dl_train.pos_map)
    # r = ma.predict(dl_test)

    e = 0

