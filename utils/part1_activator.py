import torch.nn.functional as F
from torch.utils.data import DataLoader
from sys import stdout

from utils.loggers import PrintLogger
from utils.part1_data_loader import Part1DataLoader
from utils.part1_rnn_model import Part1RnnLSTM


class Part1ModelActivator:
    def __init__(self, train, dev, batch_size=1, lr=0.005):
        self._len_data = 0
        # init models with current models
        self._model = None
        # empty list for every model - y axis for plotting loss by epochs
        self._data_loader = None
        self._bath_size = batch_size
        self._load_data_and_model(train, dev, batch_size, lr)

    # load dataset
    def _load_data_and_model(self, train, dev, batch_size, lr):

        # create model
        self._model = Part1RnnLSTM((20, 5, 1), train.len_vocab(), embedding_dim=30)
        self._model.set_optimizer(lr=lr)        # set train loader
        self._train_loader = DataLoader(
            train,
            batch_size=batch_size, shuffle=False
        )

        # set validation loader
        self._dev_loader = DataLoader(
            dev,
            batch_size=batch_size, shuffle=False
        )

    # train a model, input is the enum of the model type
    def train(self, total_epoch, validation_rate=1):
        logger = PrintLogger("NN_train")
        logger.info("start_train")
        loss_vec_dev = []
        loss_vec_train = []
        accuracy_vec_dev = []
        accuracy_vec_train = []

        for epoch_num in range(total_epoch):
            logger.info("\nepoch:" + str(epoch_num))
            # set model to train mode
            self._model.train()

            # calc number of iteration in current epoch
            for batch_index, (label, data) in enumerate(self._train_loader):
                # print progress
                stdout.write("\r\r\r%d" % int(100 * (batch_index + 1) / len(self._train_loader)) + "%")
                stdout.flush()

                self._model.zero_grad()                         # zero gradients
                output = self._model(data)                      # calc output of current model on the current batch

                loss = F.binary_cross_entropy(output, label)  # define loss node (negative log likelihood)
                loss.backward()                                 # back propagation
                self._model.optimizer.step()                    # update weights

            print("")
            # validate on Dev and Train
            if validation_rate and epoch_num % validation_rate == 0:
                # log
                logger.info("validating dev...    epoch:" + "\t" + str(epoch_num) + "/" + str(total_epoch))

                # validate train
                loss, accuracy = self._validate(self._train_loader, job="Train")
                loss_vec_train.append((epoch_num, loss))
                accuracy_vec_train.append((epoch_num, accuracy))

                # validate dev
                loss, accuracy = self._validate(self._dev_loader, job="Dev")
                loss_vec_dev.append((epoch_num, loss))
                accuracy_vec_dev.append((epoch_num, accuracy))

        return loss_vec_dev, accuracy_vec_dev, loss_vec_train, accuracy_vec_train

    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self, data_loader, job=""):
        logger = PrintLogger(job + "_NN_validation")
        loss_count = 0
        self._model.eval()

        good_preds = 0
        # run though all examples in validation set (from input)
        for batch_index, (label, data) in enumerate(data_loader):
            output = self._model(data)                                               # calc output of the model
            loss_count += F.binary_cross_entropy(output, label).item()    # sum total loss of all iteration
            good_preds += 1 if round(output.item()) == round(label.item()) else 0

        loss = float(loss_count / len(data_loader.dataset))
        accuracy = good_preds / len(data_loader.dataset)
        logger.info("loss=" + str(loss) + "  ------  accuracy=" + str(accuracy))
        return loss, accuracy


if __name__ == "__main__":
    lstm_part1 = Part1ModelActivator()
    lstm_part1.train(50)


