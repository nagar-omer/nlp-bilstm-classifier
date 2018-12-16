import torch
from torch import nn


class Part1RnnLSTM(nn.Module):
    def __init__(self, dims, vocab_size, embedding_dim=50, lr=0.01):
        super(Part1RnnLSTM, self).__init__()

        # layer sizes
        lstm_out_dim = dims[0]
        hidden_mlp_dim = dims[1]

        # useful info in forward function
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._lstm_layer = nn.LSTM(embedding_dim, lstm_out_dim, 2, dropout=0.5)  # 2 layers RNN
        self._layer1 = nn.Linear(lstm_out_dim, hidden_mlp_dim)
        self._output_layer = nn.Linear(hidden_mlp_dim, 1)

        # set optimizer
        self.optimizer = self.set_optimizer(lr)

    # init optimizer with RMS_prop
    def set_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.embeddings(x)
        # x = torch.tanh(x)
        output_seq, _ = self._lstm_layer(x.squeeze(dim=0).unsqueeze(dim=1))
        x = output_seq.squeeze(dim=0)[-1]
        x = self._layer1(x)
        x = torch.tanh(x)
        x = self._output_layer(x)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    vocab_size = 100000
    RNN = Part1RnnLSTM((100, 50, 20), vocab_size)
    l = 0
    x = torch.LongTensor(4).random_(0, vocab_size)
    RNN(x)
    e = 0
