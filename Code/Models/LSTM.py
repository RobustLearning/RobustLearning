import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes=2, num_layers=2):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True, dropout=0.5,
                            num_layers=self.num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x_embedding = self.embedding(x)
        out, (_, _) = self.lstm(x_embedding)
        out_1 = out[:, -1, :]
        output = self.fc(out_1).squeeze(0)
        return output.view(-1, 2)