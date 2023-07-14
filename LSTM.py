import torch.nn as nn
import torch
from Embedding import embedding
torch.set_printoptions(precision=20)


class lstm(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers=5, bi=False):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, bidirectional=bi)
        if not bi:
            self.fc = nn.Linear(hidden_size, embedding_dim)
        else:
            self.fc = nn.Linear(2 * hidden_size, embedding_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        output, _ = self.lstm(x)
        # [S, B, E] -> [S, B, H]
        output_fc = self.fc(self.gelu(output[-1]))
        # [S, B, H] -> [B, H] -> [B, E]
        return output_fc


if __name__ == '__main__':
    embedding_dim = 3
    hidden_size = 20
    emb = embedding(100, embedding_dim)
    lstm = lstm(embedding_dim, hidden_size, bi=True)
    test = torch.randint(size=(10, 2), high=100, low=0) # 2为batch_size
    print(test)
    print(test.shape)
    test = emb(test)
    print(test)
    print(test.shape)
    test = lstm(test)
    print(test)
    print(test.shape)
    print(emb.weight().shape)
