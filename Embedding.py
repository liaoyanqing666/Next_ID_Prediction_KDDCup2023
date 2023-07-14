import torch
from torch import nn


class embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, batchnorm=True):
        super(embedding, self).__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.ln = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(embedding_dim, affine=False)

    def forward(self, x):
        x = self.emb(x)
        x = self.ln(x)
        if self.batchnorm:
            size = x.size()
            emb = x.size(-1)
            x = x.view(-1, emb)
            x = self.bn(x)
            x = x.view(size)
        return x

    def weight(self):
        w = self.emb.weight
        w = self.ln(w)
        if self.batchnorm:
            w = self.bn(w)
        return w

if __name__ == '__main__':
    tensor = torch.randint(size=(10, ), high=100, low=0)
    print(tensor)
    print(tensor.shape)
    emb = embedding(100, 256)
    output = emb(tensor)
    print(output)
    print(output.shape)