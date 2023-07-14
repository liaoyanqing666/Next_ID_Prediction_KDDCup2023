import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    本函数及droppath类均来源于ViT的复现代码
    https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/vision_transformer/vit_model.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor

    return output


class Droppath(nn.Module):
    def __init__(self, drop_prob=None):
        super(Droppath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.):
        """
        需要注意满足 emb_dim // num_heads == 0
        :param emb_dim: 嵌入维度
        :param num_heads: head数
        :param dropout: dropout
        """
        super(Attention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        self.head_dim = emb_dim // num_heads

        self.query_linear = nn.Linear(emb_dim, emb_dim)
        self.key_linear = nn.Linear(emb_dim, emb_dim)
        self.value_linear = nn.Linear(emb_dim, emb_dim)
        self.output_linear = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(p=dropout) 

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)

        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        x = torch.matmul(attention_weights, value)

        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, seq_length, self.emb_dim)

        x = self.output_linear(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, emb_dim, active_dim=None, dropout=0.):
        super(FeedForward, self).__init__()
        if not active_dim:
            active_dim = 4 * emb_dim
        self.fc1 = nn.Linear(emb_dim, active_dim)
        self.fc2 = nn.Linear(active_dim, emb_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, emb_dim, num_heads=1, dropout=0.0, droppath=0.0):
        super(Block, self).__init__()
        self.attention = Attention(emb_dim=emb_dim, num_heads=num_heads, dropout=dropout)
        self.fc = FeedForward(emb_dim=emb_dim, active_dim=4*emb_dim, dropout=dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.droppath = Droppath(drop_prob=droppath)


    def forward(self, x):
        x_res = self.attention(x)
        x_res = self.droppath(x_res)
        x = x + x_res
        x = self.layer_norm(x)

        x_res = self.fc(x)
        x_res = self.droppath(x_res)
        x = x + x_res
        x = self.layer_norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, emb_dim, seq_len, num_layers=8, num_heads=8, dropout=0., droppath=0.):
        super(Transformer, self).__init__()
        self.pos_encode = nn.Parameter(torch.randn(1, seq_len, emb_dim)) # 设置可学习的positional encoding
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim)) # 为每个序列增加一个可学习的<CLS> token
        self.layers = nn.ModuleList([Block(emb_dim, num_heads=num_heads, dropout=dropout, droppath=droppath) for _ in range(num_layers)])
        self.fc = nn.Sequential(nn.Linear(emb_dim, emb_dim*4),
                                nn.GELU(),
                                nn.Dropout(),
                                nn.Linear(emb_dim*4, emb_dim))

    def forward(self, x):
        seq_len, batch_size, emb_dim = x.size()
        # [S, B, D]

        # 转换维度
        x = x.transpose(0, 1)
        # [S, B, D] -> [B, S, D]

        # 添加位置编码
        x = x + self.pos_encode
        # 利用广播机制，维度不变

        # 添加<CLS>
        cls_expand = self.cls_token.expand(batch_size, -1, -1) # 将参数张量在维度0上复制B次
        x = torch.cat((x, cls_expand), dim=1)
        # [B, S, D] -> [B, S+1, D]

        # 逐层计算
        for layer in self.layers:
            x = layer(x)
            # 维度不变

        # 取出<CLS>
        x = x[:, -1, :].squeeze(1)
        # [B, S+1, D] -> [B, D]

        # 全连接
        x = self.fc(x)
        return x



if __name__ == '__main__':
    model = Transformer(8, 10, num_layers=12, num_heads=8, dropout=0.3, droppath=0.3)

    model.train()
    tensor = torch.randn(10, 2, 8) # [Seq_len, Batch_size, Emb_dim]
    x = model(tensor)
    print(x.shape)
    print(x)

    model.eval()
    x = model(tensor)
    print(x.shape)
    print(x)
    # atten = Attention(8, 4)
    # x = torch.randn(13, 5, 8)
    # print(atten(x).shape)