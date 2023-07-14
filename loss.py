from Embedding import embedding
import torch
import torch.nn as nn


def newloss(predict, true, emb):
    emb_weight = emb.weight()
    len = emb_weight.size(0)
    dis_all = torch.cdist(predict, emb_weight)
    mse = nn.MSELoss(reduction='none').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    dis_true = torch.sqrt(torch.sum(mse(predict, emb(true)), dim=1))
    loss = torch.mean(dis_true / ((torch.sum(dis_all, dim=-1) - dis_true) / (len - 1)))

    return loss


if __name__ == '__main__':
    tensor1 = torch.randn(236, 256) # B, D
    true = torch.randint(size=(236, ), high=4000)
    emb = embedding(4000, 768)
    # print(tensor1, true, emb.weight())
    print(newloss(tensor1, true, emb))
