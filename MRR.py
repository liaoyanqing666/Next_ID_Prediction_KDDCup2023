import torch
from torch import nn
from Embedding import embedding


def compute_scores(embedding, predict_emb, real_index, new=False):
    """
    :param embedding: embedding层，必须有weight()参数
    :param predict_emb: 预测结果序列
    :param real_index: 真实结果的索引序列
    :return: 预测结果的得分序列
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 计算张量与嵌入层中每个索引的距离
    distances = torch.cdist(predict_emb, embedding.weight()) # 范式

    # weight = embedding.weight() # 余弦相似度
    # distances = torch.ones(predict_emb.size(0), weight.size(0))
    # for i in range(0, predict_emb.size(0)):
    #     for j in range(0, weight.size(0)):
    #         distances[i, j] -= torch.cosine_similarity(predict_emb[i], weight[j], dim=0)
    # 或者 distances = torch.ones(predict_emb.size(0), weight.size(0)) - torch.cosine_similarity(predict_emb.unsqueeze(1), weight.unsqueeze(0), dim=2)

    # 排序并获取前100个索引
    sorted_indices = torch.argsort(distances)[:, :100]

    # # 比较直观的计算分数方法
    # scores = torch.zeros(sorted_indices.size(0), dtype=torch.int64)
    # for i in range(sorted_indices.size(0)):
    #     indices = torch.where(sorted_indices[i] == real_index[i])[0]
    #     if len(indices) > 0:
    #         scores[i] = 100 - indices.item()

    # 这种写法计算分数更省时，当batch_size=900000时，第一种方法用时11.0738秒，第二种0.06528秒
    indices = torch.nonzero(sorted_indices == real_index[:, None]).to(device)
    if new:
        scores = torch.zeros(sorted_indices.size(0), dtype=torch.int64).to(device)
        scores[indices[:, 0]] = 100 - indices[:, 1]
    else:
        scores = torch.zeros(sorted_indices.size(0)).to(device)
        scores[indices[:, 0]] = 1 / (indices[:, 1] + 1)


    return scores


if __name__ == '__main__':
    # 示例
    size = 100
    dim = 3
    embedding = embedding(size, dim)

    # 创建一个大小为 [batch_size * 3, dim] 的张量
    batch_size = 3
    tensor1 = torch.ones(batch_size, dim)
    tensor2 = torch.zeros(batch_size, dim)
    tensor3 = torch.randn(batch_size, dim)
    tensor = torch.cat((tensor1, tensor2, tensor3), dim=0)
    print(tensor)
    print(tensor.shape)

    real_key = torch.randint(size=(3 * batch_size, ), low=0, high=size)
    print(real_key)
    print(real_key.shape)

    scores = compute_scores(embedding, tensor, real_key, new=True)
    print(scores)
