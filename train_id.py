from math import ceil
from torch.utils.tensorboard import SummaryWriter
from data_id import dataset_id
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from LSTM import lstm
from Embedding import embedding
from MRR import compute_scores
from transformer import Transformer
from loss import newloss
torch.set_printoptions(precision=10)
torch.manual_seed(42)

# 部分可调参数
test_model = 'transformer'
batch_size = 512
learning_rate_model = 0.00005
learning_rate_emb = 0.00005
epoch = 2000
country = 'FR'
previous_sequence_length = 20
embedding_dim = 256
batch_norm = False
new_loss = False
record = True

data_train = dataset_id(country, prev_length=previous_sequence_length)
data_test = dataset_id(country, train=False, prev_length=previous_sequence_length)
train_len = len(data_train)
test_len = len(data_test)
print("训练数据{}条，测试数据{}条".format(train_len, test_len))
dl_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
dl_test = DataLoader(data_test, batch_size=test_len, shuffle=False)
num_id = len(data_train.id2index) # 商品ID总数
print("数据读取完成，商品ID总数：", num_id)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


emb = embedding(num_embeddings=num_id + 1, embedding_dim=embedding_dim, batchnorm=batch_norm).to(device)
if test_model == 'lstm':
    model = lstm(embedding_dim=embedding_dim, hidden_size=embedding_dim, num_layers=10, bi=False).to(device)
elif test_model == 'transformer':
    model = Transformer(emb_dim=embedding_dim, seq_len=previous_sequence_length, num_heads=8, num_layers=12, dropout=0, droppath=0).to(device)
else:
    model = None
optim_model = torch.optim.Adam(model.parameters(), lr=learning_rate_model)
optim_emb = torch.optim.Adam(emb.parameters(), lr=learning_rate_emb)

# emb.load_state_dict(torch.load('embedding_dim256.pth'))
# model.load_state_dict(torch.load('transformer_dim256.pth'))

mseloss = nn.MSELoss().to(device)

if record:
    writer = SummaryWriter()


max_score = 0 # 最高测试得分
# 运行之前先测试一下初始值
with torch.no_grad():
    model.eval()
    total_test_loss = 0
    score = torch.ones(0).to(device)
    for j, (pre_index, next_index) in enumerate(dl_test):
        pre_index = torch.stack(pre_index).to(device)
        next_index = next_index.to(device)
        pre_emb = emb(pre_index) # 所有训练数据一起计算batch_norm（如果有的话）
        num_tensors = next_index.size(0) // batch_size # 切分训练数据进行预测
        sliced_pre = torch.split(pre_emb, batch_size, dim=1)
        sliced_index = torch.split(next_index, batch_size)
        for k in range(int(ceil(next_index.size(0) / batch_size))):
            next_predict = model(sliced_pre[k])
            if new_loss:
                total_test_loss += newloss(next_predict, sliced_index[k], emb)
            else:
                next_emb = emb(sliced_index[k])
                total_test_loss += mseloss(next_predict, next_emb)
            score = torch.cat((score, compute_scores(emb, next_predict, sliced_index[k], new=False)))
        print('Epoch:', 0, 'Total test loss:', total_test_loss, 'Average score:', torch.sum(score) / test_len)
        max_score = torch.sum(score) / test_len

        if record:
            writer.add_scalar('score', torch.sum(score) / test_len)
            writer.add_scalar('test_loss', total_test_loss)


# 开始训练
total_train_step = 0
for i in range(epoch):
    print('Epoch:', i + 1)
    model.train()
    for j, (pre_index, next_index) in enumerate(dl_train):
        pre_index = torch.stack(pre_index).to(device)
        next_index = next_index.to(device)
        total_train_step += 1

        pre_emb = emb(pre_index)
        next_predict = model(pre_emb)
        if new_loss:
            loss = newloss(next_predict, next_index, emb)
        else:
            next_emb = emb(next_index)
            loss = mseloss(next_predict, next_emb)
        if total_train_step % 50 == 1:
            print('step:', total_train_step, 'loss:', loss)
            if record:
                writer.add_scalar('train_loss', loss)

        optim_model.zero_grad()
        optim_emb.zero_grad()
        loss.backward()
        optim_model.step()
        optim_emb.step()

    model.eval()
    with torch.no_grad():
        model.eval()
        total_test_loss = 0
        score = torch.ones(0).to(device)
        for j, (pre_index_test, next_index_test) in enumerate(dl_test):
            pre_index = torch.stack(pre_index_test).to(device)
            next_index = next_index_test.to(device)
            pre_emb = emb(pre_index)  # 所有训练数据一起计算batch_norm（如果有的话）
            num_tensors = next_index.size(0) // batch_size  # 切分训练数据进行预测
            sliced_pre = torch.split(pre_emb, batch_size, dim=1)
            sliced_index = torch.split(next_index, batch_size)
            for k in range(int(ceil(next_index.size(0) / batch_size))):
                next_predict = model(sliced_pre[k])
                if new_loss:
                    total_test_loss += newloss(next_predict, sliced_index[k], emb)
                else:
                    next_emb = emb(sliced_index[k])
                    total_test_loss += mseloss(next_predict, next_emb)
                score = torch.cat((score, compute_scores(emb, next_predict, sliced_index[k], new=False)))
        print('Epoch:', i+1, 'Total test loss:', total_test_loss, 'Average score:', torch.sum(score)/test_len)
        print(score)
        if (torch.sum(score)/test_len > max_score):
            torch.save(emb.state_dict(), 'embedding_dim_trans{}.pth'.format(embedding_dim))
            torch.save(model.state_dict(), 'transformer_dim{}.pth'.format(embedding_dim))
            max_score = torch.sum(score)/test_len

        if record:
            writer.add_scalar('score', torch.sum(score)/test_len)
            writer.add_scalar('test_loss', total_test_loss)

        # print(emb.weight())
    print('最高分:', max_score)