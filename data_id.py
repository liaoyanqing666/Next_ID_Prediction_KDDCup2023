import pandas as pd
import re
import torch
from get_index import get_index
from torch.utils.data import Dataset, DataLoader


class dataset_id(Dataset):
    def __init__(self, country=None, train=True, prev_length=10):
        super(dataset_id, self).__init__()
        if train:
            data = pd.read_csv('KDD-CUP-data/train_ID.csv')
        else:
            data = pd.read_csv('KDD-CUP-data/test_ID.csv')

        if country:
            data = data[data['locale'] == country]

        # 构建id->索引的字典（索引从1开始）
        self.id2index = get_index(country=country)

        # 切割读取的字符串为id数组
        def string2list(string):
            return re.findall(r"'([A-Za-z0-9]{10})'", string)

        # 将id数组转换成对应索引数组
        def list2index(list):
            ans = []
            for item in list:
                id_index = self.id2index.get(item)
                if id_index:
                    ans.append(id_index)
                else:
                    print('Warning: 存在无法确定索引的内容')
            return ans

        # 将id转换成对应索引
        def id2index(id):
            id_index = self.id2index.get(id)
            if id_index:
                return id_index
            else:
                print('Warning: 存在无法确定索引的内容')

        # print(data)
        data['prev_items'] = data['prev_items'].apply(string2list)
        # print(data)
        data['prev_items'] = data['prev_items'].apply(list2index)
        # print(data)
        data['next_item'] = data['next_item'].apply(id2index)
        # print(data)

        # # 查找previous item序列最大长度
        # maxlen = max([len(item) for item in data['prev_items']])
        # print('previous item序列最大长度:', maxlen)

        # 索引数组补全与截取
        def modify(list):
            if len(list) > prev_length:
                return list[-prev_length:]
            else:
                return [0] * (prev_length - len(list)) + list

        data['prev_items'] = data['prev_items'].apply(modify)
        # print(data)

        # 删除以节省内存
        if not train:
            self.id2index = None
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data.iloc[item, 0], self.data.iloc[item, 1]

    def id2index(self):
        return self.id2index


if __name__ == '__main__':
    dataset = dataset_id('IT')
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    for i, (pre, next) in enumerate(loader):
        pre = torch.stack(pre)
        print(pre.shape, next.shape)
        print(pre)
        if i == 9:
            break

