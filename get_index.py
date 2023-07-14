"""
获取某语言下各id对应的索引
"""

import pandas as pd


def get_index(country=None, index_dir='KDD-CUP-data/products_train.csv'):
    # 读取CSV文件的第一列和第二列（编号和语言）
    data = pd.read_csv(index_dir, usecols=[0, 1])
    if country:
        data = data[data['locale'] == country]
    data = data['id']
    id2index = {id: index + 1 for index, id in enumerate(data)}
    return id2index


if __name__ == '__main__':
    index = get_index('FR')
    print(index)
    # print(index.get('B095NNZCR6'))