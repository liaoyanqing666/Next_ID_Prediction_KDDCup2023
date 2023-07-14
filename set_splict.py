import pandas as pd
import os

data = pd.read_csv('KDD-CUP-data/sessions_train.csv')


def split(country, train_dir, test_dir, header=False, scale=10):
    data_split = data[data['locale'] == country]
    num = len(data_split)
    print('国家：{}，总数据：{}'.format(country, num))
    data_split[: num - num // scale].to_csv(train_dir, mode='a', header=header, index=False)
    data_split[num - num // scale:].to_csv(test_dir, mode='a', header=header, index=False)


def main(folder='KDD-CUP-data', train_file='train_ID.csv', test_file='test_ID.csv'):
    if folder != '':
        train_dir = os.path.join(folder, train_file)
        test_dir = os.path.join(folder, test_file)
    else:
        train_dir = train_file
        test_dir = test_file

    # 已存在该文件则删除
    if os.path.exists(train_dir):
        os.remove(train_dir)
    if os.path.exists(test_dir):
        os.remove(test_dir)

    # 德语
    print('选取德语')
    split('DE', train_dir, test_dir, header=True)
    split('JP', train_dir, test_dir)
    split('UK', train_dir, test_dir)
    split('ES', train_dir, test_dir)
    split('FR', train_dir, test_dir)
    split('IT', train_dir, test_dir)


if __name__ == '__main__':
    main()