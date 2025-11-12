#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/1/31 11:15
# @Author : ZM7
# @File : generate_neg.py
# @Software: PyCharm

import pandas as pd
import pickle
import argparse
from utils import user_neg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Games', help='dataset name: Beauty, cd, Games, Movie')
    args = parser.parse_args()

    dataset = args.data
    data = pd.read_csv('./Data/' + dataset + '.csv')

    user = data['user_id'].unique()
    item = data['item_id'].unique()
    user_num = len(user)
    item_num = len(item)

    print(f"Dataset: {dataset}, Users: {user_num}, Items: {item_num}")

    # 生成负样本
    data_neg = user_neg(data, item_num)

    # 保存为 pickle
    out_file = f'./Data/{dataset}_neg.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(data_neg, f)

    print(f"Negative samples saved to {out_file}")


