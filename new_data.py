#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/15 7:30
# @Author : ZM7
# @File : new_data
# @Software: PyCharm

import dgl
import pandas as pd
import numpy as np
import datetime
import argparse
from dgl.sampling import select_topk
import torch
import os
from dgl import save_graphs
from joblib import Parallel, delayed
from tqdm import tqdm   # ✅ NEW
import sys              # ✅ NEW

# 计算item序列的相对次序，用户视角
def cal_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['order'] = range(len(data))
    return data

# 计算user序列的相对次序，物品视角
def cal_u_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['u_order'] = range(len(data))
    return data

# 保证数据时间的单调递增
def refine_time(data):
    data = data.sort_values(['time'], kind='mergesort')
    time_seq = data['time'].values
    time_gap = 1
    for i, da in enumerate(time_seq[0:-1]):
        if time_seq[i] == time_seq[i+1] or time_seq[i] > time_seq[i+1]:
            time_seq[i+1] = time_seq[i+1] + time_gap
            time_gap += 1
    data['time'] = time_seq
    return data

# bi-directional bipartite graph, with edges having timestamps
def generate_graph(data):
    data = data.groupby('user_id').apply(refine_time).reset_index(drop=True)
    data = data.groupby('user_id').apply(cal_order).reset_index(drop=True)
    data = data.groupby('item_id').apply(cal_u_order).reset_index(drop=True)
    user = data['user_id'].values
    item = data['item_id'].values
    time = data['time'].values
    graph_data = {
        ('item','by','user'):(torch.tensor(item), torch.tensor(user)),
        ('user','pby','item'):(torch.tensor(user), torch.tensor(item))
    }
    graph = dgl.heterograph(graph_data)
    graph.edges['by'].data['time'] = torch.LongTensor(time)
    graph.edges['pby'].data['time'] = torch.LongTensor(time)
    graph.nodes['user'].data['user_id'] = torch.LongTensor(np.unique(user))
    graph.nodes['item'].data['item_id'] = torch.LongTensor(np.unique(item))
    return graph

# 以单个用户为单位，滑动时间窗口生成一系列小图；每个小图代表该用户在时间点 t_j 前后的交互上下文；在每个时刻 j 构建一个「预测下一次行为」的子图
def generate_user(user, data, graph, item_max_length, user_max_length, train_path, test_path, k_hop=3, val_path=None,
                  cw_pos_k=2):
    data_user = data[data['user_id'] == user].sort_values('time')
    u_time = data_user['time'].values
    u_seq = data_user['item_id'].values
    split_point = len(u_seq) - 1
    train_num = 0
    test_num = 0
    # 生成训练数据
    if len(u_seq) < 3:
        return 0, 0
    else:
        # j是index，t是时间；取到sequence的倒数第二个；每一次循环代表在时间 t（索引 j）时构建一个动态子图样本
        for j, t in enumerate(u_time[0:-1]):
            # 第一次交互没有之前的交互历史用来构建图，所以跳过
            if j == 0:
                continue
            # DGSR 要为用户在时间点 u_time[j] 创建一个子图，这个子图只包含用户最近的一段历史交互
            if j < item_max_length:
                start_t = u_time[0]
            else:
                start_t = u_time[j - item_max_length]
            # item_max_length：控制“窗口大小”，往前看多少次交互；start_t：时间窗口的起始时间；u_time[j]：当前时间点（窗口的结束时间）
            sub_u_eid = (graph.edges['by'].data['time'] < u_time[j+1]) & (graph.edges['by'].data['time'] >= start_t)
            sub_i_eid = (graph.edges['pby'].data['time'] < u_time[j+1]) & (graph.edges['pby'].data['time'] >= start_t)
            sub_graph = dgl.edge_subgraph(graph, edges={'by':sub_u_eid, 'pby':sub_i_eid}, relabel_nodes=False)
            # dgl的图的edges（）函数会返回一对tensor:(src_nodes, dst_nodes) = graph.edges(etype='by')
            u_temp = torch.tensor([user])
            his_user = torch.tensor([user])
            graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user':u_temp})
            # torch.unique会返回这个张量中去重后的唯一值，并且会按升序排序；去掉重复的 item 节点 ID，保留唯一集合
            i_temp = torch.unique(graph_i.edges(etype='by')[0])
            his_item = torch.unique(graph_i.edges(etype='by')[0])
            # dgl.NID代表的是每条边在原始图（即全局图 graph）中的全局编号
            edge_i = [graph_i.edges['by'].data[dgl.NID]]
            edge_u = []
            # the codes above involves: subgraph constructed within the time window from the global graph, sub-subgraph sampling accoridng 
            # to the weight and centre user, record the items interacted by the centre user in the sub-subgraph
            for _ in range(k_hop-1):
                # graph_u中每一个item都留下了user_max_length个邻居；u_temp把之前graph_u中留下的所有邻居根据weight再筛选出最终的user_max_length个
                graph_u = select_topk(sub_graph, user_max_length, weight='time', nodes={'item': i_temp})
                u_temp = np.setdiff1d(torch.unique(graph_u.edges(etype='pby')[0]), his_user)[-user_max_length:]
                graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user': u_temp})
                his_user = torch.unique(torch.cat([torch.tensor(u_temp), his_user]))
                i_temp = np.setdiff1d(torch.unique(graph_i.edges(etype='by')[0]), his_item)
                his_item = torch.unique(torch.cat([torch.tensor(i_temp), his_item]))
                edge_i.append(graph_i.edges['by'].data[dgl.NID])
                edge_u.append(graph_u.edges['pby'].data[dgl.NID])
            all_edge_u = torch.unique(torch.cat(edge_u))
            all_edge_i = torch.unique(torch.cat(edge_i))
            fin_graph = dgl.edge_subgraph(sub_graph, edges={'by':all_edge_i,'pby':all_edge_u})
            # target为next item
            target = u_seq[j+1]
            last_item = u_seq[j]
            # u_alis为anchor user node在dgl里的global index；last_alis为last item node在dgl里的global index
            u_alis = torch.where(fin_graph.nodes['user'].data['user_id']==user)[0]
            last_alis = torch.where(fin_graph.nodes['item'].data['item_id']==last_item)[0]
            pos_steps = [step for step in range(max(1, j - cw_pos_k), j)]
            pos_tensor = torch.tensor(pos_steps, dtype=torch.long) if len(pos_steps) > 0 else torch.tensor([-1], dtype=torch.long)
            step_tensor = torch.tensor([j], dtype=torch.long)
            # 输入：子图 + user1（当前用户） + itemC（last cliked item）；预测：他下一次会点哪个 item（target = D）
            if j < split_point-1:
                save_graphs(train_path+ '/' + str(user) + '/'+ str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis':u_alis, 'last_alis': last_alis,
                             'cw_pos_steps': pos_tensor, 'step': step_tensor})
                train_num += 1
            if j == split_point - 1 - 1:
                save_graphs(val_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis': u_alis,
                             'last_alis': last_alis, 'cw_pos_steps': pos_tensor, 'step': step_tensor})
            if j == split_point - 1:
                save_graphs(test_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,
                           {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis':u_alis, 'last_alis': last_alis,
                             'cw_pos_steps': pos_tensor, 'step': step_tensor})
                test_num += 1
        return train_num, test_num


def generate_data(data, graph, item_max_length, user_max_length, train_path, test_path, val_path, job=10, k_hop=3,
                  cw_pos_k=2):
    user = data['user_id'].unique()
    # ✅ tqdm progress bar for users
    a = Parallel(n_jobs=job)(delayed(lambda u: generate_user(u, data, graph, item_max_length, user_max_length, train_path, test_path, k_hop, val_path, cw_pos_k))(u) for u in user)
    return a

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sample', help='data name: sample')
    parser.add_argument('--graph', action='store_true', help='no_batch')
    parser.add_argument('--item_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--user_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--job', type=int, default=10, help='number of jobs for parallel')
    parser.add_argument('--k_hop', type=int, default=2, help='k_hop')
    parser.add_argument('--cw_pos_k', type=int, default=2, help='number of previous windows used as positives')
    opt = parser.parse_args()

    data_path = './Data/' + opt.data + '.csv'
    graph_path = './Data/' + opt.data + '_graph'
    data = pd.read_csv(data_path).groupby('user_id').apply(refine_time).reset_index(drop=True)
    data['time'] = data['time'].astype('int64')

    if not os.path.exists(graph_path):
        graph = generate_graph(data)
        save_graphs(graph_path, graph)
    else:
        graph = dgl.load_graphs(graph_path)[0][0]

    train_path = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train/'
    val_path = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/val/'
    test_path = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/'

    print('start:', datetime.datetime.now())
    all_num = generate_data(data, graph, opt.item_max_length, opt.user_max_length, train_path, test_path, val_path, job=opt.job,
                            k_hop=opt.k_hop, cw_pos_k=opt.cw_pos_k)
    train_num = 0
    test_num = 0
    for num_ in all_num:
        train_num += num_[0]
        test_num += num_[1]
    print('The number of train set:', train_num)
    print('The number of test set:', test_num)
    print('end:', datetime.datetime.now())



