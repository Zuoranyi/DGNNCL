import os
from torch.utils.data import Dataset, DataLoader
import _pickle as cPickle
import dgl
import torch
import numpy as np
import pandas as pd
import re
from collections import defaultdict

# ====================================================
# 加载 pickle 文件
# ====================================================
def pickle_loader(path):
    return cPickle.load(open(path, 'rb'))


# ====================================================
# 负采样工具
# ====================================================
def user_neg(data, item_num):
    item = range(item_num)
    def select(data_u, item):
        return np.setdiff1d(item, data_u)
    return data.groupby('user_id')['item_id'].apply(lambda x: select(x, item))


def neg_generate(user, data_neg, neg_num=100):
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u], neg_num, replace=False)
    return neg


# ====================================================
# 主 Dataset：myFloder
# ====================================================
class myFloder(Dataset):
    def __init__(self, root_dir, loader):
        self.root = root_dir
        self.loader = loader
        self.dir_list = load_data(root_dir)
        self.size = len(self.dir_list)

        # ✅ 构建 user_id → step 列表索引
        self.user_steps = defaultdict(list)
        for path in self.dir_list:
            m = re.search(r'/(\d+)/\1_(\d+)\.bin$', path)
            if m:
                user_id, step = map(int, m.groups())
                self.user_steps[user_id].append(step)
        for uid in self.user_steps:
            self.user_steps[uid] = sorted(self.user_steps[uid])

    def __getitem__(self, index):
        dir_ = self.dir_list[index]
        graphs, labels = self.loader(dir_)
        if isinstance(labels, dict):
            labels['path'] = dir_
        else:
            labels = {'path': dir_}

        # ✅ 解析 user_id 与 step
        m = re.search(r'/(\d+)/\1_(\d+)\.bin$', dir_)
        if m:
            user_id, step = map(int, m.groups())
        else:
            user_id, step = None, None

        # ✅ 查找该用户的历史窗口作为 cw_pos_steps
        cw_k = 2  # 默认取前两个窗口；你也可以从 opt.cw_pos_k 动态传入
        cw_pos_steps = []
        if user_id is not None and step is not None:
            history = self.user_steps[user_id]
            prev_steps = [s for s in history if s < step]
            cw_pos_steps = prev_steps[-cw_k:]

        labels.update({
            'user_id': user_id,
            'step': step,
            'cw_pos_steps': cw_pos_steps
        })

        return graphs, labels

    def __len__(self):
        return self.size


# ====================================================
# collate() —— 拼 batch，并提取 cw 信息
# ====================================================
def collate(data):
    user = []
    graph = []
    last_item = []
    label = []
    extras = []

    for da in data:
        g, lbl = da
        path = lbl.get('path', None)
        user_id = lbl.get('user_id', None)
        step = lbl.get('step', None)
        cw_pos_steps = lbl.get('cw_pos_steps', [])

        extras.append({
            'user_id': user_id,
            'path': path,
            'step': step,
            'cw_pos_steps': cw_pos_steps
        })

        # 兼容性处理（DGSR 的 batch 输入）
        user.append(user_id if user_id is not None else 0)
        graph.append(g)
        last_item.append(0)
        label.append(0)

    return (
        torch.tensor(user).long(),
        torch.tensor(user).long(),
        dgl.batch_hetero(graph),
        torch.tensor(label).long(),
        torch.tensor(last_item).long(),
        extras
    )


# ====================================================
# 测试集拼接
# ====================================================
def collate_test(data, user_neg):
    user_alis, graph, last_item, label, user, length = [], [], [], [], [], []
    for da in data:
        user_alis.append(int(da[0].item()) if torch.is_tensor(da[0]) else int(da[0]))
        graph.append(da[1])
        last_item.append(int(da[2].item()) if torch.is_tensor(da[2]) else int(da[2]))
        label.append(int(da[3].item()) if torch.is_tensor(da[3]) else int(da[3]))
        user.append(int(da[4].item()) if torch.is_tensor(da[4]) else int(da[4]))
        length.append(int(da[5].item()) if torch.is_tensor(da[5]) else int(da[5]))
    return (
        torch.tensor(user_alis).long(),
        dgl.batch_hetero(graph),
        torch.tensor(last_item).long(),
        torch.tensor(label).long(),
        torch.tensor(length).long(),
        torch.tensor(neg_generate(user, user_neg)).long()
    )


# ====================================================
# 其他通用工具
# ====================================================
def load_data(data_path):
    data_dir = []
    dir_list = os.listdir(data_path)
    dir_list.sort()
    for filename in dir_list:
        for fil in os.listdir(os.path.join(data_path, filename)):
            data_dir.append(os.path.join(os.path.join(data_path, filename), fil))
    return data_dir


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


# ====================================================
# 评估指标
# ====================================================
def eval_metric(all_top, all_label, all_length, random_rank=True):
    recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = [], [], [], [], [], []
    data_l = np.zeros((100, 7))
    for index in range(len(all_top)):
        per_length = all_length[index]
        if random_rank:
            prediction = (-all_top[index]).argsort(1).argsort(1)
            predictions = prediction[:, 0]
            for i, rank in enumerate(predictions):
                if rank < 20:
                    ndgg20.append(1 / np.log2(rank + 2))
                    recall20.append(1)
                else:
                    ndgg20.append(0)
                    recall20.append(0)
                if rank < 10:
                    ndgg10.append(1 / np.log2(rank + 2))
                    recall10.append(1)
                else:
                    ndgg10.append(0)
                    recall10.append(0)
                if rank < 5:
                    ndgg5.append(1 / np.log2(rank + 2))
                    recall5.append(1)
                else:
                    ndgg5.append(0)
                    recall5.append(0)
        else:
            for top_, target in zip(all_top[index], all_label[index]):
                recall20.append(np.isin(target, top_))
                recall10.append(np.isin(target, top_[0:10]))
                recall5.append(np.isin(target, top_[0:5]))
                if len(np.where(top_ == target)[0]) == 0:
                    ndgg20.append(0)
                    ndgg10.append(0)
                    ndgg5.append(0)
                else:
                    pos = np.where(top_ == target)[0][0]
                    ndgg20.append(1 / np.log2(pos + 2))
                    ndgg10.append(1 / np.log2(pos + 2))
                    ndgg5.append(1 / np.log2(pos + 2))
    return np.mean(recall5), np.mean(recall10), np.mean(recall20), \
           np.mean(ndgg5), np.mean(ndgg10), np.mean(ndgg20), \
           pd.DataFrame(data_l, columns=['r5', 'r10', 'r20', 'n5', 'n10', 'n20', 'number'])


# ====================================================
# 调试输出函数
# ====================================================
def debug_batch_info(batch_id, batch_graph, last_item, label):
    try:
        print(f"\n🟢 Batch {batch_id} ================================")
        print(f"Graph type: {type(batch_graph)}")
        if hasattr(batch_graph, 'num_nodes'):
            print(f"  - Total nodes: {batch_graph.num_nodes()}")
            print(f"  - Total edges: {batch_graph.num_edges()}")
        if hasattr(batch_graph, 'ntypes'):
            for ntype in batch_graph.ntypes:
                print(f"  - Nodes of type '{ntype}': {batch_graph.num_nodes(ntype)}")
        if hasattr(batch_graph, 'etypes'):
            for etype in batch_graph.etypes:
                print(f"  - Edges of type '{etype}': {batch_graph.num_edges(etype)}")
        print(f"  - last_item shape: {tuple(last_item.shape)}")
        print(f"  - label shape: {tuple(label.shape)}")
        print(f"  - label sample: {label[:5].tolist() if len(label) > 5 else label.tolist()}")
        print("====================================================\n")
    except Exception as e:
        print(f"⚠️ debug_batch_info error: {e}")
