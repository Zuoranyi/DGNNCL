# -*- coding: utf-8 -*-
# @Time : 2021/11/17 4:32
# @Author : ZM7
# @File : new_main
# @Software: PyCharm

# contrastive learning on the dgl subgraph,edge dropout

import datetime
import torch
from sys import exit
import pandas as pd
import numpy as np
import dgl
from DGSR import DGSR, collate, collate_test
from dgl import load_graphs
import pickle
from utils import myFloder
import warnings
import argparse
import os
import sys
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from DGSR_utils import eval_metric, mkdir_if_not_exist, Logger, edge_dropout

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='sample', help='data name: sample')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
parser.add_argument('--user_update', default='rnn')
parser.add_argument('--item_update', default='rnn')
parser.add_argument('--user_long', default='orgat')
parser.add_argument('--item_long', default='orgat')
parser.add_argument('--user_short', default='att')
parser.add_argument('--item_short', default='att')
parser.add_argument('--feat_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--attn_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
parser.add_argument('--item_max_length', type=int, default=50, help='the max length of item sequence')
parser.add_argument('--user_max_length', type=int, default=50, help='the max length of use sequence')
parser.add_argument('--k_hop', type=int, default=3, help='sub-graph size')
parser.add_argument('--gpu', default='4')
parser.add_argument('--last_item', action='store_true', help='aggreate last item')
parser.add_argument("--record", action='store_true', default=False, help='record experimental results')
parser.add_argument("--val", action='store_true', default=False)
parser.add_argument("--model_record", action='store_true', default=False, help='record model')
parser.add_argument('--lambda_cw', type=float, default=0.0, help='weight for cross-window contrastive loss')
parser.add_argument('--cw_temp', type=float, default=0.1, help='temperature for cross-window InfoNCE')
parser.add_argument('--cw_pos_k', type=int, default=2, help='number of previous windows treated as positives')
parser.add_argument('--cw_cache_size', type=int, default=10, help='maximum number of cached windows per user for contrastive sampling')
parser.add_argument('--lambda_sub', type=float, default=0.0, help='weight of subgraph contrastive loss')
parser.add_argument('--edge_drop_rate', type=float, default=0.2, help='edge dropout rate for subgraph augmentation')
parser.add_argument('--cl_temp', type=float, default=0.2, help='temperature for subgraph contrastive learning')

opt = parser.parse_args()
args, extras = parser.parse_known_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(opt)

if opt.record:
    log_file = f'results/{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
               f'US_{opt.user_short}_IS_{opt.item_short}_La_{args.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
               f'_layer_{opt.layer_num}_l2_{opt.l2}'
    mkdir_if_not_exist(log_file)
    sys.stdout = Logger(log_file)
    print(f'Logging to {log_file}')

if opt.model_record:
    model_file = f'{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
                 f'US_{opt.user_short}_IS_{opt.item_short}_La_{args.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
                 f'_layer_{opt.layer_num}_l2_{opt.l2}'

# loading data
data = pd.read_csv('./Data/' + opt.data + '.csv')
user = data['user_id'].unique()
item = data['item_id'].unique()
user_num = len(user)
item_num = len(item)

train_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train/'
test_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/'
val_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/val/'

train_set = myFloder(train_root, load_graphs)
test_set = myFloder(test_root, load_graphs)
if opt.val:
    val_set = myFloder(val_root, load_graphs)

print('train number:', train_set.size)
print('test number:', test_set.size)
print('user number:', user_num)
print('item number:', item_num)

f = open(f'./Data/{opt.data}_neg.pkl', 'rb')
data_neg = pickle.load(f)

train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate,
                        shuffle=True, pin_memory=True, num_workers=12)
test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize,
                       collate_fn=lambda x: collate_test(x, data_neg),
                       pin_memory=True, num_workers=8)
if opt.val:
    val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize,
                          collate_fn=lambda x: collate_test(x, data_neg),
                          pin_memory=True, num_workers=2)

# 初始化模型
model = DGSR(
    user_num=user_num,
    item_num=item_num,
    input_dim=opt.hidden_size,
    item_max_length=opt.item_max_length,
    user_max_length=opt.user_max_length,
    feat_drop=opt.feat_drop,
    attn_drop=opt.attn_drop,
    user_long=opt.user_long,
    user_short=opt.user_short,
    item_long=opt.item_long,
    item_short=opt.item_short,
    user_update=opt.user_update,
    item_update=opt.item_update,
    last_item=opt.last_item,
    layer_num=opt.layer_num
).cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
loss_func = nn.CrossEntropyLoss()

# ----------------------------------------------------
# CW 正样本提取与补全
# ----------------------------------------------------
def _safe_scalar(value, default=0):
    """将 tensor / list / None 转换为 python 标量。"""
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return int(value.view(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return _safe_scalar(value[0], default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _trim_user_cache(user_cache, limit):
    if limit is None or limit <= 0:
        return
    if len(user_cache) <= limit:
        return
    sorted_steps = sorted(user_cache.keys())
    for old_step in sorted_steps[:-limit]:
        user_cache.pop(old_step, None)

# 为缺失的历史窗口 embedding 做一次离线推理补全（compute-on-demand），确保对比学习能正常获取正样本
def _infer_embeddings_for_steps(model, base_path, uid, steps, device):
    """当缓存中缺少某个 step 的 embedding 时，从磁盘加载 .bin 图文件，做一次前向推理，补全缓存"""
    if not steps:
        return {}
    if base_path is None:
        return {}
    base_dir = os.path.dirname(base_path)
    if not base_dir:
        return {}

    inferred = {}
    was_training = model.training
    if was_training:
        model.eval()
# 保存模型当前状态，临时切为 eval 模式。推理时不需要 dropout，不希望扰乱训练模式，推理结束后再切回训练模式

    with torch.no_grad():
        for step in steps:
            file_path = os.path.join(base_dir, f"{uid}_{step}.bin")
            if not os.path.exists(file_path):
                continue
            graphs, labels = load_graphs(file_path)
            if not graphs:
                continue
            graph = graphs[0].to(device)
            # DGL load_graphs 返回的 labels 是 tensor 字典
            user_alias = _safe_scalar(labels.get('u_alis'), default=_safe_scalar(labels.get('user'), uid))
            last_item = _safe_scalar(labels.get('last_alis'), default=0)
            user_tensor = torch.tensor([user_alias], dtype=torch.long, device=device)
            last_tensor = torch.tensor([last_item], dtype=torch.long, device=device)
            _, embedding = model(graph, user_tensor, last_tensor, is_training=True)
            inferred[step] = embedding.squeeze(0).detach().cpu()

    if was_training:
        model.train()

    return inferred


# 从缓存中收集每个用户的“正样本 embedding”（即过去几个窗口的表示）
# def collect_positive_groups_from_cache(cache, user_ids, extra_info, max_pos, device):
def collect_positive_groups_from_cache(cache, user_ids, extra_info, max_pos, device, model=None, cache_limit=None):
    if max_pos <= 0:
        return [[] for _ in extra_info]
    positive_groups = []
    for uid, info in zip(user_ids, extra_info):
        steps = info.get('cw_pos_steps', [])
        if not steps:
            positive_groups.append([])
            continue
        # 降序排列，只取前两个
        selected_steps = sorted(steps, reverse=True)[:max_pos]
        user_cache = cache.get(uid, {})
        # pos_list = []
        step_to_tensor = {}
        missing_steps = []
        for step in selected_steps:
            cached = user_cache.get(step)
            if cached is not None:
                # pos_list.append(cached.to(device))
                            step_to_tensor[step] = cached.to(device)
            else:
                missing_steps.append(step)

        if missing_steps and model is not None:
            inferred = _infer_embeddings_for_steps(model, info.get('path'), uid, missing_steps, device)
            if inferred:
                user_cache = cache[uid]
                for step, emb in inferred.items():
                    user_cache[step] = emb.clone()
                    step_to_tensor[step] = emb.to(device)
                _trim_user_cache(user_cache, cache_limit)

        pos_list = [step_to_tensor[step] for step in selected_steps if step in step_to_tensor]
        positive_groups.append(pos_list)
    return positive_groups
# 每个样本一个 list，里面是它的正样本 embedding

# 在每个 batch 训练完后，把当前窗口（step）的 embedding 存入缓存中，并维护每个用户只保留最近 max_cache 个窗口的 embedding
# cache：dictionary
# max_cache:每个用户最多缓存多少个历史窗口
def update_embedding_cache(cache, user_ids, extra_info, embeddings, max_cache):
    if max_cache <= 0:
        return
    with torch.no_grad():
        flat_embeddings = embeddings.detach()
        # idx：样本在batch中的位置
        for idx, (uid, info) in enumerate(zip(user_ids, extra_info)):
            step = info.get('step')
            if step is None:
                continue
            user_cache = cache[uid]
            user_cache[step] = flat_embeddings[idx].detach().clone()
            if max_cache > 0 and len(user_cache) > max_cache:
                sorted_steps = sorted(user_cache.keys())
                for old_step in sorted_steps[:-max_cache]:
                    user_cache.pop(old_step, None)

# 你在训练每个 batch 时，先调用 collect_positive_groups_from_cache()从缓存中取出“过去窗口的 embedding”作为正样本。然后模型前向、反向传播完毕后，把“当前窗口”的 embedding 存入缓存，供未来批次使用

def cross_window_info_nce(anchor_embeddings, user_ids, positive_groups, temperature):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    device = anchor_embeddings.device
    user_ids = user_ids.to(device)
    anchor_norm = F.normalize(anchor_embeddings, dim=-1)
    losses = []
    for idx, positives in enumerate(positive_groups):
        if not positives:
            continue
        anchor_vec = anchor_norm[idx]
        pos_stack = torch.stack([F.normalize(p, dim=-1) for p in positives], dim=0)
        pos_logits = torch.matmul(pos_stack, anchor_vec) / temperature
        # 负样本掩码
        neg_mask = user_ids != user_ids[idx]
        if neg_mask.sum() == 0:
            continue
        neg_vectors = anchor_norm[neg_mask]
        # 取出当前batch里其它用户的embedding作为negative samples
        neg_logits = torch.matmul(neg_vectors, anchor_vec) / temperature
        # 合并正负样本
        all_logits = torch.cat([pos_logits, neg_logits], dim=0)
        # infonce formula
        log_num = torch.logsumexp(pos_logits, dim=0)
        log_denom = torch.logsumexp(all_logits, dim=0)
        losses.append(-(log_num - log_denom))
    if not losses:
        return anchor_embeddings.new_tensor(0.0)
    return torch.stack(losses).mean() 
# 衡量整个 batch（多个用户）的平均对比学习质量

def info_nce_subgraph(emb_orig, emb_aug, temperature):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    emb_orig_norm = F.normalize(emb_orig, dim=-1)
    emb_aug_norm = F.normalize(emb_aug, dim=-1)
    logits = torch.matmul(emb_orig_norm, emb_aug_norm.transpose(0, 1)) / temperature
    labels = torch.arange(emb_orig_norm.shape[0], device=emb_orig_norm.device)
    return F.cross_entropy(logits, labels)



best_result = [0, 0, 0, 0, 0, 0]
best_epoch = [0, 0, 0, 0, 0, 0]
stop_num = 0

for epoch in range(opt.epoch):
    stop = True
    epoch_loss = 0
    epoch_rec_loss = 0
    epoch_cl_loss = 0
    iter = 0
    print('start training: ', datetime.datetime.now())
    model.train()

    for user_id_tensor, user_alias, batch_graph, label, last_item in train_data:
        iter += 1
        batch_graph = batch_graph.to(device)
        user_alias = user_alias.to(device)
        last_item = last_item.to(device)
        label = label.to(device)

        score, emb_orig = model(
            batch_graph,
            user_alias,
            last_item,
            is_training=True
        )
        rec_loss = loss_func(score, label)

        loss_cl = torch.tensor(0.0, device=device)
        if opt.lambda_sub > 0:
            aug_graph = edge_dropout(batch_graph, drop_rate=opt.edge_drop_rate)
            _, emb_aug = model(
                aug_graph,
                user_alias,
                last_item,
                is_training=True
            )
            loss_cl = info_nce_subgraph(emb_orig, emb_aug, opt.cl_temp)

        total_loss = rec_loss + opt.lambda_sub * loss_cl
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_rec_loss += rec_loss.item()
        if opt.lambda_sub > 0:
            epoch_cl_loss += (opt.lambda_sub * loss_cl).item()

        if iter % 400 == 0:
            cl_print = epoch_cl_loss / iter if opt.lambda_sub > 0 else 0.0
            print('Iter {}, loss {:.4f}, rec {:.4f}, cl {:.4f}'.format(
                iter, epoch_loss / iter, epoch_rec_loss / iter, cl_print
            ), datetime.datetime.now())


    if iter > 0:
        epoch_loss /= iter
        epoch_rec_loss /= iter
        epoch_cl_loss = (epoch_cl_loss / iter) if opt.lambda_sub > 0 else 0.0
    else:
        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_cl_loss = 0.0

    model.eval()
    print('Epoch {}, loss {:.4f}, rec {:.4f}, cl {:.4f}'.format(
        epoch, epoch_loss, epoch_rec_loss, epoch_cl_loss
    ), '=============================================')

    # val
    if opt.val:
        print('start validation: ', datetime.datetime.now())
        val_loss_all, top_val = [], []
        with torch.no_grad:
            for user, batch_graph, label, last_item, neg_tar in val_data:
                score, top = model(
                    batch_graph.to(device), user.to(device), last_item.to(device),
                    neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device), is_training=False
                )
                val_loss = loss_func(score, label.cuda())
                val_loss_all.append(val_loss.item())
                top_val.append(top.detach().cpu().numpy())
            recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(top_val)
            print(
                'train_loss:%.4f\tval_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f'
                '\tNDGG10@10:%.4f\tNDGG@20:%.4f' %
                (epoch_loss, np.mean(val_loss_all), recall5, recall10, recall20, ndgg5, ndgg10, ndgg20)
            )

    # test
    print('start predicting: ', datetime.datetime.now())
    all_top, all_label, all_length = [], [], []
    iter = 0
    all_loss = []
    with torch.no_grad():
        for user, batch_graph, label, last_item, neg_tar in test_data:
            iter += 1
            score, top = model(
                batch_graph.to(device), user.to(device), last_item.to(device),
                neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device), is_training=False
            )
            test_loss = loss_func(score, label.cuda())
            all_loss.append(test_loss.item())
            all_top.append(top.detach().cpu().numpy())
            all_label.append(label.numpy())
            if iter % 200 == 0:
                print('Iter {}, test_loss {:.4f}'.format(iter, np.mean(all_loss)), datetime.datetime.now())

        recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(all_top)
        if recall5 > best_result[0]:
            best_result[0] = recall5
            best_epoch[0] = epoch
            stop = False
        if recall10 > best_result[1]:
            if opt.model_record:
                torch.save(model.state_dict(), 'save_models/' + model_file + '.pkl')
            best_result[1] = recall10
            best_epoch[1] = epoch
            stop = False
        if recall20 > best_result[2]:
            best_result[2] = recall20
            best_epoch[2] = epoch
            stop = False
        if ndgg5 > best_result[3]:
            best_result[3] = ndgg5
            best_epoch[3] = epoch
            stop = False
        if ndgg10 > best_result[4]:
            best_result[4] = ndgg10
            best_epoch[4] = epoch
            stop = False
        if ndgg20 > best_result[5]:
            best_result[5] = ndgg20
            best_epoch[5] = epoch
            stop = False

        if stop:
            stop_num += 1
        else:
            stop_num = 0
        print(
            'train_loss:%.4f\ttest_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f'
            '\tNDGG10@10:%.4f\tNDGG@20:%.4f\tEpoch:%d,%d,%d,%d,%d,%d' %
            (
                epoch_loss, np.mean(all_loss),
                best_result[0], best_result[1], best_result[2],
                best_result[3], best_result[4], best_result[5],
                best_epoch[0], best_epoch[1], best_epoch[2],
                best_epoch[3], best_epoch[4], best_epoch[5]
            )
        )





