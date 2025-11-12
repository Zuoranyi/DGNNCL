# -*- coding: utf-8 -*-
# @Time : 2021/11/17 4:32
# @Author : ZM7
# @File : new_main
# @Software: PyCharm

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
from DGSR_utils import eval_metric, mkdir_if_not_exist, Logger

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

# 从缓存中收集每个用户的“正样本 embedding”（即过去几个窗口的表示）
def collect_positive_groups_from_cache(cache, user_ids, extra_info, max_pos, device):
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
        pos_list = []
        for step in selected_steps:
            cached = user_cache.get(step)
            if cached is not None:
                pos_list.append(cached.to(device))
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


best_result = [0, 0, 0, 0, 0, 0]
best_epoch = [0, 0, 0, 0, 0, 0]
stop_num = 0
embedding_cache = defaultdict(dict)
cache_size = max(opt.cw_cache_size, opt.cw_pos_k)

for epoch in range(opt.epoch):
    stop = True
    epoch_loss = 0
    epoch_rec_loss = 0
    epoch_cw_loss = 0
    iter = 0
    print('start training: ', datetime.datetime.now())
    model.train()

    # for user_id_tensor, user_alias, batch_graph, label, last_item, extra in train_data:
    #     iter += 1
    #     import os, re
    #     for e in extra:
    #         path = e.get('path')
    #         step = None
    #         if path:
    #             filename = os.path.basename(path)
    #             match = re.search(r'_(\d+)\.bin$', filename)
    #             if match:
    #                 step = int(match.group(1))
    #         e['step'] = step

    #     if iter < 3:
    #         print(f"\n[Debug] Batch {iter}")
    #         print("Example extra:", extra[:2])

    #     score, anchor_embedding = model(
    #         batch_graph.to(device),
    #         user_alias.to(device),
    #         last_item.to(device),
    #         is_training=True
    #     )
    #     anchor_embedding = anchor_embedding.detach()
    #     loss = loss_func(score, label.to(device))
    #     user_ids_cpu = user_id_tensor
    #     user_ids_gpu = user_ids_cpu.to(device)

    #     score, anchor_embedding = model(
    #         batch_graph.to(device),
    #         user_alias.to(device),
    #         last_item.to(device),
    #         is_training=True
    #     )
    #     rec_loss = loss_func(score, label.to(device))

    #     if opt.lambda_cw > 0 and iter < 5:
    #         print(f"\n[Debug] Batch {iter}")
    #         print("Example extra:", extra[:2])
    #         print("Cache size:", len(embedding_cache))

    #     if opt.lambda_cw > 0:
    #         positive_groups = collect_positive_groups_from_cache(
    #             embedding_cache, user_ids_cpu.tolist(), extra,
    #             opt.cw_pos_k, anchor_embedding.device
    #         )

    #         if iter < 10:
    #             valid_pos = sum(len(p) > 0 for p in positive_groups)
    #             print(f"[Iter {iter}] 有正样本的比例: {valid_pos}/{len(positive_groups)}")

    #         cw_loss = cross_window_info_nce(anchor_embedding, user_ids_gpu, positive_groups, opt.cw_temp)
    #     else:
    #         cw_loss = anchor_embedding.new_tensor(0.0)

    #     total_loss = rec_loss + opt.lambda_cw * cw_loss
    #     optimizer.zero_grad()
    #     total_loss.backward()
    #     optimizer.step()

    #     if cache_size > 0:
    #         update_embedding_cache(embedding_cache, user_ids_cpu.tolist(), extra, anchor_embedding, cache_size)

    #     epoch_loss += total_loss.item()
    #     epoch_rec_loss += rec_loss.item()
    #     if opt.lambda_cw > 0:
    #         epoch_cw_loss += (opt.lambda_cw * cw_loss).item()

    #     if iter % 400 == 0:
    #         cw_print = epoch_cw_loss / iter if opt.lambda_cw > 0 else 0.0
    #         print('Iter {}, loss {:.4f}, rec {:.4f}, cw {:.4f}'.format(
    #             iter, epoch_loss / iter, epoch_rec_loss / iter, cw_print
    #         ), datetime.datetime.now())
    import time

    for user_id_tensor, user_alias, batch_graph, label, last_item, extra in train_data:
        iter += 1

        # 开始计时
        t_start = time.time()

        # 数据预处理
        t0 = time.time()
        import os, re
        for e in extra:
            path = e.get('path')
            step = None
            if path:
                filename = os.path.basename(path)
                match = re.search(r'_(\d+)\.bin$', filename)
                if match:
                    step = int(match.group(1))
            e['step'] = step
        t_preprocess = time.time() - t0

        # 前向传播
        t1 = time.time()
        score, anchor_embedding = model(
            batch_graph.to(device),
            user_alias.to(device),
            last_item.to(device),
            is_training=True
        )
        rec_loss = loss_func(score, label.to(device))
        t_forward = time.time() - t1

        # CW 对比学习
        t2 = time.time()
        if opt.lambda_cw > 0:
            positive_groups = collect_positive_groups_from_cache(
                embedding_cache, user_id_tensor.tolist(), extra,
                opt.cw_pos_k, anchor_embedding.device
            )
            cw_loss = cross_window_info_nce(anchor_embedding, user_id_tensor.to(device), positive_groups, opt.cw_temp)
        else:
            cw_loss = anchor_embedding.new_tensor(0.0)
        t_cw = time.time() - t2

        # 反向传播与优化
        t3 = time.time()
        total_loss = rec_loss + opt.lambda_cw * cw_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        t_backward = time.time() - t3

        # 缓存更新
        t4 = time.time()
        if cache_size > 0:
            update_embedding_cache(embedding_cache, user_id_tensor.tolist(), extra, anchor_embedding, cache_size)
        t_cache = time.time() - t4

        # 累计
        epoch_loss += total_loss.item()
        epoch_rec_loss += rec_loss.item()
        if opt.lambda_cw > 0:
            epoch_cw_loss += (opt.lambda_cw * cw_loss).item()

        # 打印
        if iter % 400 == 0:
            t_total = time.time() - t_start
            cw_print = epoch_cw_loss / iter if opt.lambda_cw > 0 else 0.0
            print(f"Iter {iter}, loss {epoch_loss/iter:.4f}, rec {epoch_rec_loss/iter:.4f}, cw {cw_print:.4f}")
            print(f"   [Time per batch] preprocess={t_preprocess:.3f}s | "
                f"forward={t_forward:.3f}s | cw={t_cw:.3f}s | backward={t_backward:.3f}s | "
                f"cache={t_cache:.3f}s | total={t_total:.3f}s  {datetime.datetime.now()}")


    if iter > 0:
        epoch_loss /= iter
        epoch_rec_loss /= iter
        epoch_cw_loss = (epoch_cw_loss / iter) if opt.lambda_cw > 0 else 0.0
    else:
        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_cw_loss = 0.0

    model.eval()
    print('Epoch {}, loss {:.4f}, rec {:.4f}, cw {:.4f}'.format(
        epoch, epoch_loss, epoch_rec_loss, epoch_cw_loss
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






