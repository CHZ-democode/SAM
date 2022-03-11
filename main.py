#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Package
import torch
import numpy as np
from deeprobust.graph.data import Dataset
from GNN_models.gcn import GCN

## SAM
from SAM import SAM



import scipy.sparse as sp
from utils import ASR_PA, ASR_PGD_PA, ASR_EA, to_tensor, preprocess, set_seed, accuracy, pa_predict, normalize_adj_tensor
from GUA_utils import load_data

import argparse




parser = argparse.ArgumentParser()


# Device & Data Configuration
parser.add_argument('--device', type=int, default=0,
                    help='0-7')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str, default="acm",
                    help='The name of the network dataset.')
parser.add_argument('--new_dataset', type=bool, default=False,
                    help='If adopt the new dataset.')
parser.add_argument('--perturbations', type=int, default=100,  # query
                    help='Number of perturbations.')


# Attack Setting for SAM
parser.add_argument('--node_sim', type=bool, default=True,
                    help='if consider nodes feature similarity')
parser.add_argument('--sim_metric', type=str, default='cosine',
                    help='Metric of node similiarty')
parser.add_argument('--sim_threshold', type=float, default=0,  # larger less flips
                    help='Value of similarity threshold')
parser.add_argument('--topk', type=int, default=120,
                    help='Flip top k largest meta-gradient edge')
parser.add_argument('--free_alpha', type=bool, default=False,
                    help='free alpha')
parser.add_argument('--free_beta', type=bool, default=False,
                    help='free beta')
parser.add_argument('--free_gamma', type=bool, default=False,
                    help='free gamma')
args = parser.parse_args()



# SAM
def sam(index):
    model = SAM(args, model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                 device=device, node_sim=args.node_sim, sim_threshold=args.sim_threshold, topk=args.topk, dataset=args.dataset)
    model.attack(features, adj, labels, idx_train, idx_unlabeled, args.perturbations, ll_constraint=False)
    modified_adj = model.modified_adj

    output = pa_predict(model, features, modified_adj)
    features2 = torch.FloatTensor(features.todense()).to(device)
    output2 = surrogate.predict(features2, modified_adj)

    acc_pa = accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()
    acc_ea = accuracy(output2[idx_unlabeled], labels[idx_unlabeled]).item()

    asr_pa = ASR_PA(model, features, features, labels, idx_test, adj, modified_adj, device)
    asr_ea = ASR_EA(surrogate, features, features, labels, idx_test, adj, modified_adj, device)

    return acc_pa, acc_ea, asr_pa, asr_ea


random_seed_list = [42, 38, 25, 89, 200]

for i in range(len(random_seed_list)):
    set_seed(random_seed_list[i])
    data = Dataset(root='/tmp/', name=args.dataset, setting='gcn')
    adj, features, labels = data.adj, data.features, data.labels

    # Save adj
    adj.toarray()
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
                    with_relu=False, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)  # features csr adj csr labels ndarray

    acc_pa, acc_ea, asr_pa, asr_ea = sam(i)
    asr_pa_list.append(asr_pa)
    asr_ea_list.append(asr_ea)
    acc_pa_list.append(acc_pa)
    acc_ea_list.append(acc_ea)



average_asr_pa = sum(asr_pa_list) / len(asr_pa_list)
average_asr_ea = sum(asr_ea_list) / len(asr_ea_list)
average_acc_pa = sum(acc_pa_list) / len(acc_pa_list)
average_acc_ea = sum(acc_ea_list) / len(acc_ea_list)


