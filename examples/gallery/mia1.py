import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset, PrePtbDataset
import argparse
import os
from deeprobust.graph import utils
from torch.utils.data.dataset import TensorDataset
import copy
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
import random
from deeprobust.graph.global_attack import Random
from deeprobust.graph.targeted_attack import Nettack
from tqdm import tqdm
from deeprobust.graph.defense.adv_training import AdvTraining
from inference_utils import *
import random
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

data = Dataset(root='D:/yanjiusheng/code/DeepRobust-master/tmp', name=args.dataset, setting='nettack', seed=15)
# perturbed_data = PtbDataset(root='D:/yanjiusheng/code/DeepRobust-master/tmp', name=args.dataset0)
# perturbed_adj = perturbed_data.adj
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
# model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
# model.fit(features, adj, labels, idx_train, train_iters=500, initialize=False)
model = torch.load("D:/yanjiusheng/code3/DeepRobust-master/examples/graph/model/gcn.pt")
model = model.to(device)
output = model.predict(features, adj)
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
	      "accuracy= {:.4f}".format(acc_test.item()))
# print('---------------------开始对抗训练---------------------------')
"""random攻击"""
# adversary = Random()
# n_perturbations = int(0.01 * (adj.sum()//2))
# for i in tqdm(range(100)):
#     # modified_adj = adversary.attack(features, adj)
#     adversary.attack(adj, n_perturbations=n_perturbations, type='add')
#     modified_adj = adversary.modified_adj
#     model.fit(features, modified_adj, labels, idx_train, train_iters=50, initialize=False)

"""nettack攻击"""
# target_nodes = random.sample(idx_test.tolist(), 20)
# for target_node in target_nodes:
#     # set up Nettack
#     adversary = Nettack(model, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
#     adversary = adversary.to(device)
#     degrees = adj.sum(0).A1
#     n_perturbations = int(degrees[target_node]) + 2
#     adversary.attack(features, adj, labels, target_nodes, n_perturbations)
#     perturbed_adj = adversary.modified_adj
#     model.fit(features, perturbed_adj, labels, idx_train)
	  # adversary.attack(features, adj, labels, target_node, n_perturbations)
	  # perturbed_adj = adversary.modified_adj
	  # model.fit(features, perturbed_adj, labels, idx_train)
model.eval()
# model.fit(features, perturbed_adj, labels, idx_train, train_iters=200, verbose=False)
output = model.predict(features, adj)
# output = model.output
output_train_benign1 = output[idx_train]
output_test_benign1 = output[idx_test]
train_label = labels[idx_train]
test_label = labels[idx_test]
output_train_benign = output_train_benign1.cpu().detach().numpy()
output_test_benign = output_test_benign1.cpu().detach().numpy()
inference_via_confidence(output_train_benign, output_test_benign, train_label, test_label)