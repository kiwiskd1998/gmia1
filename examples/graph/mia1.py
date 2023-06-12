import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from deeprobust.graph.defense import GCN
from deeprobust.graph.defense.gcn import embedding_GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from torch.nn.parameter import Parameter
from deeprobust.graph.data import PtbDataset, PrePtbDataset
import argparse
import time
import os
from deeprobust.graph import utils
from torch.utils.data.dataset import TensorDataset
import copy
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
import random
#from deeprobust.graph.global_attack import Random
#from deeprobust.graph.targeted_attack import Nettack
from tqdm import tqdm
#from deeprobust.graph.defense.adv_training import AdvTraining
from inference_utils import *
import random
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100, help='Random seed.')
parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def dot_product_decode(Z):
	Z = F.normalize(Z, p=2, dim=1)
	A_pred = torch.relu(torch.matmul(Z, Z.t()))
	# A_pred = torch.matmul(Z, Z.t())
	return A_pred

def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict
data = Dataset(root='E:/yanjiusheng/code/tmp', name=args.dataset, setting='nettack',seed=15)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
model.fit(features, adj, labels, idx_train, idx_val=idx_val, train_iters=200, initialize=False)
#model = torch.load("gcn_gcn_model/gcn_cora.pt")

#adj = sp.csr_matrix((2708, 2708))
output = model.predict(features, adj)
#acc_test = accuracy(output[idx_test], labels[idx_test])
#print("Test set results:", "accuracy= {:.4f}".format(acc_test.item()))

#embedding = embedding_GCN(nfeat=features.shape[1], nhid=16, device=device) #一层GCN，维度是1433变为16
#embedding.load_state_dict(transfer_state_dict(model.state_dict(), embedding.state_dict()))
adj = adj.todense()
adj = torch.tensor(adj)
#features = features.todense()
#features = torch.tensor(features)
#em = embedding(features, adj)  # adj_nrom是经过更改后的邻接矩阵
#adj1 = dot_product_decode(em)
#print(adj1)
adj1, features1 = to_tensor(adj, features)
#b = adj1.sum(1)
#c = adj1.T
#d = torch.div(c, b)
#adj2 = d.T
#output12 = F.softmax(output, dim=1)
#output13 = F.sigmoid(output)
output12 = torch.mm(adj1, output)
#output11 = F.log_softmax(output12, dim=1)
#output11 = F.sigmoid(output12)
#torch.set_printoptions(profile="full")
#print(output11)
acc_test = accuracy(output12[idx_test], labels[idx_test])
print("Test set results:",
	      "accuracy= {:.4f}".format(acc_test.item()))
#print(output13)
model.eval()
#model.test(idx_test)
# model.fit(features, perturbed_adj, labels, idx_train, train_iters=200, verbose=False)
#output = model.predict(features, adj)
# output = model.output
output_train_benign1 = output12[idx_train]
output_test_benign1 = output12[idx_test]
train_label = labels[idx_train]
test_label = labels[idx_test]
output_train_benign = output_train_benign1.cpu().detach().numpy()
output_test_benign = output_test_benign1.cpu().detach().numpy()
inference_via_confidence(output_train_benign, output_test_benign, train_label, test_label)

