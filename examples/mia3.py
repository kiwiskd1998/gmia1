import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset, PrePtbDataset
import argparse
from deeprobust.graph import utils
def inference_via_confidence(confidence_mtx1, confidence_mtx2, label_vec1, label_vec2):
    confidence1 = []
    confidence2 = []
    acc1 = 0
    acc2 = 0
    for num in range(confidence_mtx1.shape[0]):
        confidence1.append(confidence_mtx1[num, label_vec1[num]])  # 添加的是真实label的概率
        # confidence1.append(np.argmax(confidence_mtx1[num, :]))
        if np.argmax(confidence_mtx1[num, :]) == label_vec1[num]:  # 如果最大的概率就是真实的label，acc1+1
            acc1 += 1

    for num in range(confidence_mtx2.shape[0]):
        confidence2.append(confidence_mtx2[num, label_vec2[num]])
        # confidence2.append(np.argmax(confidence_mtx2[num, :]))
        if np.argmax(confidence_mtx2[num, :]) == label_vec2[num]:
            acc2 += 1
    confidence1 = np.array(confidence1)  # 训练集中真实标签的概率
    confidence2 = np.array(confidence2)  # 测试集中真实标签的概率

    print('model accuracy for training and test-', (acc1 / confidence_mtx1.shape[0], acc2 / confidence_mtx2.shape[0]))

    # sort_confidence = np.sort(confidence1)
    sort_confidence = np.sort(np.concatenate((confidence1, confidence2)))  # 所有节点的真实标签的概率
    max_accuracy = 0.5
    best_precision = 0.5
    best_recall = 0.5
    for num in range(len(sort_confidence)):
        delta = sort_confidence[num]
        ratio1 = np.sum(confidence1 >= delta) / confidence_mtx1.shape[0]  # 训练集中大于这个阈值的所有节点所占的比例
        ratio2 = np.sum(confidence2 >= delta) / confidence_mtx2.shape[0]  # 测试集中大于这个阈值的所有节点所占的比例
        accuracy_now = 0.5 * (ratio1 + 1 - ratio2)
        if accuracy_now > max_accuracy:
            max_accuracy = accuracy_now
            best_precision = ratio1 / (ratio1 + ratio2)
            best_recall = ratio1
    print('membership inference accuracy is:', max_accuracy)
    return max_accuracy

def dot_product_decode(Z):
	Z = F.normalize(Z, p=2, dim=1)
	A_pred = torch.relu(torch.matmul(Z, Z.t()))
	return A_pred

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = Dataset(root='E:/yanjiusheng/code/tmp', name=args.dataset, setting='gcn',seed=15)
# perturbed_data = PtbDataset(root='D:/yanjiusheng/code/DeepRobust-master/tmp', name=args.dataset0)
# perturbed_adj = perturbed_data.adj
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
model.fit(features, adj, labels, idx_train, train_iters=500, initialize=False)
#model = torch.load("gcn_gcn_model/gcn_cora.pt")
model.eval()

#output = model.predict(features, adj)
#acc_test = accuracy(output[idx_test], labels[idx_test])
#print("Test set results:","accuracy= {:.4f}".format(acc_test.item()))

output_train_benign1 = output[idx_train]
output_test_benign1 = output[idx_test]
train_label = labels[idx_train]
test_label = labels[idx_test]
output_train_benign = output_train_benign1.cpu().detach().numpy()
output_test_benign = output_test_benign1.cpu().detach().numpy()
inference_via_confidence(output_train_benign, output_test_benign, train_label, test_label)
