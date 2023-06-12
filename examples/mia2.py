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
def inference_via_confidence(loss_train, loss_test):
	sort_confidence = np.sort(np.concatenate((loss_train, loss_test)))  # 所有节点的真实标签的概率
	max_accuracy = 0.5
	best_precision = 0.5
	best_recall = 0.5
	for num in range(len(sort_confidence)):
		delta = sort_confidence[num]
		ratio1 = np.sum(loss_train <= delta) / loss_train.shape[0]  # 训练集中大于这个阈值的所有节点所占的比例
		ratio2 = np.sum(loss_test <= delta) / loss_test.shape[0]  # 测试集中大于这个阈值的所有节点所占的比例
		accuracy_now = 0.5 * (ratio1 + 1 - ratio2)
		if accuracy_now > max_accuracy:
			max_accuracy = accuracy_now
			best_precision = ratio1 / (ratio1 + ratio2)
			best_recall = ratio1
			best_delta = delta
	print('membership inference accuracy is:', max_accuracy)

	return delta

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
output = model.predict(features, adj)
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
	      "accuracy= {:.4f}".format(acc_test.item()))

label = torch.argmax(output, dim=1)
#output_train_benign1 = output[idx_train]
#output_test_benign1 = output[idx_test]
#train_label = label[idx_train]
#test_label = label[idx_test]
train1 = np.random.choice(idx_train, 100)
test1 = np.random.choice(idx_test, 100)
labels = torch.LongTensor(labels)

loss = F.cross_entropy(output, labels, reduction='none')
loss_train1 = loss[idx_train]
loss_test1 = loss[idx_test]
loss_train = loss_train1.cpu().detach().numpy()
loss_test = loss_test1.cpu().detach().numpy()
inference_via_confidence(loss_train, loss_test)
#loss1 = F.cross_entropy(output, label, reduction='none')
#loss1 = loss1.cpu().detach().numpy()
#sort_confidence = np.sort(loss1)  # 所有节点的真实标签的概率
#ratio1 = np.sum(loss_train <= k) / loss_train.shape[0]  # 训练集中大于这个阈值的所有节点所占的比例
#ratio2 = np.sum(loss_test <= k) / loss_test.shape[0]  # 测试集中大于这个阈值的所有节点所占的比例
#accuracy_now = 0.5 * (ratio1 + 1 - ratio2)
#print('membership inference accuracy is:', accuracy_now)