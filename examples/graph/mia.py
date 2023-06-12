import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset, PrePtbDataset
import argparse
import attack_model
import os
from attack_model import init_params as w_init
from deeprobust.graph import utils
from torch.utils.data.dataset import TensorDataset
import copy
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
import random
#from deeprobust.graph.global_attack import Random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')

#攻击模型的隐藏层宽度
n_hidden = 64
out_classes = 2
#攻击模型的学习率
LR_ATTACK = 0.001
#攻击模型的权重衰减
REG = 1e-7
LR_DECAY = 0.96

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

data = Dataset(root='E:/yanjiusheng/code/tmp', name=args.dataset, setting='gcn', seed=15)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

# print(idx_train.shape)   247
# print(idx_val.shape)     249
# print(idx_test.shape)    1988
train_len = int(len(idx_train)/2)
val_len = int(len(idx_val)/2)
test_len = int(len(idx_test)/2)
#进行数据分割，用原数据集三分之一训练初始模型，三分之训练影子模型，三分之一用来测试
idx_train_shadow, idx_train_target = np.split(idx_train, (train_len,))
idx_val_shadow, idx_val_target = np.split(idx_val, (val_len, ))
idx_test_shadow, idx_test_target = np.split(idx_test, (test_len, ))

# model = torch.load('gnn.pt')
# model.test(idx_test)
model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
model = model.to(device)
#先对模型进行训练
model.fit(features, adj, labels, idx_train, train_iters=200, verbose=False)
# #添加对抗训练
a = False

if (a):
	print("=======================添加对抗训练=============================")
	perturbed_data = PtbDataset(root='D:/yanjiusheng/code/DeepRobust-master/tmp', name=args.dataset)
	perturbed_adj = perturbed_data.adj  #scr矩阵
	# Setup Target Model

	adversary = Random()
	# test on original adj
	print('=== test on original adj ===')
	output = model.output
	acc_test = accuracy(output[idx_test], labels[idx_test])
	print("Test set results:",
	      "accuracy= {:.4f}".format(acc_test.item()))

	print('=== testing GCN on perturbed graph ===')
	model.fit(features, perturbed_adj, labels, idx_train, train_iters=500)
	output = model.output
	acc_test = accuracy(output[idx_test], labels[idx_test])
	print("Test set results:",
	      "accuracy= {:.4f}".format(acc_test.item()))


	# For poisoning attack, the adjacency matrix you have
	# is alreay perturbed
	print('=== Adversarial Training for Poisoning Attack===')
	model.initialize()
	n_perturbations = int(0.01 * (adj.sum()//2))
	for i in tqdm(range(100)):
	    # modified_adj = adversary.attack(features, adj)
	    adversary.attack(adj, n_perturbations=n_perturbations, type='add')
	    modified_adj = adversary.modified_adj
	    model.fit(features, modified_adj, labels, idx_train, train_iters=50, initialize=False)

	model.eval()

	# test directly or fine tune
	print('=== test on perturbed adj ===')
	output = model.predict(features, perturbed_adj)
	acc_test = accuracy(output[idx_test], labels[idx_test])
	print("Test set results:",
	      "accuracy= {:.4f}".format(acc_test.item()))
	print("======================对抗训练完成==========================")

"""          
抽取子图操作         
"""
# def find_neighbor_idx(p_adj: np.ndarray, p_hops: int, p_node_idx: int):
# 	neighbor_matrix = np.array([p_node_idx], dtype=np.int16)
# 	for lp_hop in range(p_hops):
# 		for lp_node_id in range(neighbor_matrix.shape[0]):
# 			temp_neighbors = np.where(p_adj[neighbor_matrix[lp_node_id]] != 0)[0]
# 			for idx in temp_neighbors:
# 				if neighbor_matrix.__contains__(idx):
# 					continue
# 				else:
# 					neighbor_matrix = np.append(neighbor_matrix, idx)
# 	return np.sort(neighbor_matrix)
#
# def sub_l(pseudo_labels, temp_neighbor_set):
# 	n = temp_neighbor_set.shape[0]
# 	id2 = torch.zeros(1, n)
# 	id1 = id2.type(torch.long)
# 	for i in range(n):
# 		id1[0][i] = pseudo_labels[temp_neighbor_set[i]]
# 	return id1
#
# def construct_sub_graph(p_adj, p_feat, p_node_set: np.ndarray):
# 	proj_o_to_s = {}  # origin to sub
# 	proj_s_to_o = {}  # sub to origin
# 	for lp_set_id in range(p_node_set.shape[0]):
# 		proj_s_to_o[lp_set_id] = p_node_set[lp_set_id]
# 		proj_o_to_s[p_node_set[lp_set_id]] = lp_set_id
# 	sub_adj = np.zeros([p_node_set.shape[0], p_node_set.shape[0]])
# 	sub_adj = sub_adj.astype(np.float32)
# 	for lp_node_i in p_node_set:
# 		for lp_node_j in p_node_set:
# 			if p_adj[lp_node_i, lp_node_j] == 1:
# 				sub_idx_i = proj_o_to_s[lp_node_i]
# 				sub_idx_j = proj_o_to_s[lp_node_j]
# 				sub_adj[sub_idx_i, sub_idx_j] = 1
# 	# sub_d = np.diag(p_adj[p_node_set].sum(1))
# 	sub_feat = np.copy(p_feat[p_node_set])
# 	return sub_adj, sub_feat
#
# temp_neighbor_set = find_neighbor_idx(adj, 5, 0)
# temp_neighbor_set1 = torch.from_numpy(temp_neighbor_set)
# sub_labels = self.sub_l(pseudo_labels, temp_neighbor_set1) #找到上面序列的标签
# sub_adj, sub_feat = self.construct_sub_graph(modified_adj, modified_features, temp_neighbor_set) #得到子图的邻接矩阵。特征矩阵
# sub_adj = torch.from_numpy(sub_adj)
# sub_feat = torch.from_numpy(sub_feat)
# sub_adj_norm = utils.normalize_adj_tensor(sub_adj)
"""
抽取子图完成
"""

shadow_model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
shadow_model = shadow_model.to(device)

shadow_model_data = np.concatenate((idx_train_shadow, idx_val_shadow, idx_test_shadow),axis=0)

shadow_model.fit(features, adj, labels, idx_train_shadow, idx_val_shadow, patience=200)
shadow_model.eval()
#有点问题，是在整个图上进行的预测，应该是在提取的图上进行的预测
output = shadow_model.predict()
trainX = torch.cat((output[idx_train_shadow], output[idx_val_shadow],output[idx_test_shadow]), dim=0)
trainY = torch.cat((torch.ones(1,len(idx_train_shadow)), torch.ones(1,len(idx_val_shadow)),  torch.zeros(1, len(idx_test_shadow))), dim=1)
# trainX = torch.cat((output[idx_train], output[idx_test]), dim=0)
# trainY = torch.cat((torch.ones(1,247), torch.zeros(1, 1988)), dim=1)
input_size = labels.max() + 1

atmodel = attack_model.AttackMLP(input_size,n_hidden,out_classes).to(device)

#初始化攻击模型参数
atmodel.apply(w_init)

attack_loss = nn.CrossEntropyLoss()
attack_optimizer = torch.optim.Adam(atmodel.parameters(), lr=LR_ATTACK, weight_decay=REG)
attack_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(attack_optimizer, gamma=LR_DECAY)

#影子模型的训练和测试数据
attackdataset = (trainX, trainY)

def train_per_epoch(model,
                    train_iterator,
                    criterion,
                    optimizer,
                    device,
                    bce_loss=False):
	epoch_loss = 0
	epoch_acc = 0
	correct = 0
	total = 0

	model.train()
	for _, (features, target) in enumerate(train_iterator):
		# Move tensors to the configured device
		features = features.to(device)
		target = target.to(device)
		# Forward pass
		target = target.type(torch.long)
		outputs = model(features)
		if bce_loss:
			# For BCE loss
			loss = criterion(outputs, target.unsqueeze(1))
		else:
			loss = criterion(outputs, target)

		# Backward pass and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Record Loss
		epoch_loss += loss.item()

		# Get predictions for accuracy calculation
		_, predicted = torch.max(outputs.data, 1)
		total += target.size(0)
		correct += (predicted == target).sum().item()

	# Per epoch valdication accuracy calculation
	epoch_acc = correct / total
	epoch_loss = epoch_loss / total

	return epoch_loss, epoch_acc


def val_per_epoch(model,
                  val_iterator,
                  criterion,
                  device,
                  bce_loss=False):
	epoch_loss = 0
	epoch_acc = 0
	correct = 0
	total = 0

	model.eval()
	with torch.no_grad():
		for _, (features, target) in enumerate(val_iterator):
			features = features.to(device)
			target = target.to(device)
			target = target.type(torch.long)

			outputs = model(features)
			# Caluclate the loss
			if bce_loss:
				# For BCE loss
				loss = criterion(outputs, target.unsqueeze(1))
			else:
				loss = criterion(outputs, target)

			# record the loss
			epoch_loss += loss.item()

			# Check Accuracy
			_, predicted = torch.max(outputs.data, 1)
			total += target.size(0)
			correct += (predicted == target).sum().item()

		# Per epoch valdication accuracy and loss calculation
		epoch_acc = correct / total
		epoch_loss = epoch_loss / total

	return epoch_loss, epoch_acc


#训练攻击模型
def train_attack_model(model,
                       dataset,
                       criterion,
                       optimizer,
                       lr_scheduler,
                       device,
                       model_path='./model',
                       epochs=10,
                       b_size=20,
                       num_workers=0,
                       verbose=False,
                       earlystopping=False):
	n_validation = len(idx_val_shadow)  # number of validation samples
	best_valacc = 0
	stop_count = 0
	patience = 5  # Early stopping

	# path = os.path.join(model_path, 'best_attack_model.ckpt')

	train_loss_hist = []
	valid_loss_hist = []
	val_acc_hist = []
	path = os.path.join(model_path, 'best_attack_model.ckpt')
	t_X, t_Y = dataset
	# Contacetnae list of tensors to a single tensor
	# #Create Attack Dataset
	t_Y = t_Y[0]
	t_X = t_X.detach()
	t_X.requires_grad = True
	attackdataset = TensorDataset(t_X,t_Y)

	print('Shape of Attack Feature Data : {}'.format(t_X.shape))
	print('Shape of Attack Target Data : {}'.format(t_Y.shape))
	print('Length of Attack Model train dataset : [{}]'.format(len(attackdataset)))
	print('Epochs [{}] and Batch size [{}] for Attack Model training'.format(epochs, b_size))

	# Create Train and Validation Split
	n_train_samples = len(attackdataset) - n_validation
	train_data, val_data = torch.utils.data.random_split(attackdataset,
	                                                     [n_train_samples, n_validation])

	train_loader = torch.utils.data.DataLoader(dataset=train_data,
	                                           batch_size=b_size,
	                                           shuffle=True,
	                                           num_workers=num_workers)

	val_loader = torch.utils.data.DataLoader(dataset=val_data,
	                                         batch_size=b_size,
	                                         shuffle=False,
	                                         num_workers=num_workers)

	print('----Attack Model Training------')
	for i in range(epochs):

		train_loss, train_acc = train_per_epoch(model, train_loader, criterion, optimizer, device)
		valid_loss, valid_acc = val_per_epoch(model, val_loader, criterion, device)

		valid_loss_hist.append(valid_loss)
		train_loss_hist.append(train_loss)
		val_acc_hist.append(valid_acc)

		lr_scheduler.step()

		print('Epoch [{}/{}], Train Loss: {:.3f} | Train Acc: {:.2f}% | Val Loss: {:.3f} | Val Acc: {:.2f}%'
		      .format(i + 1, epochs, train_loss, train_acc * 100, valid_loss, valid_acc * 100))

		if earlystopping:
			if best_valacc <= valid_acc:
				print('Saving model checkpoint')
				best_valacc = valid_acc
				# Store best model weights
				best_model = copy.deepcopy(model.state_dict())
				torch.save(best_model, path)
				stop_count = 0
			else:
				stop_count += 1
				if stop_count >= patience:  # early stopping check
					print('End Training after [{}] Epochs'.format(epochs + 1))
					break
		else:  # Continue model training for all epochs
			print('Saving model checkpoint')
			best_valacc = valid_acc
			# Store best model weights
			best_model = copy.deepcopy(model.state_dict())
			torch.save(best_model, path)

	return best_valacc


def attack_inference(model,
                     dataset,
                     device):
	print('----Attack Model Testing----')

	targetnames = ['Non-Member', 'Member']
	pred_y = []
	true_y = []


	X, Y = dataset
	Y = Y[0]
	X = X.detach()
	X.requires_grad = True

	# Create Inference dataset
	inferdataset = TensorDataset(X, Y)

	dataloader = torch.utils.data.DataLoader(dataset=inferdataset,
	                                         batch_size=50,
	                                         shuffle=False,
	                                         num_workers=0)

	# Evaluation of Attack Model
	model.eval()
	with torch.no_grad():
		correct = 0
		total = 0
		for i, (inputs, labels) in enumerate(dataloader):
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)

			# Predictions for accuracy calculations
			_, predictions = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predictions == labels).sum().item()

			# print('True Labels for Batch [{}] are : {}'.format(i,labels))
			# print('Predictions for Batch [{}] are : {}'.format(i,predictions))

			true_y.append(labels.cpu())
			pred_y.append(predictions.cpu())

	attack_acc = correct / total
	print('Attack Test Accuracy is  : {:.2f}%'.format(100 * attack_acc))

	true_y = torch.cat(true_y).numpy()
	pred_y = torch.cat(pred_y).numpy()

	print('---Detailed Results----')
	print(classification_report(true_y, pred_y, target_names=targetnames))


attack_valacc = train_attack_model(atmodel, attackdataset, attack_loss,
                                   attack_optimizer, attack_lr_scheduler, device)
print('Validation Accuracy for the Best Attack Model is: {:.2f} %'.format(100 * attack_valacc))

# Load the trained attack model
# modelDir = './model'
# attack_path = os.path.join(modelDir, 'best_attack_model.ckpt')
# attack_model.load_state_dict(torch.load(attack_path))
print('=======================================================')
output = model.predict()

adj1, features1 = to_tensor(adj, features)
adj1 = adj1.to_dense()
output11 = torch.mm(adj1, output)

test_targetX = torch.cat((output[idx_train_target], output[idx_test_target]), dim=0)
test_targetY = torch.cat((torch.ones(1,len(idx_train_target)), torch.zeros(1, len(idx_test_target))), dim=1)
attackdataset_test = (test_targetX, test_targetY)
attack_inference(atmodel,attackdataset_test, device)