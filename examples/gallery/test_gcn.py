import torch
import numpy as np
import torch.nn.functional as F
from examples.gallery.model.gcn import GCN
from deeprobust.graph.utils import *
from examples.gallery.callbacks import ModelCheckpoint
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset, PrePtbDataset
import argparse
from examples.gallery.Graph.graph import Graph
from deeprobust.graph import utils
cuda = torch.cuda.is_available()
print('cuda: %s' % cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = Dataset(root='D:/yanjiusheng/code/DeepRobust-master/tmp', name="cora", setting='nettack')
# perturbed_data = PtbDataset(root='D:/yanjiusheng/code/DeepRobust-master/tmp', name="cora")

# perturbed_adj = perturbed_data.adj
adj_matrix, attr_matrix, label = data.adj, data.features, data.labels

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
graph = Graph(adj_matrix, attr_matrix, label, copy=False)

trainer = GCN(device=device, seed=123).setup_graph(graph).build()
cb =ModelCheckpoint('model.pth', monitor='val_accuracy')
trainer.fit(idx_train, idx_val, verbose=1, callbacks=[cb], epochs=200)
results = trainer.evaluate(idx_test)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
# from examples.gallery.inference_utils import *
# output = trainer.predict()
# output_train_benign1 = output[idx_train]
# output_test_benign1 = output[idx_test]
# train_label = label[idx_train]
# test_label = label[idx_test]
# output_train_benign = output_train_benign1.cpu().detach().numpy()
# output_test_benign = output_test_benign1.cpu().detach().numpy()
# inference_via_confidence(output_train_benign, output_test_benign, train_label, test_label)
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# loss = utils.loss_acc(output, label, idx_train)
# loss = loss.cpu().detach().numpy()
# loss1 = utils.loss_acc(output, label, idx_test)
# loss1 = loss1.cpu().detach().numpy()
# s1 = pd.Series(loss)
# s2 = pd.Series(loss1)
# sns.distplot(s1,kde=True, color="red")
# sns.distplot(s2, kde=True, color="blue")
# # sns.distplot(s, kde=True,color="blue")
# # sns.distplot(s1, kde=True, color="green")
# plt.text(0, 5, "train:red",bbox=dict(facecolor='white', alpha=0.5), fontsize=10)
# plt.text(0, 4.5, "test:blue",bbox=dict(facecolor='white', alpha=0.5), fontsize=10)
# plt.show()
import random
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.defense import GCN
target_nodes = random.sample(idx_test.tolist(), 20)
# Setup Surrogate model
surrogate = GCN(nfeat=attr_matrix.shape[1], nclass=label.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)
surrogate = surrogate.to(device)
surrogate.fit(attr_matrix, adj_matrix, label, idx_train)

all_margins = []
all_adv_margins = []

for target_node in target_nodes:
    # set up Nettack
    adversary = Nettack(surrogate, nnodes=adj_matrix.shape[0], attack_structure=True, attack_features=False, device=device)
    adversary = adversary.to(device)
    degrees = adj_matrix.sum(0).A1
    n_perturbations = int(degrees[target_node]) + 2
    adversary.attack(attr_matrix,adj_matrix, label, target_node, n_perturbations)
    perturbed_adj = adversary.modified_adj

    # model = GCN(nfeat=features.shape[1], nclass=labels.max()+1,
    #         nhid=16, dropout=0, with_relu=False, with_bias=True, device=device)
    # model = model.to(device)
		#
    # print('=== testing GCN on perturbed graph ===')
    # model.fit(features, perturbed_adj, labels, idx_train)
    # output = model.output
    # margin = classification_margin(output[target_node], labels[target_node])
    # all_margins.append(margin)

    print('=== testing adv-GCN on perturbed graph ===')
    graph = Graph(perturbed_adj, attr_matrix, label, copy=False)
    trainer1 = trainer.setup_graph(graph).build()
    output =trainer1.predict()
    adv_margin = classification_margin(output[target_node], label[target_node])
    all_adv_margins.append(adv_margin)


print("No adversarial training: classfication margin for {0} nodes: {1}".format(len(target_nodes), np.mean(all_margins)))

print("Adversarial training: classfication margin for {0} nodes: {1}".format(len(target_nodes), np.mean(all_adv_margins)))