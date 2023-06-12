import torch
import numpy as np
import torch.nn.functional as F
from examples.gallery.model.graphvat import GraphVAT
from deeprobust.graph.utils import *
from examples.gallery.callbacks import ModelCheckpoint
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset, PrePtbDataset
import argparse
from examples.gallery.Graph.graph import Graph
cuda = torch.cuda.is_available()
print('cuda: %s' % cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = Dataset(root='D:/yanjiusheng/code/DeepRobust-master/tmp', name="cora", setting='nettack')
# perturbed_data = PrePtbDataset(root='D:/yanjiusheng/code/DeepRobust-master/tmp',name="cora",attack_method='mettack')
# adj_matrix = perturbed_data.adj
adj_matrix, attr_matrix, label = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
print(len(idx_train))
print(len(idx_test))
graph = Graph(adj_matrix, attr_matrix, label, copy=False)
trainer = GraphVAT(device=device, seed=123).setup_graph(graph, feat_transform="normalize_feat").build()
cb = ModelCheckpoint('model.pth', monitor='val_accuracy')
trainer.fit(idx_train, idx_val, verbose=1, callbacks=[cb], epochs=700)
results = trainer.evaluate(idx_test)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')
from examples.gallery.inference_utils import *
output = trainer.predict()
# loss = utils.loss_acc(output, labels, train)
# loss = loss.cpu().detach().numpy()
# loss1 = utils.loss_acc(output, labels, test)
# loss1 = loss1.cpu().detach().numpy()
# s = pd.Series(loss)
# s1 = pd.Series(loss1)
# sns.distplot(s, kde=True,color="red")
# sns.distplot(s1, kde=True, color="blue")
# plt.show()

output_train_benign1 = output[idx_train]
output_test_benign1 = output[idx_test]
train_label = label[idx_train]
test_label = label[idx_test]
output_train_benign = output_train_benign1.cpu().detach().numpy()
output_test_benign = output_test_benign1.cpu().detach().numpy()
inference_via_confidence(output_train_benign, output_test_benign, train_label, test_label)