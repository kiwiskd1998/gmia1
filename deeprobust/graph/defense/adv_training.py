import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np


class AdvTraining:
    """Adversarial training framework for defending against attacks.

    Parameters
    ----------
    model :
        model to protect, e.g, GCN
    adversary :
        attack model
    device : str
        'cpu' or 'cuda'
    """

    def __init__(self, model, adversary=None, device='cpu'):

        self.model = model
        if adversary is None:
            adversary = RND()
        self.adversary = adversary
        self.device = device

    def adv_train(self, features, adj, labels, idx_train, train_iters, **kwargs):
        """Start adversarial training.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        """
        for i in range(train_iters):
            modified_adj = self.adversary.attack(features, adj)
            self.model.fit(features, modified_adj, train_iters, initialize=False)

            victim_model.eval()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]
            # adj_grad = self.adj_changes.grad

            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            # self.adj_changes.grad.zero_()
            self.projection(n_perturbations)


            victim_model.train()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



