import torch
import torch.multiprocessing as mp
from deeprobust.graph.targeted_attack import BaseAttack
from torch.nn.parameter import Parameter
from deeprobust.graph import utils
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch import optim
from torch.nn import functional as F
from torch.nn.modules.module import Module
import numpy as np
from tqdm import tqdm
import math
import scipy.sparse as sp
class IGAttack(BaseAttack):
    def __init__(self, model, nnodes=None, feature_shape=None, attack_structure=True, attack_features=True,
                 device='cpu'):
        super(IGAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.modified_adj = None
        self.modified_features = None
        self.target_node = None
        self.device = device

    def attack(self, ori_features, ori_adj, labels, idx_train, target_node, n_perturbations, steps=10, **kwargs):
        self.surrogate.eval()
        self.target_node = target_node
        modified_adj = ori_adj.todense()
        modified_features = ori_features.todense()
        adj, features, labels = utils.to_tensor(modified_adj, modified_features, labels, device=self.device)
        pseudo_labels = self.surrogate.predict().detach().argmax(1)  #替代模型预测出的伪标签
        output = self.surrogate.predict()[target_node]
        # t_label = output.argmin(0)
        t_label = self.label1(output)  #返回除正确标签以外第二大的标签
        # print(pseudo_labels[target_node])
        # print(t_label)
        pseudo_labels[idx_train] = labels[idx_train]
        modified_adj1 = np.asarray(modified_adj)
        temp_neighbor_set = self.find_neighbor_idx(modified_adj1, 1, target_node)
        a = self.yuansu(temp_neighbor_set, target_node) #找到目标节点在子图序列中的下标
        temp_neighbor_set1 = torch.from_numpy(temp_neighbor_set)
        sub_labels = self.sub_l(pseudo_labels, temp_neighbor_set1) #找到上面序列的标签
        sub_adj, sub_feat = self.construct_sub_graph(modified_adj, modified_features, temp_neighbor_set) #得到子图的邻接矩阵。特征矩阵
        sub_adj = torch.from_numpy(sub_adj)
        sub_feat = torch.from_numpy(sub_feat)
        sub_adj_norm = utils.normalize_adj_tensor(sub_adj)  #邻接矩阵正则化
        # sub_adj_norm = sub_adj
        target_label = self.target_label(pseudo_labels, t_label)  #得到和目标标签相同的节点
        degree = self.degree(adj, target_label) #按节点的度从低到高排列
        # output = self.surrogate(sub_feat, sub_adj_norm)
        # print(torch.argmax(output, dim=1))
        # print(sub_labels)
        """
        抽取子图的效果与真实图中相似，抽取子图的目的是看目标节点的label是否发生改变，经过验证不但目标节点的label没有改变，并且其余节点
        的label基本没有改变，只有极个别节点的标签发生了改变，是在允许的误差范围内
        """
        if self.attack_structure:
            s_e = self.calc_importance_edge(sub_feat, sub_adj_norm, sub_labels, steps, a, t_label)
        if self.attack_features:
            s_f = self.calc_importance_feature(sub_feat, sub_adj_norm, sub_labels, steps, a)
        for t in (range(int(n_perturbations/2)+1)):
            s_e_max1 = np.argmax(s_e)
            # s_f_max1 = np.argmax(s_f)
            # s_f_max = temp_neighbor_set1[s_f_max1]
            s_e_max = temp_neighbor_set1[s_e_max1]
            # if s_e[s_e_max] >= s_f[s_f_max]:
            if modified_adj[target_node, s_e_max] == 1:
            # value = np.abs(1 - modified_adj[target_node, s_e_max])  # abs取绝对值
                # if value == 0:
                modified_adj[target_node, s_e_max] = 0
                modified_adj[s_e_max, target_node] = 0
            # else:
                modified_adj[target_node, degree[t]] = 1
                modified_adj[degree[t], target_node] = 1
            s_e[s_e_max1] = 0

            # else:
            #     modified_features[target_node, s_f_max] = np.abs(1 - modified_features[target_node, s_f_max])
            #     s_f[s_f_max] = 0

        self.modified_adj = sp.csr_matrix(modified_adj)
        self.modified_features = sp.csr_matrix(modified_features)
        self.check_adj(modified_adj)

    def find_neighbor_idx(self, p_adj: np.ndarray, p_hops: int, p_node_idx: int):
        neighbor_matrix = np.array([p_node_idx], dtype=np.int16)
        for lp_hop in range(p_hops):
            for lp_node_id in range(neighbor_matrix.shape[0]):
                temp_neighbors = np.where(p_adj[neighbor_matrix[lp_node_id]] != 0)[0]
                for idx in temp_neighbors:
                    if neighbor_matrix.__contains__(idx):
                        continue
                    else:
                        neighbor_matrix = np.append(neighbor_matrix, idx)
        return np.sort(neighbor_matrix)

    def sub_l(self, pseudo_labels, temp_neighbor_set):
        n = temp_neighbor_set.shape[0]
        id2 = torch.zeros(1, n)
        id1 = id2.type(torch.long)
        for i in range(n):
            id1[0][i] = pseudo_labels[temp_neighbor_set[i]]
        return id1

    # def sub_node(self, adj, sub_adj, node):
    #     proj_o_to_s = {}  # origin to sub
    #     proj_s_to_o = {}  # sub to origin
    #     for lp_set_id in range(p_node_set.shape[0]):
    #         proj_s_to_o[lp_set_id] = p_node_set[lp_set_id]
    #         proj_o_to_s[p_node_set[lp_set_id]] = lp_set_id
    #     return 0

    def construct_sub_graph(self, p_adj, p_feat, p_node_set: np.ndarray):
        proj_o_to_s = {}  # origin to sub
        proj_s_to_o = {}  # sub to origin
        for lp_set_id in range(p_node_set.shape[0]):
            proj_s_to_o[lp_set_id] = p_node_set[lp_set_id]
            proj_o_to_s[p_node_set[lp_set_id]] = lp_set_id
        sub_adj = np.zeros([p_node_set.shape[0], p_node_set.shape[0]])
        sub_adj = sub_adj.astype(np.float32)
        for lp_node_i in p_node_set:
            for lp_node_j in p_node_set:
                if p_adj[lp_node_i, lp_node_j] == 1:
                    sub_idx_i = proj_o_to_s[lp_node_i]
                    sub_idx_j = proj_o_to_s[lp_node_j]
                    sub_adj[sub_idx_i, sub_idx_j] = 1
        # sub_d = np.diag(p_adj[p_node_set].sum(1))
        sub_feat = np.copy(p_feat[p_node_set])
        return sub_adj, sub_feat

    def calc_importance_edge(self, features, adj_norm, labels, steps, node, t_label):
        labels = labels[0]
        baseline_add = adj_norm.clone()
        baseline_remove = adj_norm.clone()
        baseline_add.data[node] = 1
        baseline_remove.data[node] = 0
        adj_norm.requires_grad = True
        integrated_grad_list = []
        i = node
        # i = self.target_node
        for j in tqdm(range(adj_norm.shape[1])):
            if adj_norm[i][j]:
    # #用普通梯度代替积分梯度进行计算
    #             adj_norm = adj_norm.to(self.device)
    #             features = features.to(self.device)
    #             labels = labels.to(self.device)
    #             output = self.surrogate(features, adj_norm)
    #             loss = F.nll_loss(output[[node]], labels[[node]])
    #             adj_grad = torch.autograd.grad(loss, adj_norm)[0]
    #             adj_grad = adj_grad[i][j]
    #             integrated_grad_list.append(adj_grad.detach().item())
    #     integrated_grad_list[i] = 0
    #     return integrated_grad_list
    # #    代码结尾
                scaled_inputs = [baseline_remove + (float(k)/ steps) * (adj_norm - baseline_remove) for k in range(0, steps+1)]
            # else:
            #     scaled_inputs = [baseline_add - (float(k)/ steps) * (baseline_add - adj_norm) for k in range(0, steps + 1)]
            _sum = 0
            for new_adj in scaled_inputs:
                new_adj = new_adj.to(self.device)
                features = features.to(self.device)
                output = self.surrogate(features, new_adj)
                labels = labels.to(self.device)
                # output = self.surrogate.predict()[node]
                # loss = F.cross_entropy(output[[node]],labels[[node]])
                loss = F.nll_loss(output[[node]], labels[[node]])
                adj_grad = torch.autograd.grad(loss, adj_norm)[0]
                # output1 = output[node][t_label]
                # adj_grad = torch.autograd.grad(output1, adj_norm)[0]
                adj_grad = adj_grad[i][j]
                _sum += adj_grad
            if adj_norm[i][j]:
                avg_grad = (adj_norm[i][j] - 0) * _sum.mean()
            else:
                avg_grad = (1 - adj_norm[i][j]) * _sum.mean()
            integrated_grad_list.append(avg_grad.detach().item())
        integrated_grad_list[i] = 0
        integrated_grad_list = np.array(integrated_grad_list)
        adj = (adj_norm > 0).cpu().numpy()
        # print(adj[node])
        # print(integrated_grad_list)
        integrated_grad_list = (-2 * adj[node] + 1) * integrated_grad_list  #什么意思
        integrated_grad_list[node] = -10
        return integrated_grad_list

    def calc_importance_feature(self, features, adj_norm, labels, steps, node):
        labels = labels[0]
        baseline_add = features.clone()
        baseline_remove = features.clone()
        baseline_add.data[0] = 1
        baseline_remove.data[0] = 0
        features.requires_grad = True
        integrated_grad_list = []
        i = node
        # i = self.target_node
        for j in tqdm(range(features.shape[1])):
            if features[i][j]:
                scaled_inputs = [baseline_add + (float(k)/ steps) * (features - baseline_add) for k in range(0, steps + 1)]
            # else:
            #     scaled_inputs = [baseline_remove - (float(k)/ steps) * (baseline_remove - features) for k in range(0, steps + 1)]
            _sum = 0
            for new_features in scaled_inputs:
                output = self.surrogate(new_features, adj_norm)
                loss = F.nll_loss(output[[node]],
                        labels[[node]])
                feature_grad = torch.autograd.grad(loss, features)[0]
                feature_grad = feature_grad[i][j]
                _sum += feature_grad
            if features[i][j]:
                avg_grad = (features[i][j] - 0) * _sum.mean()
            else:
                avg_grad = (1 - features[i][j]) * _sum.mean()
            integrated_grad_list.append(avg_grad.detach().item())
        features = (features > 0).cpu().numpy()
        integrated_grad_list = np.array(integrated_grad_list)
        integrated_grad_list = (-2 * features[0] + 1) * integrated_grad_list
        return integrated_grad_list

    def target_label(self, labels, target):
        target_label = []
        for i in range (labels.shape[0]):
            if labels[i] == target:
                target_label.append(i)
        return target_label

    def degree(self, adj, label):
        degrees = adj.cpu().numpy().sum(0)
        child = []
        for i in range(len(label)):
            tmp = []
            tmp.append(label[i])
            tmp.append(degrees[label[i]])
            child.append(tmp)

        result = sorted(child, key=(lambda x: [x[1]]))  #x是child,lambda是根据child的[1]元素进行排序
        # result.reverse()
        # print(result)
        nodes = []
        for i in range(len(result)):
            nodes.append(result[i][0])
        return nodes

    def label1(self, output):
        # print(output)
        child = []
        for i in range(len(output)):
            tmp = []
            tmp.append(i)
            tmp.append(output[i])
            child.append(tmp)
        result = sorted(child, key=(lambda x: [x[1]]))
        result.reverse()
        node = result[1][0]
        # print(node)
        return node

    def yuansu(self, list, target_node):
        for i in range (len(list)):
            if list[i] == target_node:
                return i




