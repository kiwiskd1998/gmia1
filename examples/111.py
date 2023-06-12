import dgl
import torch
import torch.nn as nn
import dgl.function as fn
from dgl.data import CoraDataset
import torch.optim as optim
from torch.utils.data import DataLoader
dataset = CoraDataset()
g = dataset[0]
features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
test_mask = g.ndata['test_mask']
class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads, num_layers):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(GATLayer(in_feats, hidden_feats, num_heads, activation=torch.nn.ELU()))
        # 隐层
        for l in range(num_layers - 1):
            self.layers.append(GATLayer(hidden_feats * num_heads, hidden_feats, num_heads, activation=torch.nn.ELU()))
        # 输出层
        self.layers.append(GATLayer(hidden_feats * num_heads, out_feats, 1, activation=None))

    def forward(self, g, inputs):
        h = inputs
        for l, layer in enumerate(self.layers):
            h = layer(g, h).flatten(1)
        return h
class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, activation, feat_drop=0.0, attn_drop=0.0):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.activation = activation
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = (z2 * self.attn_l) + (z2 * self.attn_r)
        return {'e': torch.leaky_relu(a.sum(-1), negative_slope=0.2)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = torch.softmax(nodes.mailbox['e'], dim=1)
        alpha = self.attn_drop(alpha)
        h = nodes.mailbox['z']
        return {'z': (alpha * h).sum(1)}

    def forward(self, g, h):
        z = self.feat_drop(h)
        g.ndata['z'] = self.fc(z).view(-1, self.num_heads, self.out_feats)
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata.pop('z')
        if self.activation is not None:
            h = self.activation(h)
        return h

def train(model, optimizer, g, features, labels, train_mask):
    model.train()
    optimizer.zero_grad()
    logits = model(g, features)
    loss = criterion(logits[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        _, predicted_labels = logits[mask].max(dim=1)
        correct = predicted_labels.eq(labels[mask]).sum().item()
        accuracy = correct / mask.sum().item()
    return accuracy

num_heads = 8
num_layers = 2
in_feats = features.shape[1]
hidden_feats = 64
out_feats = dataset.num_classes
model = GAT(in_feats, hidden_feats, out_feats, num_heads, num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
dgl
best_val_acc = 0
best_model_path = 'best_model.pt'
num_epochs = 100
for epoch in range(num_epochs):
    # 训练
    loss = train(model, optimizer, g, features, labels, train_mask)

    # 验证
    train_acc = evaluate(model, g, features, labels, train_mask)
    val_acc = evaluate(model, g, features, labels, val_mask)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")

    # 保存表现最好的模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

# 加载最好的模型并进行测试
best_model = GAT(in_feats, hidden_feats, out_feats, num_heads, num_layers)
best_model.load_state_dict(torch.load(best_model_path))
test_acc = evaluate(best_model, g, features, labels, test_mask)
print(f"Test Accuracy: {test_acc:.4f}")
