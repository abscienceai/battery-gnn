import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import pickle
import numpy as np
import pandas as pd
import os

class NodeLevelGCN(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.soh_head = nn.Linear(hidden_dim, 1)
        self.rul_head = nn.Linear(hidden_dim, 1)
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        soh = self.soh_head(x).squeeze()
        rul = self.rul_head(x).squeeze()
        return soh, rul

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'test_graphs.pkl'), 'rb') as f:
    test_graphs = pickle.load(f)

sample = test_graphs[0]
node_in_dim = sample.x.shape[1]
model = NodeLevelGCN(node_in_dim).to(device)
model.load_state_dict(torch.load('/home/msoylu/alper/analysis/results/nodelevel_best.pth', map_location=device))
model.eval()

loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
results = []
with torch.no_grad():
    for data in loader:
        data = data.to(device)
        soh_pred, _ = model(data.x, data.edge_index)
        for i, cell_id in enumerate(data.cell_id):
            domain = data.domain[i].item()
            true_cap = data.y[i, 0].item()
            pred_cap = soh_pred[i].item()
            results.append((cell_id, domain, true_cap, pred_cap))

df = pd.DataFrame(results, columns=['cell_id', 'domain', 'true', 'pred'])
domain_names = {0: 'MATR', 1: 'CALCE', 2: 'HUST', 3: 'RWTH'}
print("SOH MAE (Ah) per dataset:")
for d in range(4):
    sub = df[df['domain'] == d]
    if len(sub) > 0:
        mae = (sub['true'] - sub['pred']).abs().mean()
        print(f"  {domain_names[d]}: {mae:.4f}")
    else:
        print(f"  {domain_names[d]}: no samples")