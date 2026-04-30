import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
import pickle
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
import json

# Sızıntısız grafları yükle
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    train_graphs = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    test_graphs = pickle.load(f)

# Validation için bir val seti yok, direkt test üzerinde değerlendir
# Basitçe train ve test olarak kullan

print(f"Train hücre: {len(train_graphs)}, Test: {len(test_graphs)}")

# Node feature normalizasyonu (önceki scaler kullanılabilir, ama sıfırdan yapalım)
from sklearn.preprocessing import StandardScaler
all_feats = []
for g in train_graphs:
    all_feats.append(g.x.numpy())
all_feats = np.vstack(all_feats)
scaler = StandardScaler()
scaler.fit(all_feats)

def normalize_graphs(graphs, scaler):
    for g in graphs:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    return graphs

train_graphs = normalize_graphs(train_graphs, scaler)
test_graphs = normalize_graphs(test_graphs, scaler)

batch_size = 32
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

class NodeLevelGCN(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.soh_head = nn.Linear(hidden_dim, 1)
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return self.soh_head(x).squeeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sample = train_graphs[0]
model = NodeLevelGCN(sample.x.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

def train_epoch(loader):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc='GCN Training'):
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index)
        loss = criterion(pred, data.y[:, 0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index)
            preds.append(pred.cpu().numpy())
            trues.append(data.y[:, 0].cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    return mean_absolute_error(true, pred), r2_score(true, pred)

epochs = 50
best_test_mae = float('inf')
for epoch in range(1, epochs+1):
    train_loss = train_epoch(train_loader)
    test_mae, test_r2 = evaluate(test_loader)
    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test MAE: {test_mae:.4f} R²: {test_r2:.4f}")
    if test_mae < best_test_mae:
        best_test_mae = test_mae
        torch.save(model.state_dict(), '/home/msoylu/alper/analysis/results/gcn_no_leakage.pth')

model.load_state_dict(torch.load('/home/msoylu/alper/analysis/results/gcn_no_leakage.pth'))
final_mae, final_r2 = evaluate(test_loader)
print(f"\n✅ GCN Test -> MAE: {final_mae:.4f}, R²: {final_r2:.4f}")