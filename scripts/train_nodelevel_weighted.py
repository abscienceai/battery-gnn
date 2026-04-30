import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import pickle
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import json

# -------------------------------
# 1. Veriyi yükle
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs.pkl'), 'rb') as f:
    train_graphs = pickle.load(f)
with open(os.path.join(data_dir, 'val_graphs.pkl'), 'rb') as f:
    val_graphs = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs.pkl'), 'rb') as f:
    test_graphs = pickle.load(f)

print(f"Train hücre: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

# -------------------------------
# 2. Model tanımı (node-level GCN)
# -------------------------------
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
            x = F.relu(conv(x, edge_index))
        soh = self.soh_head(x).squeeze()
        rul = self.rul_head(x).squeeze()
        return soh, rul

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sample = train_graphs[0]
node_in_dim = sample.x.shape[1]
model = NodeLevelGCN(node_in_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
w_soh, w_rul = 1.0, 0.01

batch_size = 32
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

def train_epoch(loader):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc='Training'):
        data = data.to(device)
        optimizer.zero_grad()
        soh_pred, rul_pred = model(data.x, data.edge_index)
        loss = w_soh * criterion(soh_pred, data.y[:,0]) + w_rul * criterion(rul_pred, data.y[:,1])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader):
    model.eval()
    total_loss = 0
    soh_preds, soh_trues = [], []
    rul_preds, rul_trues = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            soh_pred, rul_pred = model(data.x, data.edge_index)
            loss = w_soh * criterion(soh_pred, data.y[:,0]) + w_rul * criterion(rul_pred, data.y[:,1])
            total_loss += loss.item()
            soh_preds.append(soh_pred.cpu().numpy())
            soh_trues.append(data.y[:,0].cpu().numpy())
            rul_preds.append(rul_pred.cpu().numpy())
            rul_trues.append(data.y[:,1].cpu().numpy())
    soh_pred = np.concatenate(soh_preds)
    soh_true = np.concatenate(soh_trues)
    rul_pred = np.concatenate(rul_preds)
    rul_true = np.concatenate(rul_trues)
    return {
        'soh_mae': mean_absolute_error(soh_true, soh_pred),
        'soh_rmse': np.sqrt(mean_squared_error(soh_true, soh_pred)),
        'soh_r2': r2_score(soh_true, soh_pred),
        'rul_mae': mean_absolute_error(rul_true, rul_pred),
        'rul_rmse': np.sqrt(mean_squared_error(rul_true, rul_pred)),
        'rul_r2': r2_score(rul_true, rul_pred)
    }

epochs = 50
best_val_mae = float('inf')
for epoch in range(1, epochs+1):
    train_loss = train_epoch(train_loader)
    val_metrics = evaluate(val_loader)
    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val SOH MAE: {val_metrics['soh_mae']:.4f} R²: {val_metrics['soh_r2']:.4f}")
    if val_metrics['soh_mae'] < best_val_mae:
        best_val_mae = val_metrics['soh_mae']
        torch.save(model.state_dict(), '/home/msoylu/alper/analysis/results/nodelevel_best.pth')
        print("  ✅ Model kaydedildi.")

model.load_state_dict(torch.load('/home/msoylu/alper/analysis/results/nodelevel_best.pth'))
test_metrics = evaluate(test_loader)
print("\n✅ Node-Level GCN Test Sonuçları:")
print(f"  SOH MAE: {test_metrics['soh_mae']:.4f}, R²: {test_metrics['soh_r2']:.4f}")
print(f"  RUL MAE: {test_metrics['rul_mae']:.4f}, R²: {test_metrics['rul_r2']:.4f}")

with open('/home/msoylu/alper/analysis/results/nodelevel_weighted_metrics.json', 'w') as f:
    json.dump(test_metrics, f, indent=2)
print("Metrikler kaydedildi: results/nodelevel_weighted_metrics.json")