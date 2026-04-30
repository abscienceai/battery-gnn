import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import pickle
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
import json

# -------------------------------
# 1. Veriyi yükle
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs.pkl'), 'rb') as f:
    all_train = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs.pkl'), 'rb') as f:
    all_test = pickle.load(f)

# Kaynak: MATR (domain 0), Hedef: HUST (domain 2)
source_graphs = [g for g in all_train if g.domain.item() == 0]   # MATR
target_graphs = [g for g in all_train if g.domain.item() == 2]   # HUST
target_test = [g for g in all_test if g.domain.item() == 2]       # HUST test

print(f"Source (MATR) train: {len(source_graphs)}")
print(f"Target (HUST) train: {len(target_graphs)}")
print(f"Target (HUST) test: {len(target_test)}")

# -------------------------------
# 2. Model tanımı (GNN)
# -------------------------------
class GNNModel(nn.Module):
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
node_in_dim = source_graphs[0].x.shape[1]
model = GNNModel(node_in_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
reg_criterion = nn.MSELoss()

batch_size = 32
source_loader = DataLoader(source_graphs, batch_size=batch_size, shuffle=True)
target_test_loader = DataLoader(target_test, batch_size=batch_size, shuffle=False)

# -------------------------------
# 3. Kaynak (MATR) eğitimi
# -------------------------------
def train_epoch(loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        soh_pred, rul_pred = model(data.x, data.edge_index)
        loss = reg_criterion(soh_pred, data.y[:,0]) + reg_criterion(rul_pred, data.y[:,1])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader):
    model.eval()
    soh_preds, soh_trues = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            soh_pred, _ = model(data.x, data.edge_index)
            soh_preds.append(soh_pred.cpu().numpy())
            soh_trues.append(data.y[:,0].cpu().numpy())
    soh_pred = np.concatenate(soh_preds)
    soh_true = np.concatenate(soh_trues)
    return {
        'soh_mae': mean_absolute_error(soh_true, soh_pred),
        'soh_r2': r2_score(soh_true, soh_pred)
    }

epochs_source = 30
for epoch in range(1, epochs_source+1):
    loss = train_epoch(source_loader)
    print(f"Source epoch {epoch:03d} | Loss: {loss:.4f}")

zero_shot = evaluate(target_test_loader)
print("\nZero-shot (MATR -> HUST test):")
print(f"  SOH MAE: {zero_shot['soh_mae']:.4f}, R²: {zero_shot['soh_r2']:.4f}")

# -------------------------------
# 4. Few-shot fine-tune (GNN)
# -------------------------------
ratios = [0.01, 0.05, 0.10]
few_shot_results = {}
for ratio in ratios:
    n_samples = max(1, int(len(target_graphs) * ratio))
    indices = np.random.choice(len(target_graphs), n_samples, replace=False)
    few_loader = DataLoader([target_graphs[i] for i in indices], batch_size=min(16, n_samples), shuffle=True)
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_mae = float('inf')
    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for data in few_loader:
            data = data.to(device)
            optimizer_ft.zero_grad()
            soh_pred, _ = model(data.x, data.edge_index)
            loss = reg_criterion(soh_pred, data.y[:,0])
            loss.backward()
            optimizer_ft.step()
            total_loss += loss.item()
        val_metrics = evaluate(target_test_loader)
        if val_metrics['soh_mae'] < best_val_mae:
            best_val_mae = val_metrics['soh_mae']
            torch.save(model.state_dict(), f'/tmp/ft_model_{ratio}.pth')
    model.load_state_dict(torch.load(f'/tmp/ft_model_{ratio}.pth'))
    ft_metrics = evaluate(target_test_loader)
    few_shot_results[f'{int(ratio*100)}%'] = ft_metrics
    print(f"\nFew-shot ({ratio*100:.0f}% of HUST train, n={n_samples}):")
    print(f"  SOH MAE: {ft_metrics['soh_mae']:.4f}, R²: {ft_metrics['soh_r2']:.4f}")

results = {'zero_shot': zero_shot, 'few_shot': few_shot_results}
with open('/home/msoylu/alper/analysis/results/few_shot_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n✅ Sonuçlar kaydedildi: results/few_shot_results.json")