import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import pickle
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Model tanımı (SOH-only)
# -------------------------------
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

# -------------------------------
# Veriyi yükle
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    train_graphs = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    test_graphs = pickle.load(f)

print(f"Train graf: {len(train_graphs)}, Test graf: {len(test_graphs)}")

# -------------------------------
# Normalizasyon (eğitim setine göre)
# -------------------------------
def normalize_features(train_graphs, test_graphs):
    all_feats = []
    for g in train_graphs:
        all_feats.append(g.x.numpy())
    all_feats = np.vstack(all_feats)
    scaler = StandardScaler()
    scaler.fit(all_feats)
    for g in train_graphs:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    for g in test_graphs:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    return scaler

scaler = normalize_features(train_graphs, test_graphs)

# -------------------------------
# Domain bilgisi
# -------------------------------
domain_names = {0: 'MATR', 1: 'CALCE', 2: 'HUST', 3: 'RWTH'}

# Test graflarını domain bazında grupla
test_by_domain = {d: [] for d in domain_names.values()}
for g in test_graphs:
    dom_id = g.domain.item() if isinstance(g.domain, torch.Tensor) else int(g.domain)
    dom_name = domain_names[dom_id]
    test_by_domain[dom_name].append(g)

print("\nDomain bazlı test hücre sayıları:")
for dom, glist in test_by_domain.items():
    print(f"{dom}: {len(glist)}")

# -------------------------------
# Modeli yükle
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '/home/msoylu/alper/analysis/results/gcn_no_leakage.pth'
model = NodeLevelGCN(train_graphs[0].x.shape[1]).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

batch_size = 32
results = {}

for dom, graphs in test_by_domain.items():
    if not graphs:
        continue
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index)
            preds.append(pred.cpu().numpy())
            trues.append(data.y[:, 0].cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    results[dom] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"{dom}: MAE={mae:.4f} Ah, RMSE={rmse:.4f} Ah, R²={r2:.4f}")

# Toplam test performansı (tüm domainler)
all_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
preds_all, trues_all = [], []
with torch.no_grad():
    for data in all_loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index)
        preds_all.append(pred.cpu().numpy())
        trues_all.append(data.y[:, 0].cpu().numpy())
pred_all = np.concatenate(preds_all)
true_all = np.concatenate(trues_all)
mae_all = mean_absolute_error(true_all, pred_all)
rmse_all = np.sqrt(mean_squared_error(true_all, pred_all))
r2_all = r2_score(true_all, pred_all)
print(f"\nTüm test seti: MAE={mae_all:.4f} Ah, RMSE={rmse_all:.4f} Ah, R²={r2_all:.4f}")