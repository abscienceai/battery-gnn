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

# -------------------------------
# 1. Veriyi yükle (orijinal graflar, dQdV özellikleri içeriyor)
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    train_graphs = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    test_graphs = pickle.load(f)

# -------------------------------
# 2. dQ/dV özelliklerini sadece RWTH için bırak, diğerlerinde sıfırla
# -------------------------------
def mask_dqdv_features(graphs):
    for g in graphs:
        # RWTH domain=3, diğerleri 0,1,2
        if g.domain.item() != 3:
            # dQdV özellikleri sütun indeksleri: 2,3,4,5 (peak_voltage_1, peak_height_1, peak_voltage_2, peak_height_2)
            g.x[:, 2:6] = 0.0
    return graphs

train_graphs = mask_dqdv_features(train_graphs)
test_graphs = mask_dqdv_features(test_graphs)

# Normalizasyon (sadece train üzerinden)
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
    for data in tqdm(loader, desc='Training (RWTH-only dQdV)'):
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
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    return mae, r2

epochs = 50
best_mae = float('inf')
for epoch in range(1, epochs+1):
    train_loss = train_epoch(train_loader)
    test_mae, test_r2 = evaluate(test_loader)
    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    if test_mae < best_mae:
        best_mae = test_mae
        torch.save(model.state_dict(), '/home/msoylu/alper/analysis/results/gcn_rwth_only_dqdv.pth')

model.load_state_dict(torch.load('/home/msoylu/alper/analysis/results/gcn_rwth_only_dqdv.pth'))
final_mae, final_r2 = evaluate(test_loader)
print(f"\n✅ dQdV sadece RWTH için -> SOH MAE: {final_mae:.4f}, R²: {final_r2:.4f}")