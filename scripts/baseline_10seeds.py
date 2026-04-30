import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random

# -------------------------------
# Models (SOH only)
# -------------------------------
class NodeLevelGCN_SOH(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(node_in_dim, hidden_dim)])
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.soh_head = nn.Linear(hidden_dim, 1)
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return self.soh_head(x).squeeze()

class MLP_SOH(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze()

# -------------------------------
# Veri yükleme
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    all_train = pickle.load(f)
with open(os.path.join(data_dir, 'val_graphs_no_target.pkl'), 'rb') as f:
    all_val = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    all_test = pickle.load(f)

print(f"Train: {len(all_train)}, Val: {len(all_val)}, Test: {len(all_test)}")

# -------------------------------
# Normalizasyon (eğitim setine göre)
# -------------------------------
def normalize_features(train_graphs, val_graphs, test_graphs):
    all_feats = np.vstack([g.x.numpy() for g in train_graphs])
    scaler = StandardScaler().fit(all_feats)
    for g in train_graphs: g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    for g in val_graphs:   g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    for g in test_graphs:  g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    return scaler

# -------------------------------
# Eğitim fonksiyonları (SOH only, validation set kullanarak)
# -------------------------------
def train_gcn(train_loader, val_loader, test_loader, epochs=50, device='cpu', seed=42):
    input_dim = all_train[0].x.shape[1]
    model = NodeLevelGCN_SOH(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val_mae = float('inf')
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        for data in tqdm(train_loader, desc=f'GCN {seed} epoch {epoch}', leave=False):
            data = data.to(device)
            opt.zero_grad()
            pred = model(data.x, data.edge_index)
            loss = criterion(pred, data.y[:,0].view(-1))
            loss.backward()
            opt.step()
        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred = model(data.x, data.edge_index)
                val_preds.append(pred.cpu().numpy())
                val_trues.append(data.y[:,0].cpu().numpy())
        val_mae = mean_absolute_error(np.concatenate(val_trues), np.concatenate(val_preds))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    # Test evaluation
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index)
            test_preds.append(pred.cpu().numpy())
            test_trues.append(data.y[:,0].cpu().numpy())
    pred = np.concatenate(test_preds)
    true = np.concatenate(test_trues)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    return mae, r2

def train_mlp(train_loader, val_loader, test_loader, epochs=50, device='cpu', seed=42):
    input_dim = all_train[0].x.shape[1]
    model = MLP_SOH(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val_mae = float('inf')
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        for data in tqdm(train_loader, desc=f'MLP {seed} epoch {epoch}', leave=False):
            data = data.to(device)
            opt.zero_grad()
            pred = model(data.x)
            loss = criterion(pred, data.y[:,0].view(-1))
            loss.backward()
            opt.step()
        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred = model(data.x)
                val_preds.append(pred.cpu().numpy())
                val_trues.append(data.y[:,0].cpu().numpy())
        val_mae = mean_absolute_error(np.concatenate(val_trues), np.concatenate(val_preds))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    # Test evaluation
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data.x)
            test_preds.append(pred.cpu().numpy())
            test_trues.append(data.y[:,0].cpu().numpy())
    pred = np.concatenate(test_preds)
    true = np.concatenate(test_trues)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    return mae, r2

# -------------------------------
# Ana döngü
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50
seeds = [5782, 327, 984, 2430, 3382, 8039, 6182, 5398, 6699, 7656]

results = []

# Önce bir kez normalize et (eğitim setine göre)
scaler = normalize_features(all_train, all_val, all_test)

for seed in seeds:
    print(f"\n{'='*60}\nSeed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_loader = DataLoader(all_train, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(all_val,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(all_test,  batch_size=batch_size, shuffle=False)
    
    # GCN
    gcn_mae, gcn_r2 = train_gcn(train_loader, val_loader, test_loader, epochs, device, seed)
    results.append({'model':'GCN','seed':seed,'soh_mae':gcn_mae,'soh_r2':gcn_r2})
    print(f"GCN -> MAE={gcn_mae:.4f}, R²={gcn_r2:.4f}")
    
    # MLP
    mlp_mae, mlp_r2 = train_mlp(train_loader, val_loader, test_loader, epochs, device, seed)
    results.append({'model':'MLP','seed':seed,'soh_mae':mlp_mae,'soh_r2':mlp_r2})
    print(f"MLP -> MAE={mlp_mae:.4f}, R²={mlp_r2:.4f}")

# Ridge ve XGBoost için sabit sonuçlar (bir kez hesapla)
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

X_train = np.vstack([g.x.numpy() for g in all_train])
y_train = np.concatenate([g.y[:,0].numpy() for g in all_train])
X_test = np.vstack([g.x.numpy() for g in all_test])
y_test = np.concatenate([g.y[:,0].numpy() for g in all_test])

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
pred = ridge.predict(X_test)
ridge_mae, ridge_r2 = mean_absolute_error(y_test, pred), r2_score(y_test, pred)
for seed in seeds:
    results.append({'model':'Ridge','seed':seed,'soh_mae':ridge_mae,'soh_r2':ridge_r2})

xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)
xgb_mae, xgb_r2 = mean_absolute_error(y_test, pred), r2_score(y_test, pred)
for seed in seeds:
    results.append({'model':'XGBoost','seed':seed,'soh_mae':xgb_mae,'soh_r2':xgb_r2})

# Kaydet
df = pd.DataFrame(results)
df.to_csv('/home/msoylu/alper/analysis/results/baseline_10seeds_fixed.csv', index=False)
print("\nSaved to /home/msoylu/alper/analysis/results/baseline_10seeds_fixed.csv")