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
import copy
import random
from scipy.stats import wilcoxon

# -------------------------------
# Model (SOH only)
# -------------------------------
class NodeLevelGCN(nn.Module):
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

def normalize_features(train_graphs, val_graphs, test_graphs):
    all_feats = np.vstack([g.x.numpy() for g in train_graphs])
    scaler = StandardScaler().fit(all_feats)
    for g in train_graphs: g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    for g in val_graphs:   g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    for g in test_graphs:  g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    return scaler

# -------------------------------
# Varyant uygulama
# -------------------------------
def apply_no_cycle(graphs):
    for g in graphs:
        g.edge_index = torch.empty((2,0), dtype=torch.long)
    return graphs

def apply_no_dqdv(graphs):
    for g in graphs:
        g.x[:, 3:7] = 0.0
    return graphs

# -------------------------------
# Eğitim fonksiyonu (genel)
# -------------------------------
def train_model(model, train_loader, val_loader, test_loader, epochs=50, device='cpu', seed=42, is_gcn=True):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val_mae = float('inf')
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        for data in tqdm(train_loader, desc=f'Seed {seed} epoch {epoch}', leave=False):
            data = data.to(device)
            opt.zero_grad()
            if is_gcn:
                pred = model(data.x, data.edge_index)
            else:
                pred = model(data.x)
            loss = criterion(pred, data.y[:,0].view(-1))
            loss.backward()
            opt.step()
        # validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                if is_gcn:
                    pred = model(data.x, data.edge_index)
                else:
                    pred = model(data.x)
                val_preds.append(pred.cpu().numpy())
                val_trues.append(data.y[:,0].cpu().numpy())
        val_mae = mean_absolute_error(np.concatenate(val_trues), np.concatenate(val_preds))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    # test
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            if is_gcn:
                pred = model(data.x, data.edge_index)
            else:
                pred = model(data.x)
            test_preds.append(pred.cpu().numpy())
            test_trues.append(data.y[:,0].cpu().numpy())
    pred = np.concatenate(test_preds)
    true = np.concatenate(test_trues)
    return mean_absolute_error(true, pred), r2_score(true, pred)

# -------------------------------
# Ana döngü
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50
seeds = [5782, 327, 984, 2430, 3382, 8039, 6182, 5398, 6699, 7656]

variants = {
    'full_model': {'apply': lambda x: x, 'is_gcn': True},
    'no_cycle': {'apply': apply_no_cycle, 'is_gcn': True},
    'no_dqdv': {'apply': apply_no_dqdv, 'is_gcn': True},
    'no_gnn_mlp': {'apply': lambda x: x, 'is_gcn': False}  # MLP, aynı özellikler
}

results = []

for seed in seeds:
    print(f"\n{'='*60}\nSeed: {seed}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    # Normalizasyon eğitim setine göre (tüm varyantlar için aynı scaler kullanılabilir, ancak özellikler değişebilir - her varyant için ayrı normalize etmek daha doğru)
    # Her varyant için kopyalar oluşturup normalize edeceğiz
    for var_name, var_info in variants.items():
        print(f"  Varyant: {var_name}")
        train_cp = copy.deepcopy(all_train)
        val_cp = copy.deepcopy(all_val)
        test_cp = copy.deepcopy(all_test)
        # varyantı uygula
        train_cp = var_info['apply'](train_cp)
        val_cp = var_info['apply'](val_cp)
        test_cp = var_info['apply'](test_cp)
        # normalize
        scaler = normalize_features(train_cp, val_cp, test_cp)
        train_loader = DataLoader(train_cp, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_cp, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_cp, batch_size=batch_size, shuffle=False)
        # model
        input_dim = train_cp[0].x.shape[1]
        if var_info['is_gcn']:
            model = NodeLevelGCN(input_dim).to(device)
        else:
            model = MLP_SOH(input_dim).to(device)
        mae, r2 = train_model(model, train_loader, val_loader, test_loader, epochs, device, seed, var_info['is_gcn'])
        results.append({'variant': var_name, 'seed': seed, 'soh_mae': mae, 'soh_r2': r2})
        print(f"    -> MAE={mae:.4f}, R²={r2:.4f}")

# Kaydet
df = pd.DataFrame(results)
df.to_csv('/home/msoylu/alper/analysis/results/ablation_10seeds.csv', index=False)

# Wilcoxon testleri: full vs diğerleri
full_maes = df[df['variant']=='full_model']['soh_mae'].values
for var in ['no_cycle', 'no_dqdv', 'no_gnn_mlp']:
    var_maes = df[df['variant']==var]['soh_mae'].values
    stat, p = wilcoxon(full_maes, var_maes, alternative='two-sided')
    print(f"\nFull vs {var}: W={stat}, p={p:.6f}")

print("\nSaved to ablation_10seeds.csv")