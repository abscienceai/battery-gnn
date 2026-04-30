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
from scipy.stats import wilcoxon
import copy

# -------------------------------
# Models (SOH only)
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

print(f"Total train: {len(all_train)}, val: {len(all_val)}, test: {len(all_test)}")

# -------------------------------
# Global normalizasyon (tüm eğitim setine göre)
# -------------------------------
all_feats = np.vstack([g.x.numpy() for g in all_train])
global_scaler = StandardScaler()
global_scaler.fit(all_feats)

# Normalizasyon uygulama fonksiyonu (verilen grafları global scaler ile dönüştürür)
def normalize_graphs(graphs, scaler):
    for g in graphs:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    return graphs

# -------------------------------
# Eğitim fonksiyonu (global scaler kullanır)
# -------------------------------
def train_model(model, train_loader, val_loader, test_loader, epochs=50, device='cpu', is_gcn=True):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val_mae = float('inf')
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            data = data.to(device)
            opt.zero_grad()
            if is_gcn:
                pred = model(data.x, data.edge_index)
            else:
                pred = model(data.x)
            loss = criterion(pred, data.y[:,0].view(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()
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
ratios = [0.01, 0.05, 0.10]  # 1%, 5%, 10%
results = []

for seed in seeds:
    print(f"\nSeed: {seed}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    for ratio in ratios:
        # Eğitim hücrelerinin alt kümesini rastgele seç
        num_train = len(all_train)
        num_subset = max(1, int(num_train * ratio))
        subset_indices = random.sample(range(num_train), num_subset)
        train_subset = [all_train[i] for i in subset_indices]
        # Doğrulama ve test setlerini olduğu gibi al (tüm hücreler)
        val_set = all_val
        test_set = all_test
        
        # Global scaler ile normalize et
        train_norm = normalize_graphs(copy.deepcopy(train_subset), global_scaler)
        val_norm   = normalize_graphs(copy.deepcopy(val_set),    global_scaler)
        test_norm  = normalize_graphs(copy.deepcopy(test_set),   global_scaler)
        
        train_loader = DataLoader(train_norm, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_norm,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_norm,  batch_size=batch_size, shuffle=False)
        
        # GCN
        gcn_model = NodeLevelGCN(train_norm[0].x.shape[1]).to(device)
        gcn_mae, gcn_r2 = train_model(gcn_model, train_loader, val_loader, test_loader, epochs, device, is_gcn=True)
        results.append({'ratio':ratio, 'model':'GCN', 'seed':seed, 'soh_mae':gcn_mae, 'soh_r2':gcn_r2})
        # MLP
        mlp_model = MLP_SOH(train_norm[0].x.shape[1]).to(device)
        mlp_mae, mlp_r2 = train_model(mlp_model, train_loader, val_loader, test_loader, epochs, device, is_gcn=False)
        results.append({'ratio':ratio, 'model':'MLP', 'seed':seed, 'soh_mae':mlp_mae, 'soh_r2':mlp_r2})
        print(f"ratio={ratio*100:.0f}%: GCN MAE={gcn_mae:.4f}, R²={gcn_r2:.4f} | MLP MAE={mlp_mae:.4f}, R²={mlp_r2:.4f}")

# Kaydet
df = pd.DataFrame(results)
df.to_csv('/home/msoylu/alper/analysis/results/lowdata_10seeds_globalnorm.csv', index=False)

# Wilcoxon testleri
print("\nWilcoxon signed-rank tests (GCN vs MLP):")
for ratio in ratios:
    gcn_maes = df[(df['ratio']==ratio) & (df['model']=='GCN')]['soh_mae'].values
    mlp_maes = df[(df['ratio']==ratio) & (df['model']=='MLP')]['soh_mae'].values
    stat, p = wilcoxon(gcn_maes, mlp_maes, alternative='two-sided')
    print(f"ratio {ratio*100:.0f}%: W={stat}, p={p:.6f}")