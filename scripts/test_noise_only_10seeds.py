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
from sklearn.preprocessing import StandardScaler
import copy
import random

# -------------------------------
# 1. Model Tanımları (SOH-only)
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

class MLP(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(node_in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze()

# -------------------------------
# 2. Veriyi Yükleme ve Normalizasyon
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    all_train = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    all_test = pickle.load(f)

print(f"Train graf: {len(all_train)}, Test graf: {len(all_test)}")

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

def add_gaussian_noise(graphs, sigma):
    for g in graphs:
        noise = torch.randn_like(g.x) * sigma
        g.x = g.x + noise
    return graphs

def train_model(model, train_loader, epochs=50, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            if isinstance(model, MLP):
                pred = model(data.x)
            else:
                pred = model(data.x, data.edge_index)
            loss = criterion(pred, data.y[:, 0].view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}")
    return model

def evaluate(model, test_loader, device='cpu'):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            if isinstance(model, MLP):
                pred = model(data.x)
            else:
                pred = model(data.x, data.edge_index)
            preds.append(pred.cpu().numpy())
            trues.append(data.y[:, 0].cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    return mae, r2

# -------------------------------
# 3. Senaryo B: Sadece test gürültüsü, 10 tohum
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50
noise_levels = [0.0, 0.01, 0.05, 0.10]
seeds = [5782, 327, 984, 2430, 3382, 8039, 6182, 5398, 6699, 7656]  # aynı tohumlar

# Orijinal verilerin kopyalarını sakla
original_train = copy.deepcopy(all_train)
original_test = copy.deepcopy(all_test)

# Normalizasyon
scaler = normalize_features(original_train, original_test)

# Temiz eğitim verisi ile modelleri eğit (bir kere, tüm gürültü seviyeleri için aynı model)
# Her tohum için ayrı eğitim yapılmalı çünkü model ağırlıkları random seed'e bağlı
results = {sigma: {'MLP': [], 'GCN': []} for sigma in noise_levels}

for seed in seeds:
    print(f"\n{'='*60}")
    print(f"Tohum: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Temiz eğitim verisi ile modelleri eğit
    train_loader = DataLoader(original_train, batch_size=batch_size, shuffle=True)
    
    mlp = MLP(node_in_dim=original_train[0].x.shape[1])
    mlp = train_model(mlp, train_loader, epochs, device)
    
    gcn = NodeLevelGCN(node_in_dim=original_train[0].x.shape[1])
    gcn = train_model(gcn, train_loader, epochs, device)
    
    # Her gürültü seviyesi için test setini boz ve değerlendir
    for sigma in noise_levels:
        test_graphs = copy.deepcopy(original_test)
        if sigma > 0:
            test_graphs = add_gaussian_noise(test_graphs, sigma)
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
        
        mae_mlp, r2_mlp = evaluate(mlp, test_loader, device)
        mae_gcn, r2_gcn = evaluate(gcn, test_loader, device)
        
        results[sigma]['MLP'].append((mae_mlp, r2_mlp))
        results[sigma]['GCN'].append((mae_gcn, r2_gcn))
        print(f"σ={sigma}: MLP -> MAE={mae_mlp:.4f}, R²={r2_mlp:.4f} | GCN -> MAE={mae_gcn:.4f}, R²={r2_gcn:.4f}")

# -------------------------------
# 4. Sonuçları Tablo Halinde Yazdır
# -------------------------------
print("\n" + "="*60)
print("Senaryo B (Sadece test gürültüsü) 10 tohum sonuçları (ortalama ± std)")
print("="*60)
print(f"{'σ':<8} {'MLP MAE':<18} {'MLP R²':<18} {'GCN MAE':<18} {'GCN R²':<18}")
print("-"*80)
for sigma in noise_levels:
    mae_mlp_list = [r[0] for r in results[sigma]['MLP']]
    r2_mlp_list = [r[1] for r in results[sigma]['MLP']]
    mae_gcn_list = [r[0] for r in results[sigma]['GCN']]
    r2_gcn_list = [r[1] for r in results[sigma]['GCN']]
    
    mae_mlp_mean, mae_mlp_std = np.mean(mae_mlp_list), np.std(mae_mlp_list)
    r2_mlp_mean, r2_mlp_std = np.mean(r2_mlp_list), np.std(r2_mlp_list)
    mae_gcn_mean, mae_gcn_std = np.mean(mae_gcn_list), np.std(mae_gcn_list)
    r2_gcn_mean, r2_gcn_std = np.mean(r2_gcn_list), np.std(r2_gcn_list)
    
    print(f"{sigma:<8} {mae_mlp_mean:.4f}±{mae_mlp_std:.4f}   {r2_mlp_mean:.4f}±{r2_mlp_std:.4f}   {mae_gcn_mean:.4f}±{mae_gcn_std:.4f}   {r2_gcn_mean:.4f}±{r2_gcn_std:.4f}")

# CSV kaydı
import pandas as pd
rows = []
for sigma in noise_levels:
    for idx, seed in enumerate(seeds):
        mae_mlp, r2_mlp = results[sigma]['MLP'][idx]
        mae_gcn, r2_gcn = results[sigma]['GCN'][idx]
        rows.append({'sigma': sigma, 'seed': seed, 'model': 'MLP', 'MAE': mae_mlp, 'R2': r2_mlp})
        rows.append({'sigma': sigma, 'seed': seed, 'model': 'GCN', 'MAE': mae_gcn, 'R2': r2_gcn})
df = pd.DataFrame(rows)
df.to_csv('/home/msoylu/alper/analysis/results/scenario_b_10seeds.csv', index=False)
print("\nSonuçlar '/home/msoylu/alper/analysis/results/scenario_b_10seeds.csv' kaydedildi.")