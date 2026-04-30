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

# Temiz eğitim ve test verilerini normalize et (eğitim setine göre) - sadece bir kere yap, sonra kopyala
normalize_features(all_train, all_test)
clean_test = copy.deepcopy(all_test)
clean_train = copy.deepcopy(all_train)

def add_gaussian_noise(graphs, sigma):
    for g in graphs:
        noise = torch.randn_like(g.x) * sigma
        g.x = g.x + noise
    return graphs

# -------------------------------
# 3. Eğitim ve Değerlendirme Fonksiyonları
# -------------------------------
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
# 4. Ana Döngü: 3 farklı tohum ile
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50

noise_levels = [0.0, 0.01, 0.05, 0.10]
seeds = [42, 123, 456]

# Sonuçları toplamak için sözlük: {sigma: {'MLP': [(mae,r2), ...], 'GCN': [...]}}
results_by_seed = {sigma: {'MLP': [], 'GCN': []} for sigma in noise_levels}

for seed in seeds:
    print(f"\n{'='*60}")
    print(f"Tohum: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Temiz eğitim verisini kopyala (her seed için aynı başlangıç)
    train_graphs = copy.deepcopy(clean_train)
    test_graphs = copy.deepcopy(clean_test)

    # Eğitim setini kullanarak modelleri eğit (temiz)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

    print("MLP eğitiliyor (temiz veri)...")
    mlp = MLP(node_in_dim=train_graphs[0].x.shape[1])
    mlp = train_model(mlp, train_loader, epochs, device)

    print("GCN eğitiliyor (temiz veri)...")
    gcn = NodeLevelGCN(node_in_dim=train_graphs[0].x.shape[1])
    gcn = train_model(gcn, train_loader, epochs, device)

    # Her gürültü seviyesi için test setine gürültü ekle ve değerlendir
    for sigma in noise_levels:
        test_noisy = copy.deepcopy(clean_test)
        if sigma > 0:
            test_noisy = add_gaussian_noise(test_noisy, sigma)
        test_loader = DataLoader(test_noisy, batch_size=batch_size, shuffle=False)

        mae_mlp, r2_mlp = evaluate(mlp, test_loader, device)
        mae_gcn, r2_gcn = evaluate(gcn, test_loader, device)

        results_by_seed[sigma]['MLP'].append((mae_mlp, r2_mlp))
        results_by_seed[sigma]['GCN'].append((mae_gcn, r2_gcn))

        print(f"σ={sigma}: MLP -> MAE={mae_mlp:.4f}, R²={r2_mlp:.4f} | GCN -> MAE={mae_gcn:.4f}, R²={r2_gcn:.4f}")

# -------------------------------
# 5. Ortalama ve std hesaplama
# -------------------------------
print("\n" + "="*60)
print("Senaryo B (Sadece test gürültüsü) 3 tohum sonuçları (ortalama ± std)")
print("="*60)
print(f"{'σ':<8} {'MLP MAE':<20} {'MLP R²':<20} {'GCN MAE':<20} {'GCN R²':<20}")
print("-"*80)

for sigma in noise_levels:
    mlp_maes = [x[0] for x in results_by_seed[sigma]['MLP']]
    mlp_r2s = [x[1] for x in results_by_seed[sigma]['MLP']]
    gcn_maes = [x[0] for x in results_by_seed[sigma]['GCN']]
    gcn_r2s = [x[1] for x in results_by_seed[sigma]['GCN']]

    mlp_mae_mean = np.mean(mlp_maes)
    mlp_mae_std = np.std(mlp_maes)
    mlp_r2_mean = np.mean(mlp_r2s)
    mlp_r2_std = np.std(mlp_r2s)

    gcn_mae_mean = np.mean(gcn_maes)
    gcn_mae_std = np.std(gcn_maes)
    gcn_r2_mean = np.mean(gcn_r2s)
    gcn_r2_std = np.std(gcn_r2s)

    print(f"{sigma:<8} {mlp_mae_mean:.4f}±{mlp_mae_std:.4f}     {mlp_r2_mean:.4f}±{mlp_r2_std:.4f}     {gcn_mae_mean:.4f}±{gcn_mae_std:.4f}     {gcn_r2_mean:.4f}±{gcn_r2_std:.4f}")

# CSV kaydı
import pandas as pd
rows = []
for sigma in noise_levels:
    for i, seed in enumerate(seeds):
        mae_mlp, r2_mlp = results_by_seed[sigma]['MLP'][i]
        mae_gcn, r2_gcn = results_by_seed[sigma]['GCN'][i]
        rows.append({'seed': seed, 'sigma': sigma, 'model': 'MLP', 'MAE': mae_mlp, 'R2': r2_mlp})
        rows.append({'seed': seed, 'sigma': sigma, 'model': 'GCN', 'MAE': mae_gcn, 'R2': r2_gcn})
df = pd.DataFrame(rows)
df.to_csv('/home/msoylu/alper/analysis/results/test_noise_only_seeds.csv', index=False)
print("\nSonuçlar '/home/msoylu/alper/analysis/results/test_noise_only_seeds.csv' kaydedildi.")