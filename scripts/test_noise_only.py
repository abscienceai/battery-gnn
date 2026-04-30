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

# Temiz eğitim ve test verilerini normalize et (eğitim setine göre)
normalize_features(all_train, all_test)

# Test setinin temiz kopyasını sakla
clean_test = copy.deepcopy(all_test)

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
# 4. Ana Döngü: Sadece test gürültüsü
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50

noise_levels = [0.0, 0.01, 0.05, 0.10]
results = {'MLP': {}, 'GCN': {}}

# Temiz eğitim verisi ile modelleri eğit (bir kere, tüm gürültü seviyeleri için aynı model)
train_loader = DataLoader(all_train, batch_size=batch_size, shuffle=True)

print("MLP eğitiliyor (temiz veri ile)...")
mlp = MLP(node_in_dim=all_train[0].x.shape[1])  # DÜZELTİLDİ
mlp = train_model(mlp, train_loader, epochs, device)

print("GCN eğitiliyor (temiz veri ile)...")
gcn = NodeLevelGCN(node_in_dim=all_train[0].x.shape[1])
gcn = train_model(gcn, train_loader, epochs, device)

# Her gürültü seviyesi için test setini boz ve değerlendir
for sigma in noise_levels:
    print(f"\n{'='*60}")
    print(f"Test gürültü seviyesi σ = {sigma}")
    test_graphs = copy.deepcopy(clean_test)
    if sigma > 0:
        test_graphs = add_gaussian_noise(test_graphs, sigma)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    mae_mlp, r2_mlp = evaluate(mlp, test_loader, device)
    mae_gcn, r2_gcn = evaluate(gcn, test_loader, device)
    
    results['MLP'][sigma] = (mae_mlp, r2_mlp)
    results['GCN'][sigma] = (mae_gcn, r2_gcn)
    print(f"MLP -> MAE={mae_mlp:.4f}, R²={r2_mlp:.4f}")
    print(f"GCN -> MAE={mae_gcn:.4f}, R²={r2_gcn:.4f}")

# -------------------------------
# 5. Sonuçları Tablo Halinde Yazdır
# -------------------------------
print("\n" + "="*60)
print("Sadece Test Gürültüsü Deneyi Sonuçları (SOH MAE / R²)")
print("Eğitim: temiz veri, Test: gürültülü veri")
print("="*60)
print(f"{'σ':<8} {'MLP MAE':<12} {'MLP R²':<12} {'GCN MAE':<12} {'GCN R²':<12}")
print("-"*60)
for sigma in noise_levels:
    mae_mlp, r2_mlp = results['MLP'][sigma]
    mae_gcn, r2_gcn = results['GCN'][sigma]
    print(f"{sigma:<8} {mae_mlp:<12.4f} {r2_mlp:<12.4f} {mae_gcn:<12.4f} {r2_gcn:<12.4f}")

# CSV kaydı
import pandas as pd
df = pd.DataFrame([
    {'sigma': s, 'model': 'MLP', 'MAE': results['MLP'][s][0], 'R2': results['MLP'][s][1]} for s in noise_levels
] + [
    {'sigma': s, 'model': 'GCN', 'MAE': results['GCN'][s][0], 'R2': results['GCN'][s][1]} for s in noise_levels
])
df.to_csv('/home/msoylu/alper/analysis/results/test_noise_only.csv', index=False)
print("\nSonuçlar '/home/msoylu/alper/analysis/results/test_noise_only.csv' kaydedildi.")