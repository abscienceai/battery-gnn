import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader
from torch_geometric.utils import dropout_edge
import pickle
import numpy as np
import os
import json
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import copy
import random

# -------------------------------
# 1. GAT Modeli (Deney 1'deki ile aynı)
# -------------------------------
class GAT_SOH(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, heads=4, dropout=0.1):
        super().__init__()
        hidden_per_head = hidden_dim // heads  # 32
        self.gat1 = GATConv(node_in_dim, hidden_per_head, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_per_head, heads=heads, dropout=dropout, concat=True)
        self.gat3 = GATConv(hidden_dim, hidden_per_head, heads=heads, dropout=dropout, concat=True)
        self.soh_fc1 = nn.Linear(hidden_dim, 64)
        self.soh_fc2 = nn.Linear(64, 1)
        self.dropout = dropout
    def forward(self, x, edge_index, edge_dropout_prob=0.0, training=True):
        if training and edge_dropout_prob > 0:
            edge_index, _ = dropout_edge(edge_index, p=edge_dropout_prob, training=True)
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = self.gat3(x, edge_index)
        soh = F.relu(self.soh_fc1(x))
        soh = self.soh_fc2(soh).squeeze()
        return soh

# -------------------------------
# 2. Veri Yükleme
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
# 3. Normalizasyon (temiz eğitim verisine göre)
# -------------------------------
def normalize_features(train_graphs, val_graphs, test_graphs):
    all_feats = []
    for g in train_graphs:
        all_feats.append(g.x.numpy())
    all_feats = np.vstack(all_feats)
    scaler = StandardScaler()
    scaler.fit(all_feats)
    for g in train_graphs:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    for g in val_graphs:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    for g in test_graphs:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    return scaler

# Eğitim, doğrulama ve test setlerini normalize et (temiz, gürültüsüz)
norm_scaler = normalize_features(all_train, all_val, all_test)

# Orijinal test setinin kopyasını sakla (gürültü eklemek için)
clean_test = copy.deepcopy(all_test)

def add_gaussian_noise(graphs, sigma):
    for g in graphs:
        noise = torch.randn_like(g.x) * sigma
        g.x = g.x + noise
    return graphs

# -------------------------------
# 4. Eğitim ve Değerlendirme Fonksiyonları
# -------------------------------
def train_gat_clean(model, train_loader, val_loader, epochs=50, device='cpu', edge_dropout_prob=0.1, seed=42):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val_mae = float('inf')
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f'GAT Seed {seed} Epoch {epoch}', leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            pred_soh = model(data.x, data.edge_index, edge_dropout_prob, training=True)
            loss = criterion(pred_soh, data.y[:, 0].view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred_soh = model(data.x, data.edge_index, edge_dropout_prob=0.0, training=False)
                val_preds.append(pred_soh.cpu().numpy())
                val_trues.append(data.y[:, 0].cpu().numpy())
        val_pred = np.concatenate(val_preds)
        val_true = np.concatenate(val_trues)
        val_mae = mean_absolute_error(val_true, val_pred)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"Seed {seed} Epoch {epoch}: Val MAE = {val_mae:.4f}")
    model.load_state_dict(best_state)
    return model

def evaluate_gat(model, test_loader, device='cpu'):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, edge_dropout_prob=0.0, training=False)
            preds.append(pred.cpu().numpy())
            trues.append(data.y[:, 0].cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    return mae, r2

# -------------------------------
# 5. Ana Döngü: 10 seed, 4 sigma (Senaryo B)
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50
edge_dropout_prob = 0.1
seeds = [5782, 327, 984, 2430, 3382, 8039, 6182, 5398, 6699, 7656]
sigma_levels = [0.0, 0.01, 0.05, 0.10]

# Sonuçları toplamak için sözlük
results = {sigma: {'mae': [], 'r2': []} for sigma in sigma_levels}

for seed in seeds:
    print(f"\n{'='*60}")
    print(f"Tohum: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # DataLoader'lar (temiz eğitim ve doğrulama)
    train_loader = DataLoader(all_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(all_val, batch_size=batch_size, shuffle=False)
    
    # GAT modeli oluştur ve eğit (temiz veriyle)
    model = GAT_SOH(node_in_dim=all_train[0].x.shape[1], hidden_dim=128, heads=4, dropout=0.1)
    model = train_gat_clean(model, train_loader, val_loader, epochs, device, edge_dropout_prob, seed)
    
    # Her sigma için test setine gürültü ekle ve değerlendir
    for sigma in sigma_levels:
        test_graphs = copy.deepcopy(clean_test)
        if sigma > 0:
            test_graphs = add_gaussian_noise(test_graphs, sigma)
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
        mae, r2 = evaluate_gat(model, test_loader, device)
        results[sigma]['mae'].append(mae)
        results[sigma]['r2'].append(r2)
        print(f"σ={sigma}: MAE={mae:.4f}, R²={r2:.4f}")

# -------------------------------
# 6. Ortalama ve standart sapma hesapla
# -------------------------------
output = {}
for sigma in sigma_levels:
    mae_list = results[sigma]['mae']
    r2_list = results[sigma]['r2']
    output[f'sigma_{sigma}'] = {
        'mae_mean': np.mean(mae_list),
        'mae_std': np.std(mae_list),
        'r2_mean': np.mean(r2_list),
        'r2_std': np.std(r2_list)
    }

# JSON olarak kaydet
output_path = '/home/msoylu/alper/analysis/results/gat_noise_scenario_b.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "="*60)
print("GAT Senaryo B (Sadece test gürültüsü) 10 tohum sonuçları (ortalama ± std)")
print("="*60)
for sigma in sigma_levels:
    print(f"σ={sigma}: MAE = {output[f'sigma_{sigma}']['mae_mean']:.4f} ± {output[f'sigma_{sigma}']['mae_std']:.4f}, "
          f"R² = {output[f'sigma_{sigma}']['r2_mean']:.4f} ± {output[f'sigma_{sigma}']['r2_std']:.4f}")

print(f"\nSonuçlar kaydedildi: {output_path}")