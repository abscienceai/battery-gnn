import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import pickle
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import copy

# -------------------------------
# 1. Model Tanımları (MLP ve GCN)
# -------------------------------
class MLP_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze()

class GCN_RUL(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.rul_head = nn.Linear(hidden_dim, 1)
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return self.rul_head(x).squeeze()

# -------------------------------
# 2. Veriyi Yükleme ve RUL Hedefi Ekleme
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    all_train = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    all_test = pickle.load(f)

print(f"Train graf: {len(all_train)}, Test graf: {len(all_test)}")

def add_rul_target(graph):
    """Her düğüm için RUL = kalan döngü sayısı (toplam_döngü - mevcut_index - 1)"""
    num_nodes = graph.x.shape[0]
    rul = torch.zeros(num_nodes)
    for i in range(num_nodes):
        rul[i] = num_nodes - i - 1
    graph.rul = rul
    return graph

for g in all_train:
    add_rul_target(g)
for g in all_test:
    add_rul_target(g)

# -------------------------------
# 3. Normalizasyon (Node feature'ları için)
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

normalize_features(all_train, all_test)

# -------------------------------
# 4. Eğitim ve Değerlendirme Fonksiyonları
# -------------------------------
def train_mlp(model, train_loader, test_loader, epochs=50, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_mae = float('inf')
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f'MLP Epoch {epoch}', leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.x)
            loss = criterion(pred, data.rul.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                pred = model(data.x)
                preds.append(pred.cpu().numpy())
                trues.append(data.rul.cpu().numpy())
        pred = np.concatenate(preds)
        true = np.concatenate(trues)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, MAE={mae:.2f}, R²={r2:.4f}")
    model.load_state_dict(best_state)
    # Final evaluation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data.x)
            preds.append(pred.cpu().numpy())
            trues.append(data.rul.cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    return mean_absolute_error(true, pred), r2_score(true, pred)

def train_gcn(model, train_loader, test_loader, epochs=50, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_mae = float('inf')
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f'GCN Epoch {epoch}', leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)
            loss = criterion(pred, data.rul.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                pred = model(data.x, data.edge_index)
                preds.append(pred.cpu().numpy())
                trues.append(data.rul.cpu().numpy())
        pred = np.concatenate(preds)
        true = np.concatenate(trues)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, MAE={mae:.2f}, R²={r2:.4f}")
    model.load_state_dict(best_state)
    preds, trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index)
            preds.append(pred.cpu().numpy())
            trues.append(data.rul.cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    return mean_absolute_error(true, pred), r2_score(true, pred)

def train_sklearn(model, train_graphs, test_graphs):
    """Ridge ve XGBoost için eğitim: tüm düğüm özelliklerini ve RUL hedeflerini düzleştir"""
    X_train = []
    y_train = []
    for g in train_graphs:
        X_train.append(g.x.numpy())
        y_train.append(g.rul.numpy().flatten())
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    X_test = []
    y_test = []
    for g in test_graphs:
        X_test.append(g.x.numpy())
        y_test.append(g.rul.numpy().flatten())
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    return mae, r2

# -------------------------------
# 5. Ana Döngü
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50

# DataLoader'lar
train_loader = DataLoader(all_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(all_test, batch_size=batch_size, shuffle=False)

results = {}

# 5.1 MLP
print("\n" + "="*60)
print("MLP eğitiliyor...")
mlp = MLP_RUL(input_dim=all_train[0].x.shape[1])
mae_mlp, r2_mlp = train_mlp(mlp, train_loader, test_loader, epochs, device)
results['MLP'] = (mae_mlp, r2_mlp)
print(f"MLP -> MAE={mae_mlp:.2f}, R²={r2_mlp:.4f}")

# 5.2 GCN
print("\n" + "="*60)
print("GCN eğitiliyor...")
gcn = GCN_RUL(node_in_dim=all_train[0].x.shape[1])
mae_gcn, r2_gcn = train_gcn(gcn, train_loader, test_loader, epochs, device)
results['GCN'] = (mae_gcn, r2_gcn)
print(f"GCN -> MAE={mae_gcn:.2f}, R²={r2_gcn:.4f}")

# 5.3 Ridge
print("\n" + "="*60)
print("Ridge eğitiliyor...")
ridge = Ridge(alpha=1.0)
mae_ridge, r2_ridge = train_sklearn(ridge, all_train, all_test)
results['Ridge'] = (mae_ridge, r2_ridge)
print(f"Ridge -> MAE={mae_ridge:.2f}, R²={r2_ridge:.4f}")

# 5.4 XGBoost
print("\n" + "="*60)
print("XGBoost eğitiliyor...")
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
mae_xgb, r2_xgb = train_sklearn(xgb, all_train, all_test)
results['XGBoost'] = (mae_xgb, r2_xgb)
print(f"XGBoost -> MAE={mae_xgb:.2f}, R²={r2_xgb:.4f}")

# -------------------------------
# 6. Sonuçları Tablo Halinde Yazdır
# -------------------------------
print("\n" + "="*60)
print("RUL Tahmin Performansı Karşılaştırması (MAE / R²)")
print("="*60)
print(f"{'Model':<10} {'MAE (döngü)':<15} {'R²':<10}")
print("-"*60)
for model, (mae, r2) in results.items():
    print(f"{model:<10} {mae:<15.2f} {r2:<10.4f}")

# CSV kaydı
import pandas as pd
df = pd.DataFrame([{'Model': k, 'MAE': v[0], 'R2': v[1]} for k, v in results.items()])
df.to_csv('/home/msoylu/alper/analysis/results/rul_all_models.csv', index=False)
print("\nSonuçlar '/home/msoylu/alper/analysis/results/rul_all_models.csv' kaydedildi.")