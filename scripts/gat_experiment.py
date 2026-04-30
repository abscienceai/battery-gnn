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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import copy
import random

# -------------------------------
# 1. Model: 3 katmanlı GAT + Edge Dropout
# -------------------------------
class GAT_SOH_RUL(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, heads=4, dropout=0.1):
        super().__init__()
        # İlk iki katmanda concat=True (çıktı boyutu heads * hidden_per_head)
        hidden_per_head = hidden_dim // heads  # 32
        self.gat1 = GATConv(node_in_dim, hidden_per_head, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_per_head, heads=heads, dropout=dropout, concat=True)
        # Son katmanda concat=False, çıktı boyutu = hidden_per_head * 1 = 32, ama hidden_dim=128 istiyorsak?
        # Kullanıcı "hidden dim = 128 (head başına 32)" demiş, yani toplam 128. Son katman concat=False yaparsak çıktı 32 olur.
        # Bu nedenle son katmanı concat=True ile 4 head, 32*4=128 çıktı alalım. Alternatif: hidden_per_head=128, heads=1.
        # Daha basit: son katmanı heads=1, hidden_per_head=128 yapalım. Ama tutarlılık için 3 katman da heads=4 kullanalım.
        # O zaman son katman concat=True -> 128 çıktı.
        self.gat3 = GATConv(hidden_dim, hidden_per_head, heads=heads, dropout=dropout, concat=True)
        
        # SOH ve RUL regresyon başlıkları (128 → 64 → 1)
        self.soh_fc1 = nn.Linear(hidden_dim, 64)
        self.soh_fc2 = nn.Linear(64, 1)
        self.rul_fc1 = nn.Linear(hidden_dim, 64)
        self.rul_fc2 = nn.Linear(64, 1)
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_dropout_prob=0.0, training=True):
        # Edge dropout (sadece eğitimde)
        if training and edge_dropout_prob > 0:
            edge_index, _ = dropout_edge(edge_index, p=edge_dropout_prob, training=True)
        
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = self.gat3(x, edge_index)
        
        soh = F.relu(self.soh_fc1(x))
        soh = self.soh_fc2(soh).squeeze()
        rul = F.relu(self.rul_fc1(x))
        rul = self.rul_fc2(rul).squeeze()
        return soh, rul

# -------------------------------
# 2. Veri Yükleme ve RUL Hedefi Ekleme
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    all_train = pickle.load(f)
with open(os.path.join(data_dir, 'val_graphs_no_target.pkl'), 'rb') as f:
    all_val = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    all_test = pickle.load(f)

print(f"Train: {len(all_train)}, Val: {len(all_val)}, Test: {len(all_test)}")

def add_rul_target(graph):
    num_nodes = graph.x.shape[0]
    rul = torch.zeros(num_nodes)
    for i in range(num_nodes):
        rul[i] = num_nodes - i - 1
    graph.rul = rul
    return graph

for g in all_train + all_val + all_test:
    add_rul_target(g)

# Node feature normalizasyonu (eğitim setine göre)
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

scaler = normalize_features(all_train, all_val, all_test)

# -------------------------------
# 3. Eğitim ve Değerlendirme Fonksiyonu
# -------------------------------
def train_model(model, train_loader, val_loader, test_loader, epochs=50, device='cpu',
                edge_dropout_prob=0.1, seed=42):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    soh_criterion = nn.MSELoss()
    rul_criterion = nn.MSELoss()
    
    best_val_mae_soh = float('inf')
    best_state = None
    best_epoch = 0
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f'Seed {seed} Epoch {epoch}', leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            pred_soh, pred_rul = model(data.x, data.edge_index, edge_dropout_prob, training=True)
            loss_soh = soh_criterion(pred_soh, data.y[:, 0].view(-1))
            loss_rul = rul_criterion(pred_rul, data.rul.view(-1))
            loss = loss_soh + 0.01 * loss_rul
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_pred_soh, val_true_soh = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred_soh, _ = model(data.x, data.edge_index, edge_dropout_prob=0.0, training=False)
                val_pred_soh.append(pred_soh.cpu().numpy())
                val_true_soh.append(data.y[:, 0].cpu().numpy())
        pred = np.concatenate(val_pred_soh)
        true = np.concatenate(val_true_soh)
        val_mae_soh = mean_absolute_error(true, pred)
        val_r2_soh = r2_score(true, pred)
        
        if val_mae_soh < best_val_mae_soh:
            best_val_mae_soh = val_mae_soh
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        
        if epoch % 10 == 0:
            print(f"Seed {seed} Epoch {epoch}: Val MAE SOH = {val_mae_soh:.4f}, R² = {val_r2_soh:.4f}")
    
    # Best model evaluation on test set
    model.load_state_dict(best_state)
    model.eval()
    test_pred_soh, test_true_soh = [], []
    test_pred_rul, test_true_rul = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred_soh, pred_rul = model(data.x, data.edge_index, edge_dropout_prob=0.0, training=False)
            test_pred_soh.append(pred_soh.cpu().numpy())
            test_true_soh.append(data.y[:, 0].cpu().numpy())
            test_pred_rul.append(pred_rul.cpu().numpy())
            test_true_rul.append(data.rul.cpu().numpy())
    pred_soh = np.concatenate(test_pred_soh)
    true_soh = np.concatenate(test_true_soh)
    pred_rul = np.concatenate(test_pred_rul)
    true_rul = np.concatenate(test_true_rul)
    
    mae_soh = mean_absolute_error(true_soh, pred_soh)
    rmse_soh = np.sqrt(mean_squared_error(true_soh, pred_soh))
    r2_soh = r2_score(true_soh, pred_soh)
    mae_rul = mean_absolute_error(true_rul, pred_rul)
    r2_rul = r2_score(true_rul, pred_rul)
    
    return {
        'seed': seed,
        'soh_mae': float(mae_soh),
        'soh_rmse': float(rmse_soh),
        'soh_r2': float(r2_soh),
        'rul_mae': float(mae_rul),
        'rul_r2': float(r2_rul),
        'best_epoch': best_epoch
    }

# -------------------------------
# 4. Ana Döngü: 10 tohum
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50
edge_dropout_prob = 0.1
seeds = [5782, 327, 984, 2430, 3382, 8039, 6182, 5398, 6699, 7656]

results = []

for seed in seeds:
    print(f"\n{'='*60}\nTohum: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # DataLoader'lar
    train_loader = DataLoader(all_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(all_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(all_test, batch_size=batch_size, shuffle=False)
    
    model = GAT_SOH_RUL(node_in_dim=all_train[0].x.shape[1], hidden_dim=128, heads=4, dropout=0.1)
    res = train_model(model, train_loader, val_loader, test_loader, epochs, device,
                      edge_dropout_prob, seed)
    results.append(res)
    print(f"Seed {seed} Test: SOH MAE={res['soh_mae']:.4f}, R²={res['soh_r2']:.4f}; RUL MAE={res['rul_mae']:.2f}, R²={res['rul_r2']:.4f}")

# -------------------------------
# 5. Sonuçları Kaydet
# -------------------------------
output_path = '/home/msoylu/alper/analysis/results/gat_10seeds.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSonuçlar kaydedildi: {output_path}")

# Ayrıca bir özet tablosu yazdır
print("\n" + "="*60)
print("GAT + Edge Dropout 10 Tohum Sonuçları (Ortalama ± Std)")
print("="*60)
soh_mae_list = [r['soh_mae'] for r in results]
soh_r2_list = [r['soh_r2'] for r in results]
rul_mae_list = [r['rul_mae'] for r in results]
rul_r2_list = [r['rul_r2'] for r in results]

print(f"SOH MAE: {np.mean(soh_mae_list):.4f} ± {np.std(soh_mae_list):.4f}")
print(f"SOH R²:  {np.mean(soh_r2_list):.4f} ± {np.std(soh_r2_list):.4f}")
print(f"RUL MAE: {np.mean(rul_mae_list):.2f} ± {np.std(rul_mae_list):.2f}")
print(f"RUL R²:  {np.mean(rul_r2_list):.4f} ± {np.std(rul_r2_list):.4f}")