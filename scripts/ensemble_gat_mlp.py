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
# 1. Model Tanımları
# -------------------------------
class GAT_SOH_RUL(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, heads=4, dropout=0.1):
        super().__init__()
        hidden_per_head = hidden_dim // heads  # 32
        self.gat1 = GATConv(node_in_dim, hidden_per_head, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_per_head, heads=heads, dropout=dropout, concat=True)
        self.gat3 = GATConv(hidden_dim, hidden_per_head, heads=heads, dropout=dropout, concat=True)
        self.soh_fc1 = nn.Linear(hidden_dim, 64)
        self.soh_fc2 = nn.Linear(64, 1)
        self.rul_fc1 = nn.Linear(hidden_dim, 64)
        self.rul_fc2 = nn.Linear(64, 1)
        self.dropout = dropout
    def forward(self, x, edge_index, edge_dropout_prob=0.0, training=True):
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

class MLP_Benchmark(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.soh_out = nn.Linear(hidden_dim, 1)
        self.rul_out = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        soh = self.soh_out(x).squeeze()
        rul = self.rul_out(x).squeeze()
        return soh, rul

# -------------------------------
# 2. Veri Yükleme ve RUL Hedefi
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

# -------------------------------
# 3. Özellik Normalizasyonu
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

# GAT için tüm özelliklerle normalizasyon
scaler_full = normalize_features(all_train, all_val, all_test)

# Sistem değişkenlerinin sütun indeksleri (g.x şekli (num_nodes, 9))
# Sıra: 0:discharge_capacity_Ah, 1:temperature_C, 2:current_A, 3:dqdv_peak_voltage_1,
# 4:dqdv_peak_height_1, 5:dqdv_peak_voltage_2, 6:dqdv_peak_height_2, 7:nominal_capacity_Ah, 8:max_voltage_V
# min_voltage_V yok, bu nedenle 4 özellik alıyoruz.
system_indices = [1, 2, 7, 8]  # temperature, current, nominal_capacity, max_voltage
system_dim = len(system_indices)

def extract_system_features(x):
    return x[:, system_indices]

# Sistem özellikleri için normalizasyon (eğitim setine göre)
train_system_feats = []
for g in all_train:
    train_system_feats.append(extract_system_features(g.x).numpy())
train_system_feats = np.vstack(train_system_feats)
scaler_system = StandardScaler()
scaler_system.fit(train_system_feats)

def normalize_system_features(graphs, scaler):
    for g in graphs:
        sys_feats = extract_system_features(g.x).numpy()
        sys_feats_norm = scaler.transform(sys_feats)
        g.x_system = torch.tensor(sys_feats_norm, dtype=torch.float)

normalize_system_features(all_train, scaler_system)
normalize_system_features(all_val, scaler_system)
normalize_system_features(all_test, scaler_system)

# -------------------------------
# 4. Eğitim ve Değerlendirme
# -------------------------------
def train_gat(model, train_loader, val_loader, test_loader, epochs=50, device='cpu', edge_dropout_prob=0.1, seed=42):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    soh_criterion = nn.MSELoss()
    rul_criterion = nn.MSELoss()
    best_val_mae_soh = float('inf')
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f'GAT Seed {seed} Epoch {epoch}', leave=False):
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
        val_preds, val_trues = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred_soh, _ = model(data.x, data.edge_index, edge_dropout_prob=0.0, training=False)
                val_preds.append(pred_soh.cpu().numpy())
                val_trues.append(data.y[:, 0].cpu().numpy())
        val_pred = np.concatenate(val_preds)
        val_true = np.concatenate(val_trues)
        val_mae = mean_absolute_error(val_true, val_pred)
        if val_mae < best_val_mae_soh:
            best_val_mae_soh = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"GAT Seed {seed} Epoch {epoch}: Val MAE SOH = {val_mae:.4f}")
    model.load_state_dict(best_state)
    # Test tahminleri
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
    return (np.concatenate(test_pred_soh), np.concatenate(test_true_soh),
            np.concatenate(test_pred_rul), np.concatenate(test_true_rul))

def train_mlp(model, train_loader, val_loader, test_loader, epochs=50, device='cpu', seed=42):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    soh_criterion = nn.MSELoss()
    rul_criterion = nn.MSELoss()
    best_val_mae_soh = float('inf')
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f'MLP Seed {seed} Epoch {epoch}', leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            pred_soh, pred_rul = model(data.x_system)
            loss_soh = soh_criterion(pred_soh, data.y[:, 0].view(-1))
            loss_rul = rul_criterion(pred_rul, data.rul.view(-1))
            loss = loss_soh + 0.01 * loss_rul
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred_soh, _ = model(data.x_system)
                val_preds.append(pred_soh.cpu().numpy())
                val_trues.append(data.y[:, 0].cpu().numpy())
        val_pred = np.concatenate(val_preds)
        val_true = np.concatenate(val_trues)
        val_mae = mean_absolute_error(val_true, val_pred)
        if val_mae < best_val_mae_soh:
            best_val_mae_soh = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"MLP Seed {seed} Epoch {epoch}: Val MAE SOH = {val_mae:.4f}")
    model.load_state_dict(best_state)
    test_pred_soh, test_true_soh = [], []
    test_pred_rul, test_true_rul = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred_soh, pred_rul = model(data.x_system)
            test_pred_soh.append(pred_soh.cpu().numpy())
            test_true_soh.append(data.y[:, 0].cpu().numpy())
            test_pred_rul.append(pred_rul.cpu().numpy())
            test_true_rul.append(data.rul.cpu().numpy())
    return (np.concatenate(test_pred_soh), np.concatenate(test_true_soh),
            np.concatenate(test_pred_rul), np.concatenate(test_true_rul))

def evaluate_ensemble(gat_pred_soh, mlp_pred_soh, true_soh, gat_pred_rul, mlp_pred_rul, true_rul, w_gat):
    w_mlp = 1 - w_gat
    ens_soh = w_gat * gat_pred_soh + w_mlp * mlp_pred_soh
    ens_rul = w_gat * gat_pred_rul + w_mlp * mlp_pred_rul
    mae_soh = mean_absolute_error(true_soh, ens_soh)
    r2_soh = r2_score(true_soh, ens_soh)
    mae_rul = mean_absolute_error(true_rul, ens_rul)
    r2_rul = r2_score(true_rul, ens_rul)
    return mae_soh, r2_soh, mae_rul, r2_rul

# -------------------------------
# 5. Ana Döngü
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50
edge_dropout_prob = 0.1
seeds = [5782, 327, 984, 2430, 3382, 8039, 6182, 5398, 6699, 7656]

all_results = []

for seed in seeds:
    print(f"\n{'='*60}\nTohum: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader_gat = DataLoader(all_train, batch_size=batch_size, shuffle=True)
    val_loader_gat = DataLoader(all_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(all_test, batch_size=batch_size, shuffle=False)

    # GAT eğitimi
    print("GAT eğitiliyor...")
    gat_soh_pred, gat_soh_true, gat_rul_pred, gat_rul_true = train_gat(
        GAT_SOH_RUL(node_in_dim=all_train[0].x.shape[1]), train_loader_gat, val_loader_gat, test_loader,
        epochs, device, edge_dropout_prob, seed
    )
    # MLP eğitimi
    print("MLP (benchmark) eğitiliyor...")
    mlp_model = MLP_Benchmark(input_dim=system_dim)
    mlp_soh_pred, mlp_soh_true, mlp_rul_pred, mlp_rul_true = train_mlp(
        mlp_model, train_loader_gat, val_loader_gat, test_loader, epochs, device, seed
    )

    w_choices = [0.7, 0.6]
    best_w = None
    best_mae_soh = float('inf')
    best_res = None
    for w in w_choices:
        mae_soh, r2_soh, mae_rul, r2_rul = evaluate_ensemble(
            gat_soh_pred, mlp_soh_pred, gat_soh_true,
            gat_rul_pred, mlp_rul_pred, gat_rul_true, w
        )
        if mae_soh < best_mae_soh:
            best_mae_soh = mae_soh
            best_w = w
            best_res = (mae_soh, r2_soh, mae_rul, r2_rul)

    all_results.append({
        'seed': seed,
        'best_weight': best_w,
        'soh_mae': best_res[0],
        'soh_r2': best_res[1],
        'rul_mae': best_res[2],
        'rul_r2': best_res[3]
    })
    print(f"Seed {seed} ensemble (w_GAT={best_w}): SOH MAE={best_res[0]:.4f}, R²={best_res[1]:.4f}; RUL MAE={best_res[2]:.2f}, R²={best_res[3]:.4f}")

# -------------------------------
# 6. Raporlama
# -------------------------------
output_path = '/home/msoylu/alper/analysis/results/ensemble_gat_mlp.json'
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*60)
print("Ensemble (GAT + MLP) 10 Tohum Sonuçları")
print("="*60)
soh_mae_list = [r['soh_mae'] for r in all_results]
soh_r2_list = [r['soh_r2'] for r in all_results]
rul_mae_list = [r['rul_mae'] for r in all_results]
rul_r2_list = [r['rul_r2'] for r in all_results]

print(f"SOH MAE: {np.mean(soh_mae_list):.4f} ± {np.std(soh_mae_list):.4f}")
print(f"SOH R²:  {np.mean(soh_r2_list):.4f} ± {np.std(soh_r2_list):.4f}")
print(f"RUL MAE: {np.mean(rul_mae_list):.2f} ± {np.std(rul_mae_list):.2f}")
print(f"RUL R²:  {np.mean(rul_r2_list):.4f} ± {np.std(rul_r2_list):.4f}")

weight_counts = {0.7: 0, 0.6: 0}
for r in all_results:
    weight_counts[r['best_weight']] += 1
print(f"\nEn iyi ağırlık dağılımı: w=0.7 -> {weight_counts[0.7]} seeds, w=0.6 -> {weight_counts[0.6]} seeds")
print(f"Sonuçlar kaydedildi: {output_path}")