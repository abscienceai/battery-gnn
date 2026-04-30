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
import copy

# -------------------------------
# 1. Model tanımları (SOH-only)
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
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze()

# -------------------------------
# 2. Gürültü fonksiyonları (test-time only)
# -------------------------------
def add_gaussian(graphs, sigma=0.05):
    for g in graphs:
        noise = torch.randn_like(g.x) * sigma
        g.x = g.x + noise
    return graphs

def thermal_drift(graphs, drift_rate=0.001, seed=None):
    if seed is not None:
        np.random.seed(seed)
    for g in graphs:
        T = g.x.shape[0]
        t = np.arange(T)
        drift = drift_rate * t + 0.005 * np.sin(2 * np.pi * t / 200)
        drift = drift[:, None]
        g.x = g.x + torch.tensor(drift, dtype=torch.float)
    return graphs

def calibration_offset(graphs, offset_std=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)
    for g in graphs:
        offset = np.random.normal(0, offset_std, size=(1, g.x.shape[1]))
        g.x = g.x + torch.tensor(offset, dtype=torch.float)
    return graphs

def sensor_dropout(graphs, dropout_rate=0.03, seed=None):
    if seed is not None:
        np.random.seed(seed)
    for g in graphs:
        T = g.x.shape[0]
        mask = np.random.rand(T) < dropout_rate
        x_np = g.x.numpy()
        for i in range(1, T):
            if mask[i]:
                x_np[i] = x_np[i-1]
        g.x = torch.tensor(x_np, dtype=torch.float)
    return graphs

def compound_bms_noise(graphs, sigma=0.10, drift_rate=0.001, offset_std=0.05, dropout_rate=0.03, seed=None):
    graphs = thermal_drift(graphs, drift_rate, seed)
    graphs = calibration_offset(graphs, offset_std, seed)
    graphs = add_gaussian(graphs, sigma)
    graphs = sensor_dropout(graphs, dropout_rate, seed)
    return graphs

# -------------------------------
# 3. Veri yükleme ve normalizasyon
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    all_train = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    all_test = pickle.load(f)

all_feats = np.vstack([g.x.numpy() for g in all_train])
scaler = StandardScaler()
scaler.fit(all_feats)

def normalize_graphs(graphs):
    for g in graphs:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    return graphs

test_clean = copy.deepcopy(all_test)
test_clean = normalize_graphs(test_clean)

# -------------------------------
# 4. Eğitilmiş modelleri yükle
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = all_train[0].x.shape[1]

gcn = NodeLevelGCN(input_dim).to(device)
gcn_path = '/home/msoylu/alper/analysis/results/gcn_no_leakage.pth'
gcn.load_state_dict(torch.load(gcn_path, map_location=device))
gcn.eval()

mlp = MLP_SOH(input_dim).to(device)
mlp_path = '/home/msoylu/alper/analysis/results/mlp_no_leakage.pth'
mlp.load_state_dict(torch.load(mlp_path, map_location=device))
mlp.eval()

print("Modeller yüklendi.")

# -------------------------------
# 5. Değerlendirme
# -------------------------------
def evaluate(model, test_loader, is_gcn=True):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            if is_gcn:
                pred = model(data.x, data.edge_index)
            else:
                pred = model(data.x)
            preds.append(pred.cpu().numpy())
            trues.append(data.y[:, 0].cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    return mae, r2

# -------------------------------
# 6. Ana döngü
# -------------------------------
batch_size = 32
seed = 42

noise_configs = [
    ('Gaussian (σ=0.10)', lambda g: add_gaussian(g, sigma=0.10)),
    ('Thermal drift (rate=0.001)', lambda g: thermal_drift(g, drift_rate=0.001, seed=seed)),
    ('Calibration offset (std=0.05)', lambda g: calibration_offset(g, offset_std=0.05, seed=seed)),
    ('Sensor dropout (3%)', lambda g: sensor_dropout(g, dropout_rate=0.03, seed=seed)),
    ('Compound BMS noise', lambda g: compound_bms_noise(g, sigma=0.10, drift_rate=0.001,
                                                         offset_std=0.05, dropout_rate=0.03, seed=seed))
]

results = []

for name, noise_func in noise_configs:
    print(f"\nProcessing: {name}")
    test_noisy = copy.deepcopy(test_clean)
    test_noisy = noise_func(test_noisy)
    test_loader = DataLoader(test_noisy, batch_size=batch_size, shuffle=False)
    mae_mlp, r2_mlp = evaluate(mlp, test_loader, is_gcn=False)
    mae_gcn, r2_gcn = evaluate(gcn, test_loader, is_gcn=True)
    results.append({
        'Noise Type': name,
        'MLP MAE (Ah)': mae_mlp,
        'MLP R²': r2_mlp,
        'GCN MAE (Ah)': mae_gcn,
        'GCN R²': r2_gcn,
        'GCN Advantage (%)': (mae_mlp - mae_gcn) / mae_mlp * 100
    })
    print(f"MLP MAE={mae_mlp:.4f}, GCN MAE={mae_gcn:.4f}, GCN advantage={results[-1]['GCN Advantage (%)']:.1f}%")

df = pd.DataFrame(results)
print("\n" + "="*80)
print("Advanced Noise Robustness Results (test-time noise only)")
print("="*80)
print(df.to_string(index=False))
df.to_csv('/home/msoylu/alper/analysis/results/advanced_noise_results.csv', index=False)
print("\nResults saved to /home/msoylu/alper/analysis/results/advanced_noise_results.csv")