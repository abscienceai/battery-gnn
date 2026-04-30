import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import pickle
import numpy as np
import os
import glob
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random
from scipy.stats import wilcoxon
from scipy.signal import find_peaks
from scipy.stats import linregress
import copy
import time
import re

# -------------------------------
# 1. Model (GCN, SOH only)
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

# -------------------------------
# 2. Veri yükleme
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
# 3. Cache dosyalarını hızlı erişim için map'e yükle
# -------------------------------
cache_dir = '/home/msoylu/alper/analysis/dqdv_cache/'
cache_files = glob.glob(os.path.join(cache_dir, '*.npz'))
print(f"Total cache files: {len(cache_files)}")

domain_names = {0: 'MATR', 1: 'CALCE', 2: 'HUST', 3: 'RWTH'}
domain_to_prefix = {
    'MATR': 'MATR_',
    'CALCE': 'CALCE_',
    'HUST': 'HUST_',
    'RWTH': 'RWTH_'
}

# Ön işleme: her cache dosyasından (domain, cycle_num) çıkar
cache_map = {}  # key: (domain, cycle_index) -> filepath
for fpath in cache_files:
    fname = os.path.basename(fpath)
    # Örn: "CALCE_CS2_33.pkl_100.npz"
    # veya "MATR_XXX_123.npz"
    # Domain prefix ile başlıyor
    for domain, prefix in domain_to_prefix.items():
        if fname.startswith(prefix):
            # cycle numarasını al: son '_' den sonra gelen sayı, .npz kaldır
            parts = fname.replace('.npz', '').split('_')
            cycle_str = parts[-1]
            try:
                cycle_num = int(cycle_str)
                cache_map[(domain, cycle_num)] = fpath
                break
            except:
                pass
print(f"Cache map built with {len(cache_map)} entries.")

def find_cache_file_fast(domain, cycle_idx):
    """cycle_idx 0-index -> cycle num = cycle_idx+1"""
    cycle_num = cycle_idx + 1
    return cache_map.get((domain, cycle_num), None)

def load_dqdv_from_cache(cache_path):
    data = np.load(cache_path)
    if 'dqdv' in data:
        return data['dqdv']
    else:
        return data['dqdv'] if 'dqdv' in data.files else None

# -------------------------------
# 4. Özellik çıkarma (normalize eğriden)
# -------------------------------
def extract_features_nmc(dqdv_curve, voltage_vals=None):
    peaks, props = find_peaks(dqdv_curve, height=0.005, distance=30)
    if len(peaks) >= 2:
        peak_voltage_1 = peaks[0]
        peak_height_1 = props['peak_heights'][0]
        peak_voltage_2 = peaks[1]
        peak_height_2 = props['peak_heights'][1]
    elif len(peaks) == 1:
        peak_voltage_1 = peaks[0]
        peak_height_1 = props['peak_heights'][0]
        peak_voltage_2 = 0.0
        peak_height_2 = 0.0
    else:
        peak_voltage_1 = peak_height_1 = peak_voltage_2 = peak_height_2 = 0.0
    return peak_voltage_1, peak_height_1, peak_voltage_2, peak_height_2

def extract_features_lfp(dqdv_curve, voltage_vals):
    if voltage_vals is None:
        return 0.0
    mask = (voltage_vals >= 3.2) & (voltage_vals <= 3.4)
    if np.sum(mask) < 2:
        return 0.0
    x = voltage_vals[mask]
    y = dqdv_curve[mask]
    slope, _, _, _, _ = linregress(x, y)
    return slope

def extract_features_lco(dqdv_curve, voltage_vals):
    if voltage_vals is None:
        return 0.0
    mask = (voltage_vals >= 3.8) & (voltage_vals <= 4.0)
    if np.sum(mask) < 2:
        return 0.0
    y = dqdv_curve[mask]
    peak_idx = np.argmax(y)
    peak_val = y[peak_idx]
    half_max = peak_val / 2.0
    left = np.where(y[:peak_idx] <= half_max)[0]
    right = np.where(y[peak_idx:] <= half_max)[0]
    left_idx = left[-1] if len(left) > 0 else 0
    right_idx = peak_idx + right[0] if len(right) > 0 else len(y)-1
    fwhm = voltage_vals[mask][right_idx] - voltage_vals[mask][left_idx]
    return fwhm

# -------------------------------
# 5. Graf güncelleme (sadece RWTH için)
# -------------------------------
def update_graph_with_normalized_dqdv(graph, domain_name):
    if domain_name != 'RWTH':
        return graph
    num_nodes = graph.x.shape[0]
    for node_idx in range(num_nodes):
        cache_path = find_cache_file_fast(domain_name, node_idx)
        if cache_path is None:
            continue
        data = np.load(cache_path)
        dqdv = data['dqdv']
        voltage = data['voltage']
        max_abs = np.max(np.abs(dqdv))
        if max_abs > 0:
            dqdv_norm = dqdv / max_abs
        else:
            dqdv_norm = dqdv
        pv1, ph1, pv2, ph2 = extract_features_nmc(dqdv_norm, voltage)
        graph.x[node_idx, 3] = pv1
        graph.x[node_idx, 4] = ph1
        graph.x[node_idx, 5] = pv2
        graph.x[node_idx, 6] = ph2
    return graph

def zero_out_dqdv_features(graph):
    graph.x[:, 3:7] = 0.0
    return graph

# -------------------------------
# 6. Eğitim fonksiyonu
# -------------------------------
def train_evaluate(train_graphs, val_graphs, test_graphs, epochs=50, seed=42, condition_name=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    
    # Normalizasyon
    all_feats = np.vstack([g.x.numpy() for g in train_graphs])
    scaler = StandardScaler().fit(all_feats)
    for g in train_graphs: g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    for g in val_graphs:   g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    for g in test_graphs:  g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size, shuffle=False)
    
    model = NodeLevelGCN(train_graphs[0].x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val_mae = float('inf')
    best_state = None
    start_time = time.time()
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f'{condition_name} Seed {seed} Epoch {epoch}', leave=False)
        for data in loop:
            data = data.to(device)
            opt.zero_grad()
            pred = model(data.x, data.edge_index)
            loss = criterion(pred, data.y[:,0].view(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        # validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred = model(data.x, data.edge_index)
                val_preds.append(pred.cpu().numpy())
                val_trues.append(data.y[:,0].cpu().numpy())
        val_mae = mean_absolute_error(np.concatenate(val_trues), np.concatenate(val_preds))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"{condition_name} Seed {seed} Epoch {epoch}: Val MAE = {val_mae:.4f}")
    model.load_state_dict(best_state)
    # test
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index)
            test_preds.append(pred.cpu().numpy())
            test_trues.append(data.y[:,0].cpu().numpy())
    pred = np.concatenate(test_preds)
    true = np.concatenate(test_trues)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    elapsed = time.time() - start_time
    print(f"{condition_name} Seed {seed} finished: Test MAE = {mae:.4f}, R² = {r2:.4f} (time {elapsed:.1f}s)")
    return mae, r2

# -------------------------------
# 7. Ana döngü
# -------------------------------
seeds = [5782, 327, 984, 2430, 3382, 8039, 6182, 5398, 6699, 7656]
conditions = ['original_dqdv', 'normalized_dqdv', 'no_dqdv']
all_results = []

for seed in seeds:
    print(f"\n{'='*60}\nSeed: {seed}\n{'='*60}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    # Orijinal
    train_orig = copy.deepcopy(all_train)
    val_orig = copy.deepcopy(all_val)
    test_orig = copy.deepcopy(all_test)
    print("Training original_dqdv...")
    mae_orig, r2_orig = train_evaluate(train_orig, val_orig, test_orig, seed=seed, condition_name="Original")
    all_results.append({'condition':'original_dqdv','seed':seed,'soh_mae':mae_orig,'soh_r2':r2_orig})
    
    # Normalized (sadece RWTH etkilenir)
    train_norm = copy.deepcopy(all_train)
    val_norm = copy.deepcopy(all_val)
    test_norm = copy.deepcopy(all_test)
    print("Applying amplitude normalization to RWTH cells...")
    for g in tqdm(train_norm+val_norm+test_norm, desc="Normalizing graphs"):
        dom_name = domain_names[g.domain.item()] if isinstance(g.domain, torch.Tensor) else domain_names[int(g.domain)]
        update_graph_with_normalized_dqdv(g, dom_name)
    print("Training normalized_dqdv...")
    mae_norm, r2_norm = train_evaluate(train_norm, val_norm, test_norm, seed=seed, condition_name="Normalized")
    all_results.append({'condition':'normalized_dqdv','seed':seed,'soh_mae':mae_norm,'soh_r2':r2_norm})
    
    # No dqdv
    train_zero = copy.deepcopy(all_train)
    val_zero = copy.deepcopy(all_val)
    test_zero = copy.deepcopy(all_test)
    for g in train_zero+val_zero+test_zero:
        zero_out_dqdv_features(g)
    print("Training no_dqdv...")
    mae_zero, r2_zero = train_evaluate(train_zero, val_zero, test_zero, seed=seed, condition_name="No dqdv")
    all_results.append({'condition':'no_dqdv','seed':seed,'soh_mae':mae_zero,'soh_r2':r2_zero})

# -------------------------------
# 8. Kaydet ve Wilcoxon
# -------------------------------
df = pd.DataFrame(all_results)
df.to_csv('/home/msoylu/alper/analysis/results/p2_amplitude_normalized.csv', index=False)
print("\nResults saved to /home/msoylu/alper/analysis/results/p2_amplitude_normalized.csv")

orig_maes = df[df['condition']=='original_dqdv']['soh_mae'].values
norm_maes = df[df['condition']=='normalized_dqdv']['soh_mae'].values
zero_maes = df[df['condition']=='no_dqdv']['soh_mae'].values
stat1, p1 = wilcoxon(orig_maes, norm_maes)
stat2, p2 = wilcoxon(orig_maes, zero_maes)
print(f"\nWilcoxon signed-rank tests:")
print(f"Original vs Normalized: W={stat1}, p={p1:.6f}")
print(f"Original vs No dqdv: W={stat2}, p={p2:.6f}")