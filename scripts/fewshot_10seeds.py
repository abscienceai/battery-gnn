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

domain_names = {0: 'MATR', 1: 'CALCE', 2: 'HUST', 3: 'RWTH'}

# Ayrıştır
src_train = [g for g in all_train if domain_names[g.domain.item()] == 'MATR']
src_val   = [g for g in all_val   if domain_names[g.domain.item()] == 'MATR']
tgt_train_all = [g for g in all_train if domain_names[g.domain.item()] == 'HUST']
tgt_val_all   = [g for g in all_val   if domain_names[g.domain.item()] == 'HUST']
tgt_test_all  = [g for g in all_test  if domain_names[g.domain.item()] == 'HUST']

print(f"Source train: {len(src_train)}, val: {len(src_val)}")
print(f"Target train: {len(tgt_train_all)}, val: {len(tgt_val_all)}, test: {len(tgt_test_all)}")

# -------------------------------
# 1. Özellik normalizasyonu (kaynak eğitim setine göre)
# -------------------------------
src_features = np.vstack([g.x.numpy() for g in src_train])
feature_scaler = StandardScaler()
feature_scaler.fit(src_features)

def normalize_features(graphs, scaler):
    for g in graphs:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    return graphs

src_train_norm = normalize_features(copy.deepcopy(src_train), feature_scaler)
src_val_norm   = normalize_features(copy.deepcopy(src_val),   feature_scaler)
tgt_train_norm = normalize_features(copy.deepcopy(tgt_train_all), feature_scaler)
tgt_val_norm   = normalize_features(copy.deepcopy(tgt_val_all),   feature_scaler)
tgt_test_norm  = normalize_features(copy.deepcopy(tgt_test_all),  feature_scaler)

# -------------------------------
# 2. SOH etiket normalizasyonu (kaynak için kendi mean/std, hedef için hedef mean/std)
# -------------------------------
# Kaynak
src_y = np.concatenate([g.y[:,0].numpy() for g in src_train])
src_y_mean, src_y_std = src_y.mean(), src_y.std()
# Hedef
tgt_y = np.concatenate([g.y[:,0].numpy() for g in tgt_train_all])
tgt_y_mean, tgt_y_std = tgt_y.mean(), tgt_y.std()

def normalize_labels(graphs, mean, std):
    for g in graphs:
        y = g.y.numpy()
        y_norm = (y - mean) / std
        g.y = torch.tensor(y_norm, dtype=torch.float)
    return graphs

def denormalize_labels(y_norm, mean, std):
    return y_norm * std + mean

# Kaynak etiketleri normalleştir
src_train_norm = normalize_labels(src_train_norm, src_y_mean, src_y_std)
src_val_norm   = normalize_labels(src_val_norm,   src_y_mean, src_y_std)
# Hedef etiketleri kendi eğitim seti istatistikleriyle normalleştir
tgt_train_norm = normalize_labels(tgt_train_norm, tgt_y_mean, tgt_y_std)
tgt_val_norm   = normalize_labels(tgt_val_norm,   tgt_y_mean, tgt_y_std)
tgt_test_norm  = normalize_labels(tgt_test_norm,   tgt_y_mean, tgt_y_std)

# -------------------------------
# Eğitim fonksiyonları
# -------------------------------
def train_model(model, train_loader, val_loader, epochs=50, device='cpu', is_gcn=True, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_mae = float('inf')
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
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
    return model

def evaluate(model, test_loader, device='cpu', is_gcn=True, denorm_mean=None, denorm_std=None):
    model.eval()
    preds_norm, trues_norm = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            if is_gcn:
                pred = model(data.x, data.edge_index)
            else:
                pred = model(data.x)
            preds_norm.append(pred.cpu().numpy())
            trues_norm.append(data.y[:,0].cpu().numpy())
    pred_norm = np.concatenate(preds_norm)
    true_norm = np.concatenate(trues_norm)
    if denorm_mean is not None and denorm_std is not None:
        pred = denormalize_labels(pred_norm, denorm_mean, denorm_std)
        true = denormalize_labels(true_norm, denorm_mean, denorm_std)
    else:
        pred, true = pred_norm, true_norm
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    return mae, r2

def finetune_model(model, train_loader, val_loader, test_loader, epochs=30, device='cpu',
                   is_gcn=True, lr=1e-4, denorm_mean=None, denorm_std=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_mae = float('inf')
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        for data in tqdm(train_loader, desc=f'FT Epoch {epoch}', leave=False):
            data = data.to(device)
            opt.zero_grad()
            if is_gcn:
                pred = model(data.x, data.edge_index)
            else:
                pred = model(data.x)
            loss = criterion(pred, data.y[:,0].view(-1))
            loss.backward()
            opt.step()
        # validation (still in normalized space)
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
    # evaluate on test (with denormalization)
    mae, r2 = evaluate(model, test_loader, device, is_gcn, denorm_mean, denorm_std)
    return mae, r2

# -------------------------------
# Ana döngü
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs_src = 50
epochs_ft = 30
seeds = [5782, 327, 984, 2430, 3382, 8039, 6182, 5398, 6699, 7656]
fewshot_ratios = [0.01, 0.05, 0.10]

results = []

for seed in seeds:
    print(f"\nSeed: {seed}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Prepare target training set for fine-tuning
    tgt_train_all_norm = tgt_train_norm  # already normalized with target stats
    tgt_val_norm_full = tgt_val_norm
    tgt_test_norm_full = tgt_test_norm

    # Source model training (on source data)
    # GCN
    src_train_loader = DataLoader(src_train_norm, batch_size=batch_size, shuffle=True)
    src_val_loader   = DataLoader(src_val_norm,   batch_size=batch_size, shuffle=False)
    gcn_src = NodeLevelGCN(src_train_norm[0].x.shape[1]).to(device)
    gcn_src = train_model(gcn_src, src_train_loader, src_val_loader, epochs_src, device, is_gcn=True)
    # MLP
    mlp_src = MLP_SOH(src_train_norm[0].x.shape[1]).to(device)
    mlp_src = train_model(mlp_src, src_train_loader, src_val_loader, epochs_src, device, is_gcn=False)

    # Zero-shot evaluation on target test (denormalize using target stats)
    tgt_test_loader = DataLoader(tgt_test_norm_full, batch_size=batch_size, shuffle=False)
    gcn_zero_mae, gcn_zero_r2 = evaluate(gcn_src, tgt_test_loader, device, is_gcn=True,
                                         denorm_mean=tgt_y_mean, denorm_std=tgt_y_std)
    mlp_zero_mae, mlp_zero_r2 = evaluate(mlp_src, tgt_test_loader, device, is_gcn=False,
                                         denorm_mean=tgt_y_mean, denorm_std=tgt_y_std)
    results.append({'setting': 'zero-shot', 'model': 'GCN', 'seed': seed,
                    'soh_mae': gcn_zero_mae, 'soh_r2': gcn_zero_r2})
    results.append({'setting': 'zero-shot', 'model': 'MLP', 'seed': seed,
                    'soh_mae': mlp_zero_mae, 'soh_r2': mlp_zero_r2})
    print(f"zero-shot: GCN MAE={gcn_zero_mae:.4f}, R²={gcn_zero_r2:.4f} | MLP MAE={mlp_zero_mae:.4f}, R²={mlp_zero_r2:.4f}")

    # Fine-tuning
    for ratio in fewshot_ratios:
        # Select subset of target training data
        num_tgt = len(tgt_train_all_norm)
        num_subset = max(1, int(num_tgt * ratio))
        indices = random.sample(range(num_tgt), num_subset)
        ft_train = [tgt_train_all_norm[i] for i in indices]
        ft_val   = tgt_val_norm_full
        ft_test  = tgt_test_norm_full

        ft_train_loader = DataLoader(ft_train, batch_size=batch_size, shuffle=True)
        ft_val_loader   = DataLoader(ft_val,   batch_size=batch_size, shuffle=False)
        ft_test_loader  = DataLoader(ft_test,  batch_size=batch_size, shuffle=False)

        # GCN fine-tune
        gcn_ft = copy.deepcopy(gcn_src)
        gcn_ft_mae, gcn_ft_r2 = finetune_model(gcn_ft, ft_train_loader, ft_val_loader,
                                               ft_test_loader, epochs_ft, device, is_gcn=True,
                                               lr=1e-4, denorm_mean=tgt_y_mean, denorm_std=tgt_y_std)
        # MLP fine-tune
        mlp_ft = copy.deepcopy(mlp_src)
        mlp_ft_mae, mlp_ft_r2 = finetune_model(mlp_ft, ft_train_loader, ft_val_loader,
                                               ft_test_loader, epochs_ft, device, is_gcn=False,
                                               lr=1e-4, denorm_mean=tgt_y_mean, denorm_std=tgt_y_std)

        results.append({'setting': f'finetune_{int(ratio*100)}%', 'model': 'GCN', 'seed': seed,
                        'soh_mae': gcn_ft_mae, 'soh_r2': gcn_ft_r2})
        results.append({'setting': f'finetune_{int(ratio*100)}%', 'model': 'MLP', 'seed': seed,
                        'soh_mae': mlp_ft_mae, 'soh_r2': mlp_ft_r2})
        print(f"finetune {int(ratio*100)}%: GCN MAE={gcn_ft_mae:.4f}, R²={gcn_ft_r2:.4f} | MLP MAE={mlp_ft_mae:.4f}, R²={mlp_ft_r2:.4f}")

# -------------------------------
# Kaydet ve Wilcoxon
# -------------------------------
df = pd.DataFrame(results)
df.to_csv('/home/msoylu/alper/analysis/results/fewshot_10seeds_fixed.csv', index=False)

print("\nWilcoxon signed-rank tests (GCN vs MLP):")
for setting in ['zero-shot', 'finetune_1%', 'finetune_5%', 'finetune_10%']:
    gcn_maes = df[(df['setting']==setting) & (df['model']=='GCN')]['soh_mae'].values
    mlp_maes = df[(df['setting']==setting) & (df['model']=='MLP')]['soh_mae'].values
    if len(gcn_maes) == 10 and len(mlp_maes) == 10:
        stat, p = wilcoxon(gcn_maes, mlp_maes, alternative='two-sided')
        print(f"{setting}: W={stat}, p={p:.6f}")