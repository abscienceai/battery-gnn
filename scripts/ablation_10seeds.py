import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import pickle
import numpy as np
import os
import json
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import copy
import random
from scipy.stats import wilcoxon

# -------------------------------
# 1. Model Tanımları
# -------------------------------
class NodeLevelGCN(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.soh_head = nn.Linear(hidden_dim, 1)
        self.rul_head = nn.Linear(hidden_dim, 1)
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        soh = self.soh_head(x).squeeze()
        rul = self.rul_head(x).squeeze()
        return soh, rul

class MLP_SameFeatures(nn.Module):
    """No GNN (MLP) — Full model ile aynı özellikleri alır (dQ/dV dahil)"""
    def __init__(self, node_in_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(node_in_dim, hidden_dim)
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

# -------------------------------
# 3. Varyantları Uygulama Fonksiyonları
# -------------------------------
def apply_no_cycle_graph(graphs):
    """Kenarları sil, edge_index boş"""
    for g in graphs:
        g.edge_index = torch.empty((2,0), dtype=torch.long)
    return graphs

def apply_no_dqdv(graphs):
    """dQ/dV sütunlarını (3,4,5,6) sıfırla — indexler: 3:peak_voltage1, 4:peak_height1, 5:peak_voltage2, 6:peak_height2"""
    for g in graphs:
        x = g.x
        x[:, 3:7] = 0.0
        g.x = x
    return graphs

# -------------------------------
# 4. Normalizasyon (eğitim setine göre)
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

# -------------------------------
# 5. Eğitim ve Değerlendirme
# -------------------------------
def train_evaluate(variant_name, train_graphs, val_graphs, test_graphs,
                   train_loader, val_loader, test_loader,
                   epochs=50, device='cpu', seed=42):
    # Normalizasyon (varyantın train,val,test setlerine göre)
    norm_scaler = normalize_features(train_graphs, val_graphs, test_graphs)
    
    # Model seçimi
    sample = train_graphs[0]
    input_dim = sample.x.shape[1]
    if variant_name == 'no_gnn_mlp':
        model = MLP_SameFeatures(input_dim).to(device)
        def forward_fn(data):
            return model(data.x)
    else:
        model = NodeLevelGCN(input_dim).to(device)
        def forward_fn(data):
            return model(data.x, data.edge_index)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    soh_criterion = nn.MSELoss()
    rul_criterion = nn.MSELoss()
    
    best_val_mae = float('inf')
    best_state = None
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f'{variant_name} Seed {seed} Epoch {epoch}', leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            pred_soh, pred_rul = forward_fn(data)
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
                pred_soh, _ = forward_fn(data)
                val_preds.append(pred_soh.cpu().numpy())
                val_trues.append(data.y[:, 0].cpu().numpy())
        val_pred = np.concatenate(val_preds)
        val_true = np.concatenate(val_trues)
        val_mae = mean_absolute_error(val_true, val_pred)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    # Test değerlendirmesi
    model.eval()
    test_soh_pred, test_soh_true = [], []
    test_rul_pred, test_rul_true = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred_soh, pred_rul = forward_fn(data)
            test_soh_pred.append(pred_soh.cpu().numpy())
            test_soh_true.append(data.y[:, 0].cpu().numpy())
            test_rul_pred.append(pred_rul.cpu().numpy())
            test_rul_true.append(data.rul.cpu().numpy())
    soh_pred = np.concatenate(test_soh_pred)
    soh_true = np.concatenate(test_soh_true)
    rul_pred = np.concatenate(test_rul_pred)
    rul_true = np.concatenate(test_rul_true)
    
    mae_soh = mean_absolute_error(soh_true, soh_pred)
    r2_soh = r2_score(soh_true, soh_pred)
    mae_rul = mean_absolute_error(rul_true, rul_pred)
    r2_rul = r2_score(rul_true, rul_pred)
    
    return {'mae_soh': mae_soh, 'r2_soh': r2_soh, 'mae_rul': mae_rul, 'r2_rul': r2_rul}

# -------------------------------
# 6. Ana Döngü: 10 tohum, 4 varyant
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50
seeds = [5782, 327, 984, 2430, 3382, 8039, 6182, 5398, 6699, 7656]

variants = {
    'full_model': {'apply': lambda x: x, 'name': 'Full Model'},
    'no_cycle_graph': {'apply': apply_no_cycle_graph, 'name': 'No Cycle Graph'},
    'no_dqdv': {'apply': apply_no_dqdv, 'name': 'No dQdV Features'},
    'no_gnn_mlp': {'apply': lambda x: x, 'name': 'No GNN (MLP)'}  # MLP ayrı model, özellikler aynı
}

# Results storage with correct metric names
results = {v: {'mae_soh': [], 'r2_soh': [], 'mae_rul': [], 'r2_rul': []} for v in variants.keys()}

for seed in seeds:
    print(f"\n{'='*60}")
    print(f"Tohum: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    for var_key, var_info in variants.items():
        print(f"\n--- Varyant: {var_info['name']} ---")
        # Kopyalar oluştur
        train_cp = copy.deepcopy(all_train)
        val_cp = copy.deepcopy(all_val)
        test_cp = copy.deepcopy(all_test)
        # Varyantı uygula (MLP hariç, çünkü MLP'de edge_index kullanılmaz ama yine de uygulanabilir; fark etmez)
        if var_key != 'no_gnn_mlp':
            train_cp = var_info['apply'](train_cp)
            val_cp = var_info['apply'](val_cp)
            test_cp = var_info['apply'](test_cp)
        # DataLoader'lar
        train_loader = DataLoader(train_cp, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_cp, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_cp, batch_size=batch_size, shuffle=False)
        # Eğit ve değerlendir
        metrics = train_evaluate(var_key, train_cp, val_cp, test_cp,
                                 train_loader, val_loader, test_loader,
                                 epochs, device, seed)
        for m in ['mae_soh', 'r2_soh', 'mae_rul', 'r2_rul']:
            results[var_key][m].append(metrics[m])
        print(f"{var_info['name']} -> SOH MAE={metrics['mae_soh']:.4f}, R²={metrics['r2_soh']:.4f}")

# -------------------------------
# 7. Wilcoxon Testleri (SOH MAE üzerinden)
# -------------------------------
full_mae_soh = results['full_model']['mae_soh']
wilcoxon_results = {}
for var in ['no_cycle_graph', 'no_dqdv', 'no_gnn_mlp']:
    stat, p = wilcoxon(full_mae_soh, results[var]['mae_soh'], alternative='two-sided')
    wilcoxon_results[f'full_vs_{var}'] = {'statistic': stat, 'p_value': p}

# -------------------------------
# 8. JSON Çıktısı
# -------------------------------
output = {
    'full_model': {'mae_soh': results['full_model']['mae_soh'],
                   'r2_soh': results['full_model']['r2_soh'],
                   'mae_rul': results['full_model']['mae_rul'],
                   'r2_rul': results['full_model']['r2_rul']},
    'no_cycle_graph': {'mae_soh': results['no_cycle_graph']['mae_soh'],
                       'r2_soh': results['no_cycle_graph']['r2_soh'],
                       'mae_rul': results['no_cycle_graph']['mae_rul'],
                       'r2_rul': results['no_cycle_graph']['r2_rul']},
    'no_dqdv': {'mae_soh': results['no_dqdv']['mae_soh'],
                'r2_soh': results['no_dqdv']['r2_soh'],
                'mae_rul': results['no_dqdv']['mae_rul'],
                'r2_rul': results['no_dqdv']['r2_rul']},
    'no_gnn_mlp': {'mae_soh': results['no_gnn_mlp']['mae_soh'],
                   'r2_soh': results['no_gnn_mlp']['r2_soh'],
                   'mae_rul': results['no_gnn_mlp']['mae_rul'],
                   'r2_rul': results['no_gnn_mlp']['r2_rul']},
    'wilcoxon': wilcoxon_results
}

output_path = '/home/msoylu/alper/analysis/results/ablation_10seeds_wilcoxon.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "="*60)
print("Ablasyon 10 Tohum Sonuçları (Ortalama ± Std)")
print("="*60)
for var in results:
    mae_soh = np.array(results[var]['mae_soh'])
    print(f"{var:15} SOH MAE: {mae_soh.mean():.4f} ± {mae_soh.std():.4f}")

print("\nWilcoxon signed-rank test p-values (Full vs others):")
for k, v in wilcoxon_results.items():
    print(f"{k}: p = {v['p_value']:.6f}")

print(f"\nSonuçlar kaydedildi: {output_path}")