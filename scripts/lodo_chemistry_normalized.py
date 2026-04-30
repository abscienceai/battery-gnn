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

# -------------------------------
# 1. Model (NodeLevelGCN) — aynı
# -------------------------------
class NodeLevelGCN(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.soh_head = nn.Linear(hidden_dim, 1)
        # RUL başlığı da ekliyoruz, ama bu deneyde sadece SOH raporlanacak
        self.rul_head = nn.Linear(hidden_dim, 1)
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        soh = self.soh_head(x).squeeze()
        rul = self.rul_head(x).squeeze()
        return soh, rul

# -------------------------------
# 2. Veri Yükleme ve Kimya Özellikleri Ekleme
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    all_train = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    all_test = pickle.load(f)

all_graphs = all_train + all_test
print(f"Toplam graf sayısı: {len(all_graphs)}")

domain_names = {0: 'MATR', 1: 'CALCE', 2: 'HUST', 3: 'RWTH'}
domain_to_id = {v: k for k, v in domain_names.items()}

# Her grafa domain etiketi eklenmiş (g.domain) — mevcut.
# Kimya bilgisi ekleyelim: nominal_voltage ve one‑hot electrode_type
# Önce domain bazlı kimya tanımları
chemistry_info = {
    'MATR': {'nominal_voltage': 3.2, 'electrode_type': [1,0,0]},  # LFP
    'CALCE': {'nominal_voltage': 3.6, 'electrode_type': [0,1,0]}, # LCO
    'HUST': {'nominal_voltage': 3.2, 'electrode_type': [1,0,0]},  # LFP
    'RWTH': {'nominal_voltage': 3.7, 'electrode_type': [0,0,1]}   # NMC
}

def add_chemistry_features(graph):
    dom_id = graph.domain.item() if isinstance(graph.domain, torch.Tensor) else graph.domain
    dom_name = domain_names[dom_id]
    info = chemistry_info[dom_name]
    # nominal_voltage (1 float)
    voltage = torch.tensor([[info['nominal_voltage']]], dtype=torch.float)
    # one‑hot (3 float)
    one_hot = torch.tensor([info['electrode_type']], dtype=torch.float)
    # Mevcut x (9 özellik) ile birleştir: yeni x shape (num_nodes, 9+1+3=13)
    num_nodes = graph.x.shape[0]
    voltage_repeat = voltage.repeat(num_nodes, 1)
    one_hot_repeat = one_hot.repeat(num_nodes, 1)
    graph.x = torch.cat([graph.x, voltage_repeat, one_hot_repeat], dim=1)
    return graph

# Tüm graflara kimya özelliklerini ekleyelim
for g in all_graphs:
    add_chemistry_features(g)

# -------------------------------
# 3. LODO döngüsü (per‑dataset z‑score normalizasyonu, kimya özellikleri hazır)
# -------------------------------
def normalize_per_dataset(train_graphs, test_graphs):
    """
    train_graphs üzerinden mean/std hesapla, uygula.
    test_graphs'a aynı scaler'ı uygula.
    """
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

def train_and_evaluate(train_graphs, test_graphs, seed, device, epochs=50, batch_size=32):
    # Normalizasyon: eğitim setine göre
    normalize_per_dataset(train_graphs, test_graphs)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    sample = train_graphs[0]
    model = NodeLevelGCN(node_in_dim=sample.x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    soh_criterion = nn.MSELoss()
    rul_criterion = nn.MSELoss()
    
    best_val_mae = float('inf')
    best_state = None
    
    # Validation için train setinin %15'ini ayıralım (doğrulama seti yok, kullanmıyoruz aslında ama LODO'da doğrulama seti yok)
    # LODO'da doğrulama seti olmadığı için, eğitim setinin son 15'ini veya rastgele ayırmak gerek.
    # Ancak LODO bağlamında doğrulama yapmıyoruz, direkt 50 epoch sonunda kaydediyoruz. 
    # Fakat daha önceki LODO scriptlerinde eğitim setini train+val olarak kullanıp test seti ayrıydı.
    # Burada sadece train_graphs üzerinde eğitip test_graphs ile test edeceğiz, validation yok.
    # Kayıp minimize edilsin, en iyi model son epoch kaydedilsin veya validation MAE için train içinden ayırmamız gerek.
    # LODO'da genelde eğitim setinin tamamı kullanılır, ayrı bir doğrulama seti yok. 
    # O zaman en düşük eğitim kaybı veya son epoch modeli kullanılabilir.
    # Basitçe son epoch modeli alalım.
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f'Seed {seed} Epoch {epoch}', leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            pred_soh, pred_rul = model(data.x, data.edge_index)
            loss_soh = soh_criterion(pred_soh, data.y[:, 0].view(-1))
            loss_rul = rul_criterion(pred_rul, data.rul.view(-1))
            loss = loss_soh + 0.01 * loss_rul
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Seed {seed} Epoch {epoch} loss: {total_loss/len(train_loader):.4f}")
        # En iyi modeli validation seti olmadığından son epoch kaydediyoruz
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    # Test
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred_soh, _ = model(data.x, data.edge_index)
            preds.append(pred_soh.cpu().numpy())
            trues.append(data.y[:, 0].cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    return mae, r2

# -------------------------------
# 4. LODO ana döngüsü: her dataset test olacak
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seeds = [5782, 327, 984]
epochs = 50
batch_size = 32

# Önce tüm grafları domain gruplarına ayıralım (kimya özellikleri eklenmiş)
graphs_by_domain = {d: [] for d in domain_names.values()}
for g in all_graphs:
    dom_id = g.domain.item() if isinstance(g.domain, torch.Tensor) else int(g.domain)
    dom_name = domain_names[dom_id]
    graphs_by_domain[dom_name].append(g)

print("Domain bazlı hücre sayıları (kimya özellikleri eklendi):")
for dom, glist in graphs_by_domain.items():
    print(f"{dom}: {len(glist)}")

results = {}

for test_domain in domain_names.values():
    print(f"\n{'='*60}")
    print(f"Test seti: {test_domain}")
    train_graphs_all = []
    for d, glist in graphs_by_domain.items():
        if d != test_domain:
            train_graphs_all.extend(glist)
    test_graphs = graphs_by_domain[test_domain]
    print(f"Eğitim hücre sayısı: {len(train_graphs_all)}, Test hücre sayısı: {len(test_graphs)}")
    
    domain_results = {'seeds': []}
    mae_list = []
    r2_list = []
    for seed in seeds:
        print(f"\n--- Tohum: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Her seed için train_graphs ve test_graphs kopyalarını al
        train_cp = copy.deepcopy(train_graphs_all)
        test_cp = copy.deepcopy(test_graphs)
        # RUL hedefini ekle (her graf için)
        for g in train_cp + test_cp:
            num_nodes = g.x.shape[0]
            rul = torch.zeros(num_nodes)
            for i in range(num_nodes):
                rul[i] = num_nodes - i - 1
            g.rul = rul
        mae, r2 = train_and_evaluate(train_cp, test_cp, seed, device, epochs, batch_size)
        mae_list.append(mae)
        r2_list.append(r2)
        domain_results['seeds'].append({'seed': seed, 'mae': mae, 'r2': r2})
        print(f"  MAE = {mae:.4f} Ah, R² = {r2:.4f}")
    domain_results['mean_mae'] = np.mean(mae_list)
    domain_results['std_mae'] = np.std(mae_list)
    domain_results['mean_r2'] = np.mean(r2_list)
    domain_results['std_r2'] = np.std(r2_list)
    results[test_domain] = domain_results

# -------------------------------
# 5. JSON çıktısı
# -------------------------------
output = {
    "MATR_as_test": results.get("MATR", {}),
    "CALCE_as_test": results.get("CALCE", {}),
    "HUST_as_test": results.get("HUST", {}),
    "RWTH_as_test": results.get("RWTH", {}),
    "normalization": "per_dataset_zscore",
    "chemistry_features_added": True
}

output_path = '/home/msoylu/alper/analysis/results/lodo_chemistry_normalized.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "="*60)
print("LODO Kimya-Duyarlı Normalizasyon Sonuçları (3 seed)")
print("="*60)
for test_domain in domain_names.values():
    res = results.get(test_domain, {})
    print(f"{test_domain}: MAE = {res.get('mean_mae', 0):.4f} ± {res.get('std_mae', 0):.4f} Ah, R² = {res.get('mean_r2', 0):.4f} ± {res.get('std_r2', 0):.4f}")

print(f"\nSonuçlar kaydedildi: {output_path}")