import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import pickle
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 1. Model tanımı (SOH-only, aynı NodeLevelGCN)
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

# -------------------------------
# 2. Veriyi yükle
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    test_graphs = pickle.load(f)

# Domain etiketlerini oku (her grafın domain özelliği var)
domain_names = {0: 'MATR', 1: 'CALCE', 2: 'HUST', 3: 'RWTH'}
graphs_by_domain = {d: [] for d in domain_names.values()}
for g in test_graphs:
    if isinstance(g.domain, torch.Tensor):
        dom_id = g.domain.item()
    else:
        dom_id = int(g.domain)
    dom_name = domain_names[dom_id]
    graphs_by_domain[dom_name].append(g)

print("Domain bazlı test hücre sayıları:")
for dom, glist in graphs_by_domain.items():
    print(f"{dom}: {len(glist)}")

# -------------------------------
# 3. Normalizasyon (eğitim setinin scaler'ını yükle)
# -------------------------------
# Normalizasyon scaler'ı daha önce kaydedildiyse yükleyelim, yoksa eğitim setiyle yeniden fit etmeliyiz.
# Burada eğitim setini yükleyip scaler'ı oluşturuyoruz (test setini normalize etmek için).
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    train_graphs = pickle.load(f)

from sklearn.preprocessing import StandardScaler
all_feats = []
for g in train_graphs:
    all_feats.append(g.x.numpy())
all_feats = np.vstack(all_feats)
scaler = StandardScaler()
scaler.fit(all_feats)
for g in test_graphs:
    g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)

# -------------------------------
# 4. Modeli yükle
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sample = test_graphs[0]
model = NodeLevelGCN(sample.x.shape[1]).to(device)
model_path = '/home/msoylu/alper/analysis/results/best_model.pth'  # En iyi modelin yolu
if not os.path.exists(model_path):
    # Alternatif: lodo_experiment.py'de kaydedilen modeli kullan
    model_path = '/home/msoylu/alper/analysis/results/lodo_MATR.pth'  # örnek, uygun olanı seçin
    print(f"best_model.pth bulunamadı, {model_path} kullanılıyor.")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -------------------------------
# 5. Her domain için değerlendirme
# -------------------------------
batch_size = 32
results = {}
for dom, graphs in graphs_by_domain.items():
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index)
            preds.append(pred.cpu().numpy())
            trues.append(data.y[:, 0].cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    results[dom] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"{dom}: MAE={mae:.4f} Ah, RMSE={rmse:.4f} Ah, R²={r2:.4f}")

# -------------------------------
# 6. Tablo formatında yazdır
# -------------------------------
print("\nPer-dataset metrics for GCN (test set):")
print(f"{'Dataset':<8} {'MAE (Ah)':<12} {'RMSE (Ah)':<12} {'R²':<10}")
for dom, m in results.items():
    print(f"{dom:<8} {m['MAE']:<12.4f} {m['RMSE']:<12.4f} {m['R2']:<10.4f}")