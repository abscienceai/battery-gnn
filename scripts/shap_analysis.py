import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# -------------------------------
# 1. Model tanımı (sadece SOH çıktısı)
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
            x = torch.relu(conv(x, edge_index))
        return self.soh_head(x).squeeze()

# -------------------------------
# 2. Veriyi yükle (sızıntısız test grafları)
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    test_graphs = pickle.load(f)

# Tüm düğüm özelliklerini topla
all_node_features = []
for g in test_graphs:
    all_node_features.append(g.x.numpy())
all_node_features = np.vstack(all_node_features)
print(f"Toplam düğüm sayısı: {len(all_node_features)}")

# Rastgele 1000 düğüm seç (SHAP hesaplama hızı için)
np.random.seed(42)
sample_idx = np.random.choice(len(all_node_features), 1000, replace=False)
sample_x = all_node_features[sample_idx]

# -------------------------------
# 3. Modeli yükle
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sample_graph = test_graphs[0]
model = NodeLevelGCN(sample_graph.x.shape[1]).to(device)
model.load_state_dict(torch.load('/home/msoylu/alper/analysis/results/gcn_no_leakage.pth', map_location=device))
model.eval()

# -------------------------------
# 4. SHAP için tahmin fonksiyonu (1D çıktı)
# -------------------------------
def predict(x):
    x_tensor = torch.tensor(x, dtype=torch.float).to(device)
    edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(x_tensor, edge_index)
    # SHAP, 1D numpy dizisi bekler (skaler değil, vektör)
    return pred.cpu().numpy().reshape(-1)

# Arka plan örnekleri (100 tane)
background = sample_x[:100]

# SHAP açıklayıcı
explainer = shap.KernelExplainer(predict, background)

# SHAP değerlerini hesapla (200 örnek için)
shap_values = explainer.shap_values(sample_x[:200], nsamples=100)

# -------------------------------
# 5. Görselleştirme
# -------------------------------
feature_names = [
    'Temperature', 'Current',
    'dQdV_peak_voltage_1', 'dQdV_peak_height_1',
    'dQdV_peak_voltage_2', 'dQdV_peak_height_2',
    'Nominal Capacity', 'Max Voltage', 'Min Voltage'
]

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, sample_x[:200], feature_names=feature_names, show=False)
plt.tight_layout()
os.makedirs('/home/msoylu/alper/analysis/figures', exist_ok=True)
plt.savefig('/home/msoylu/alper/analysis/figures/shap_summary.png', dpi=300, bbox_inches='tight')
print("SHAP özet grafiği kaydedildi: /home/msoylu/alper/analysis/figures/shap_summary.png")

# Ortalama mutlak SHAP değerlerini yazdır
mean_shap = np.abs(shap_values).mean(axis=0)
for name, val in zip(feature_names, mean_shap):
    print(f"{name}: {val:.4f}")