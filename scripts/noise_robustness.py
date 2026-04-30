import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import pickle
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Model Tanımları
# -------------------------------
class NodeLevelGCN(nn.Module):
    """Önerilen GCN modeli (fizik bilgili graf ile kullanılacak)"""
    def __init__(self, node_in_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.soh_head = nn.Linear(hidden_dim, 1)
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return self.soh_head(x).squeeze()

class MLP(nn.Module):
    """Graf yapısını kullanmayan MLP (No GNN baseline)"""
    def __init__(self, node_in_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(node_in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # x: (num_nodes, node_in_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze()

# -------------------------------
# 2. Veriyi Yükleme ve Ön İşleme
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    all_train = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    all_test = pickle.load(f)

print(f"Train graf sayısı: {len(all_train)}, Test graf sayısı: {len(all_test)}")

# Node feature'larını normalize et (eğitim setine göre)
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

# Hedef değişkeni (SOH) normalize etme (isteğe bağlı, bu deneyde doğrudan Ah cinsinden karşılaştırma yapıyoruz)
# Not: Gürültü testinde mutlak MAE (Ah) anlamlıdır, bu nedenle hedef normalize edilmez.

scaler = normalize_features(all_train, all_test)

# -------------------------------
# 3. Gürültü Ekleme Fonksiyonu
# -------------------------------
def add_gaussian_noise(graphs, sigma):
    """Graf listesindeki her grafın x özelliklerine Gauss gürültüsü ekler (yerinde değişiklik)"""
    for g in graphs:
        noise = torch.randn_like(g.x) * sigma
        g.x = g.x + noise
    return graphs

# -------------------------------
# 4. Eğitim ve Değerlendirme Fonksiyonları
# -------------------------------
def train_model(model, train_loader, test_loader, epochs=50, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_mae = float('inf')
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            if isinstance(model, MLP):
                pred = model(data.x)
            else:  # GCN
                pred = model(data.x, data.edge_index)
            loss = criterion(pred, data.y[:, 0].view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Değerlendirme
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                if isinstance(model, MLP):
                    pred = model(data.x)
                else:
                    pred = model(data.x, data.edge_index)
                preds.append(pred.cpu().numpy())
                trues.append(data.y[:, 0].cpu().numpy())
        pred = np.concatenate(preds)
        true = np.concatenate(trues)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={total_loss/len(train_loader):.4f}, Test MAE={mae:.4f}, R²={r2:.4f}")
    model.load_state_dict(best_state)
    return best_mae, r2

# -------------------------------
# 5. Ana Döngü: Farklı Gürültü Seviyeleri
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50

noise_levels = [0.0, 0.01, 0.05, 0.10]
results = {'MLP': {}, 'GCN': {}}

# Orijinal verilerin kopyalarını sakla (gürültü eklemeden önce)
import copy
train_graphs_clean = copy.deepcopy(all_train)
test_graphs_clean = copy.deepcopy(all_test)

for sigma in noise_levels:
    print(f"\n{'='*60}")
    print(f"Gürültü seviyesi σ = {sigma}")
    # Verileri kopyala ve gürültü ekle
    train_graphs = copy.deepcopy(train_graphs_clean)
    test_graphs = copy.deepcopy(test_graphs_clean)
    if sigma > 0:
        train_graphs = add_gaussian_noise(train_graphs, sigma)
        test_graphs = add_gaussian_noise(test_graphs, sigma)
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    # MLP
    sample = train_graphs[0]
    mlp = MLP(node_in_dim=sample.x.shape[1])
    print("MLP eğitiliyor...")
    mae_mlp, r2_mlp = train_model(mlp, train_loader, test_loader, epochs, device)
    results['MLP'][sigma] = (mae_mlp, r2_mlp)
    
    # GCN
    gcn = NodeLevelGCN(node_in_dim=sample.x.shape[1])
    print("GCN eğitiliyor...")
    mae_gcn, r2_gcn = train_model(gcn, train_loader, test_loader, epochs, device)
    results['GCN'][sigma] = (mae_gcn, r2_gcn)
    
    print(f"σ={sigma}: MLP -> MAE={mae_mlp:.4f}, R²={r2_mlp:.4f}  |  GCN -> MAE={mae_gcn:.4f}, R²={r2_gcn:.4f}")

# -------------------------------
# 6. Sonuçları Tablo Halinde Yazdır
# -------------------------------
print("\n" + "="*60)
print("Gürültü Dayanıklılığı Testi Sonuçları (SOH MAE ve R²)")
print("="*60)
print(f"{'σ':<8} {'MLP MAE':<12} {'MLP R²':<12} {'GCN MAE':<12} {'GCN R²':<12}")
print("-"*60)
for sigma in noise_levels:
    mae_mlp, r2_mlp = results['MLP'][sigma]
    mae_gcn, r2_gcn = results['GCN'][sigma]
    print(f"{sigma:<8} {mae_mlp:<12.4f} {r2_mlp:<12.4f} {mae_gcn:<12.4f} {r2_gcn:<12.4f}")

# İsteğe bağlı: sonuçları CSV dosyasına kaydet
import pandas as pd
df = pd.DataFrame([
    {'sigma': s, 'model': 'MLP', 'MAE': results['MLP'][s][0], 'R2': results['MLP'][s][1]} for s in noise_levels
] + [
    {'sigma': s, 'model': 'GCN', 'MAE': results['GCN'][s][0], 'R2': results['GCN'][s][1]} for s in noise_levels
])
df.to_csv('/home/msoylu/alper/analysis/results/noise_robustness.csv', index=False)
print("\nSonuçlar '/home/msoylu/alper/analysis/results/noise_robustness.csv' dosyasına kaydedildi.")