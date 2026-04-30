import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import pickle
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Model Tanımları (SOH + RUL)
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

class MLP(nn.Module):
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
# 2. Veriyi Yükleme ve Ön İşleme
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    all_train = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    all_test = pickle.load(f)

print(f"Train graf sayısı: {len(all_train)}, Test graf sayısı: {len(all_test)}")

# -------------------------------
# 3. Erken Kırpma Fonksiyonu
# -------------------------------
def truncate_graph(graph, keep_ratio):
    """Grafın sadece ilk keep_ratio oranındaki düğümlerini tutar, kenarları günceller."""
    num_nodes = graph.x.shape[0]
    keep_nodes = int(np.ceil(num_nodes * keep_ratio))
    if keep_nodes < 1:
        keep_nodes = 1
    # Düğüm özelliklerini ve hedef değişkenleri kırp
    x_trunc = graph.x[:keep_nodes]
    y_trunc = graph.y[:keep_nodes]  # SOH (orijinal kapasite)
    # Kenarları filtrele: sadece her iki ucu da keep_nodes içinde olanları tut
    edge_index = graph.edge_index
    mask = (edge_index[0] < keep_nodes) & (edge_index[1] < keep_nodes)
    edge_index_trunc = edge_index[:, mask]
    # Yeni graf oluştur
    new_graph = Data(x=x_trunc, y=y_trunc, edge_index=edge_index_trunc)
    # Domain bilgisini kopyala
    new_graph.domain = graph.domain
    return new_graph, num_nodes  # num_nodes orijinal toplam döngü sayısı

def add_rul_target(graph, total_cycles):
    """Her düğüm için RUL hedefini (kalan döngü sayısı) ekler."""
    num_nodes = graph.x.shape[0]
    rul = torch.zeros(num_nodes)
    for i in range(num_nodes):
        rul[i] = total_cycles - i - 1  # i 0-index, toplam döngü sayısı - (i+1)
    graph.rul = rul  # ek özellik olarak sakla
    return graph

# -------------------------------
# 4. Normalizasyon
# -------------------------------
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

# -------------------------------
# 5. Eğitim ve Değerlendirme
# -------------------------------
def train_model(model, train_loader, test_loader, epochs=50, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    soh_criterion = nn.MSELoss()
    rul_criterion = nn.MSELoss()
    best_mae_soh = float('inf')
    best_state = None
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            if isinstance(model, MLP):
                pred_soh, pred_rul = model(data.x)
            else:
                pred_soh, pred_rul = model(data.x, data.edge_index)
            loss_soh = soh_criterion(pred_soh, data.y[:, 0].view(-1))
            loss_rul = rul_criterion(pred_rul, data.rul.view(-1))
            loss = loss_soh + 0.01 * loss_rul  # RUL kaybını düşük ağırlıklandır
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Değerlendirme
        model.eval()
        pred_soh_all, true_soh_all = [], []
        pred_rul_all, true_rul_all = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                if isinstance(model, MLP):
                    pred_soh, pred_rul = model(data.x)
                else:
                    pred_soh, pred_rul = model(data.x, data.edge_index)
                pred_soh_all.append(pred_soh.cpu().numpy())
                true_soh_all.append(data.y[:, 0].cpu().numpy())
                pred_rul_all.append(pred_rul.cpu().numpy())
                true_rul_all.append(data.rul.cpu().numpy())
        
        pred_soh = np.concatenate(pred_soh_all)
        true_soh = np.concatenate(true_soh_all)
        pred_rul = np.concatenate(pred_rul_all)
        true_rul = np.concatenate(true_rul_all)
        
        mae_soh = mean_absolute_error(true_soh, pred_soh)
        r2_soh = r2_score(true_soh, pred_soh)
        mae_rul = mean_absolute_error(true_rul, pred_rul)
        r2_rul = r2_score(true_rul, pred_rul)
        
        if mae_soh < best_mae_soh:
            best_mae_soh = mae_soh
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f} | SOH MAE={mae_soh:.4f}, R²={r2_soh:.4f} | RUL MAE={mae_rul:.4f}, R²={r2_rul:.4f}")
    
    model.load_state_dict(best_state)
    # Final değerlendirme
    pred_soh_all, true_soh_all = [], []
    pred_rul_all, true_rul_all = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            if isinstance(model, MLP):
                pred_soh, pred_rul = model(data.x)
            else:
                pred_soh, pred_rul = model(data.x, data.edge_index)
            pred_soh_all.append(pred_soh.cpu().numpy())
            true_soh_all.append(data.y[:, 0].cpu().numpy())
            pred_rul_all.append(pred_rul.cpu().numpy())
            true_rul_all.append(data.rul.cpu().numpy())
    pred_soh = np.concatenate(pred_soh_all)
    true_soh = np.concatenate(true_soh_all)
    pred_rul = np.concatenate(pred_rul_all)
    true_rul = np.concatenate(true_rul_all)
    final_mae_soh = mean_absolute_error(true_soh, pred_soh)
    final_r2_soh = r2_score(true_soh, pred_soh)
    final_mae_rul = mean_absolute_error(true_rul, pred_rul)
    final_r2_rul = r2_score(true_rul, pred_rul)
    return final_mae_soh, final_r2_soh, final_mae_rul, final_r2_rul

# -------------------------------
# 6. Ana Döngü: Farklı Erken Oranları
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50
early_ratios = [0.1, 0.2]  # %10 ve %20
results = {'MLP': {}, 'GCN': {}}

for ratio in early_ratios:
    print(f"\n{'='*60}")
    print(f"Erken döngü oranı = {int(ratio*100)}%")
    
    # Eğitim graflarını kırp ve RUL hedeflerini ekle
    train_graphs_trunc = []
    for g in all_train:
        g_trunc, total_cycles = truncate_graph(g, ratio)
        g_trunc = add_rul_target(g_trunc, total_cycles)
        train_graphs_trunc.append(g_trunc)
    
    # Test graflarını kırp ve RUL hedeflerini ekle
    test_graphs_trunc = []
    for g in all_test:
        g_trunc, total_cycles = truncate_graph(g, ratio)
        g_trunc = add_rul_target(g_trunc, total_cycles)
        test_graphs_trunc.append(g_trunc)
    
    # Normalizasyon
    scaler = normalize_features(train_graphs_trunc, test_graphs_trunc)
    
    train_loader = DataLoader(train_graphs_trunc, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs_trunc, batch_size=batch_size, shuffle=False)
    
    sample = train_graphs_trunc[0]
    input_dim = sample.x.shape[1]
    
    # MLP
    print("MLP eğitiliyor...")
    mlp = MLP(input_dim)
    mae_soh_mlp, r2_soh_mlp, mae_rul_mlp, r2_rul_mlp = train_model(mlp, train_loader, test_loader, epochs, device)
    
    # GCN
    print("GCN eğitiliyor...")
    gcn = NodeLevelGCN(input_dim)
    mae_soh_gcn, r2_soh_gcn, mae_rul_gcn, r2_rul_gcn = train_model(gcn, train_loader, test_loader, epochs, device)
    
    results['MLP'][ratio] = (mae_soh_mlp, r2_soh_mlp, mae_rul_mlp, r2_rul_mlp)
    results['GCN'][ratio] = (mae_soh_gcn, r2_soh_gcn, mae_rul_gcn, r2_rul_gcn)
    
    print(f"Ratio {int(ratio*100)}%: MLP -> SOH MAE={mae_soh_mlp:.4f}, R²={r2_soh_mlp:.4f} | RUL MAE={mae_rul_mlp:.4f}, R²={r2_rul_mlp:.4f}")
    print(f"          GCN -> SOH MAE={mae_soh_gcn:.4f}, R²={r2_soh_gcn:.4f} | RUL MAE={mae_rul_gcn:.4f}, R²={r2_rul_gcn:.4f}")

# -------------------------------
# 7. Sonuçları Tablo Halinde Yazdır
# -------------------------------
print("\n" + "="*80)
print("Erken Tahmin Testi Sonuçları (SOH ve RUL için MAE / R²)")
print("="*80)
print(f"{'Oran':<6} {'Model':<6} {'SOH MAE':<10} {'SOH R²':<10} {'RUL MAE':<10} {'RUL R²':<10}")
print("-"*80)
for ratio in early_ratios:
    mae_s_mlp, r2_s_mlp, mae_r_mlp, r2_r_mlp = results['MLP'][ratio]
    mae_s_gcn, r2_s_gcn, mae_r_gcn, r2_r_gcn = results['GCN'][ratio]
    print(f"{int(ratio*100):<6} {'MLP':<6} {mae_s_mlp:<10.4f} {r2_s_mlp:<10.4f} {mae_r_mlp:<10.4f} {r2_r_mlp:<10.4f}")
    print(f"{int(ratio*100):<6} {'GCN':<6} {mae_s_gcn:<10.4f} {r2_s_gcn:<10.4f} {mae_r_gcn:<10.4f} {r2_r_gcn:<10.4f}")
    print("-"*80)

# CSV olarak kaydet
import pandas as pd
rows = []
for ratio in early_ratios:
    for model in ['MLP', 'GCN']:
        mae_s, r2_s, mae_r, r2_r = results[model][ratio]
        rows.append({'ratio': int(ratio*100), 'model': model, 'task': 'SOH', 'MAE': mae_s, 'R2': r2_s})
        rows.append({'ratio': int(ratio*100), 'model': model, 'task': 'RUL', 'MAE': mae_r, 'R2': r2_r})
df = pd.DataFrame(rows)
df.to_csv('/home/msoylu/alper/analysis/results/early_prediction.csv', index=False)
print("\nSonuçlar '/home/msoylu/alper/analysis/results/early_prediction.csv' dosyasına kaydedildi.")