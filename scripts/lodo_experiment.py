import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import pickle
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -------------------------------
# 1. Model tanımı
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
        return self.soh_head(x).squeeze()  # (num_nodes,)

# -------------------------------
# 2. Veriyi yükle
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    all_train = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    all_test = pickle.load(f)

all_graphs = all_train + all_test
print(f"Toplam graf sayısı: {len(all_graphs)}")

domain_names = {0: 'MATR', 1: 'CALCE', 2: 'HUST', 3: 'RWTH'}
graphs_by_domain = {d: [] for d in domain_names.values()}
for g in all_graphs:
    if isinstance(g.domain, torch.Tensor):
        dom_id = g.domain.item()
    else:
        dom_id = int(g.domain)
    dom_name = domain_names[dom_id]
    graphs_by_domain[dom_name].append(g)

print("\n=== Domain bazlı hücre sayıları ===")
for dom, glist in graphs_by_domain.items():
    print(f"{dom}: {len(glist)}")

# HUST'u LODO'dan çıkar (ölçek uyumsuzluğu)
exclude_domains = ['HUST']
domains_to_use = [d for d in domain_names.values() if d not in exclude_domains]
print(f"\nLODO'ya dahil edilecek domainler: {domains_to_use}")

# -------------------------------
# 3. Normalizasyon fonksiyonları
# -------------------------------
def normalize_features(graphs, scaler=None, fit=False):
    all_feats = []
    for g in graphs:
        all_feats.append(g.x.numpy())
    all_feats = np.vstack(all_feats)
    if fit:
        scaler = StandardScaler()
        scaler.fit(all_feats)
    for g in graphs:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    return graphs, scaler

def normalize_targets(train_graphs, test_graphs):
    """Eğitim setindeki SOH değerlerini 0-1 arasına MinMaxScaler ile normalize et.
       Her grafın y'si (num_nodes, 1) şeklinde olmalı."""
    # Train setindeki tüm y değerlerini topla (2D)
    train_y_list = []
    for g in train_graphs:
        # g.y şekli (num_nodes,) veya (num_nodes,1) olabilir
        y_np = g.y.numpy()
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        train_y_list.append(y_np)
    train_y = np.vstack(train_y_list)
    scaler = MinMaxScaler()
    scaler.fit(train_y)
    # Normalize et ve tensöre çevir (şekli (num_nodes,1) koru)
    for g in train_graphs:
        y_np = g.y.numpy()
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        y_norm = scaler.transform(y_np)
        g.y = torch.tensor(y_norm, dtype=torch.float)  # (num_nodes, 1)
    for g in test_graphs:
        y_np = g.y.numpy()
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        y_norm = scaler.transform(y_np)
        g.y = torch.tensor(y_norm, dtype=torch.float)
    return scaler

# -------------------------------
# 4. LODO döngüsü
# -------------------------------
results = {}

for test_domain in domains_to_use:
    print(f"\n{'='*60}")
    print(f"=== LODO: Test set = {test_domain} ===")
    train_graphs = []
    for d, glist in graphs_by_domain.items():
        if d != test_domain and d not in exclude_domains:
            train_graphs.extend(glist)
    test_graphs = graphs_by_domain[test_domain]
    print(f"Train hücre: {len(train_graphs)}, Test hücre: {len(test_graphs)}")
    
    # Hedef değişken normalizasyonu (eğitim setine göre)
    normalize_targets(train_graphs, test_graphs)
    
    # Node feature normalizasyonu
    train_graphs, scaler = normalize_features(train_graphs, fit=True)
    test_graphs, _ = normalize_features(test_graphs, scaler)
    
    batch_size = 32
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample = train_graphs[0]
    model = NodeLevelGCN(sample.x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    def train_epoch(loader):
        model.train()
        total_loss = 0
        for data in tqdm(loader, desc=f'Training (test={test_domain})'):
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)          # (total_nodes,)
            target = data.y[:, 0].view(-1)                # (total_nodes,)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    def evaluate(loader, return_details=False):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                pred = model(data.x, data.edge_index)
                preds.append(pred.cpu().numpy())
                trues.append(data.y[:, 0].cpu().numpy().flatten())
        pred = np.concatenate(preds)
        true = np.concatenate(trues)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        if return_details:
            return mae, r2, true, pred
        return mae, r2
    
    epochs = 50
    best_mae = float('inf')
    for epoch in range(1, epochs+1):
        train_loss = train_epoch(train_loader)
        test_mae, test_r2 = evaluate(test_loader)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        if test_mae < best_mae:
            best_mae = test_mae
            torch.save(model.state_dict(), f'/home/msoylu/alper/analysis/results/lodo_{test_domain}.pth')
    
    model.load_state_dict(torch.load(f'/home/msoylu/alper/analysis/results/lodo_{test_domain}.pth'))
    final_mae, final_r2, true_vals, pred_vals = evaluate(test_loader, return_details=True)
    results[test_domain] = {'mae': final_mae, 'r2': final_r2}
    print(f"✅ {test_domain} için nihai MAE: {final_mae:.4f}, R²: {final_r2:.4f}")

print("\n=== LODO Sonuçları Özeti (yalnızca MATR, CALCE, RWTH) ===")
for dom, res in results.items():
    print(f"{dom}: MAE = {res['mae']:.4f}, R² = {res['r2']:.4f}")

maes = [res['mae'] for res in results.values()]
if maes:
    print(f"\nOrtalama LODO MAE = {np.mean(maes):.4f} ± {np.std(maes):.4f}")