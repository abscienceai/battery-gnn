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
from sklearn.preprocessing import StandardScaler
import copy

# -------------------------------
# 1. Model (SOH-only)
# -------------------------------
class NodeLevelGCN(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.soh_head = nn.Linear(hidden_dim, 1)
    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            if edge_weight is not None:
                x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
            else:
                x = F.relu(conv(x, edge_index))
        return self.soh_head(x).squeeze()

# -------------------------------
# 2. Veri Yükleme ve Normalizasyon
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    all_train = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    all_test = pickle.load(f)

print(f"Train graf: {len(all_train)}, Test graf: {len(all_test)}")

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

# Normalizasyonu sadece bir kere yap
scaler = normalize_features(all_train, all_test)

# -------------------------------
# 3. Kenar Ağırlığı Hesaplama Fonksiyonları
# -------------------------------
def compute_pearson_weight(graph):
    """Mevcut edge_weight'leri kullan (zaten Pearson korelasyonu ile hesaplanmış)"""
    if hasattr(graph, 'edge_weight') and graph.edge_weight is not None:
        return graph.edge_weight
    else:
        return torch.ones(graph.edge_index.shape[1], dtype=torch.float)

def compute_cosine_weight(graph):
    """Kosinüs benzerliği (ardışık döngülerin x vektörleri arasında)"""
    x = graph.x.numpy()
    edge_index = graph.edge_index.numpy()
    weights = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if dst == src + 1:
            a = x[src]
            b = x[dst]
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                sim = 0.0
            else:
                sim = dot / (norm_a * norm_b)
            weights.append(sim)
        else:
            weights.append(0.0)
    return torch.tensor(weights, dtype=torch.float)

def compute_euclidean_weight(graph):
    """Öklid mesafesi tabanlı benzerlik: w = 1/(1+dist)"""
    x = graph.x.numpy()
    edge_index = graph.edge_index.numpy()
    weights = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if dst == src + 1:
            dist = np.linalg.norm(x[src] - x[dst])
            sim = 1.0 / (1.0 + dist)
            weights.append(sim)
        else:
            weights.append(0.0)
    return torch.tensor(weights, dtype=torch.float)

def constant_weight(graph):
    return torch.ones(graph.edge_index.shape[1], dtype=torch.float)

# -------------------------------
# 4. Eğitim ve Değerlendirme
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
            if hasattr(data, 'edge_weight') and data.edge_weight is not None:
                pred = model(data.x, data.edge_index, data.edge_weight)
            else:
                pred = model(data.x, data.edge_index)
            loss = criterion(pred, data.y[:, 0].view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Validation on test set
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                if hasattr(data, 'edge_weight') and data.edge_weight is not None:
                    pred = model(data.x, data.edge_index, data.edge_weight)
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
            print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    model.load_state_dict(best_state)
    # Final evaluation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            if hasattr(data, 'edge_weight') and data.edge_weight is not None:
                pred = model(data.x, data.edge_index, data.edge_weight)
            else:
                pred = model(data.x, data.edge_index)
            preds.append(pred.cpu().numpy())
            trues.append(data.y[:, 0].cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    return mean_absolute_error(true, pred), r2_score(true, pred)

# -------------------------------
# 5. Ana Döngü: Farklı Kenar Ağırlıkları
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50

strategies = {
    'Pearson korelasyonu (mevcut)': compute_pearson_weight,
    'Kosinüs benzerliği': compute_cosine_weight,
    'Öklid mesafesi': compute_euclidean_weight,
    'Sabit ağırlık': constant_weight
}

results = {}
original_train = copy.deepcopy(all_train)
original_test = copy.deepcopy(all_test)

for name, weight_func in strategies.items():
    print(f"\n{'='*60}")
    print(f"Kenar ağırlığı: {name}")
    train_graphs = copy.deepcopy(original_train)
    test_graphs = copy.deepcopy(original_test)
    
    for g in train_graphs + test_graphs:
        g.edge_weight = weight_func(g)
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    model = NodeLevelGCN(node_in_dim=original_train[0].x.shape[1]).to(device)
    mae, r2 = train_model(model, train_loader, test_loader, epochs, device)
    results[name] = (mae, r2)
    print(f"Sonuç: MAE = {mae:.4f} Ah, R² = {r2:.4f}")

print("\n" + "="*60)
print("Kenar Ağırlığı Karşılaştırması (SOH MAE / R²)")
print("="*60)
for name, (mae, r2) in results.items():
    print(f"{name:<25} MAE = {mae:.4f} Ah, R² = {r2:.4f}")