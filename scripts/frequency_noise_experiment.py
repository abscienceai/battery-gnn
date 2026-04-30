import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import pickle
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import copy

# -------------------------------
# Modeller (SOH only)
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
# Gürültü üretme fonksiyonları
# -------------------------------
def add_white_noise(graphs, sigma):
    for g in graphs:
        noise = torch.randn_like(g.x) * sigma
        g.x = g.x + noise
    return graphs

def add_low_freq_noise(graphs, sigma, rho=0.9):
    """AR(1) süreci: n_t = rho * n_{t-1} + epsilon_t"""
    for g in graphs:
        num_nodes = g.x.shape[0]
        noise = np.zeros((num_nodes, g.x.shape[1]))
        eps = np.random.randn(num_nodes, g.x.shape[1]) * sigma
        noise[0] = eps[0]
        for i in range(1, num_nodes):
            noise[i] = rho * noise[i-1] + eps[i]
        g.x = g.x + torch.tensor(noise, dtype=torch.float)
    return graphs

# -------------------------------
# Veri yükleme ve normalizasyon
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    all_train = pickle.load(f)
with open(os.path.join(data_dir, 'val_graphs_no_target.pkl'), 'rb') as f:
    all_val = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    all_test = pickle.load(f)

def normalize_features(train_graphs, val_graphs, test_graphs):
    all_feats = np.vstack([g.x.numpy() for g in train_graphs])
    scaler = StandardScaler().fit(all_feats)
    for g in train_graphs: g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    for g in val_graphs:   g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    for g in test_graphs:  g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    return scaler

# Normalize et (temiz, gürültüsüz)
scaler = normalize_features(all_train, all_val, all_test)
original_train = copy.deepcopy(all_train)
original_val = copy.deepcopy(all_val)
original_test = copy.deepcopy(all_test)

# -------------------------------
# Eğitim fonksiyonu (tek seed, temiz eğitim)
# -------------------------------
def train_model(model, train_loader, val_loader, epochs=50, device='cpu', is_gcn=True):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val_mae = float('inf')
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            opt.zero_grad()
            if is_gcn:
                pred = model(data.x, data.edge_index)
            else:
                pred = model(data.x)
            loss = criterion(pred, data.y[:,0].view(-1))
            loss.backward()
            opt.step()
        # Validation
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

def evaluate(model, test_loader, device='cpu', is_gcn=True):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            if is_gcn:
                pred = model(data.x, data.edge_index)
            else:
                pred = model(data.x)
            preds.append(pred.cpu().numpy())
            trues.append(data.y[:,0].cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    return mean_absolute_error(true, pred)

# -------------------------------
# Ana döngü
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50
sigma = 0.05
noise_types = {
    'white': add_white_noise,
    'low_freq_correlated': lambda g, s: add_low_freq_noise(g, s, rho=0.9)
}
seed = 42
results = {}

# Temiz eğitim verilerini hazırla (gürültüsüz)
train_loader_clean = DataLoader(original_train, batch_size=batch_size, shuffle=True)
val_loader_clean = DataLoader(original_val, batch_size=batch_size, shuffle=False)
test_loader_clean = DataLoader(original_test, batch_size=batch_size, shuffle=False)

# MLP ve GCN modellerini temiz veriyle eğit
mlp = MLP_SOH(original_train[0].x.shape[1]).to(device)
mlp = train_model(mlp, train_loader_clean, val_loader_clean, epochs, device, is_gcn=False)
gcn = NodeLevelGCN(original_train[0].x.shape[1]).to(device)
gcn = train_model(gcn, train_loader_clean, val_loader_clean, epochs, device, is_gcn=True)

for noise_name, noise_func in noise_types.items():
    # Test setine gürültü ekle
    test_noisy = copy.deepcopy(original_test)
    test_noisy = noise_func(test_noisy, sigma)
    test_loader = DataLoader(test_noisy, batch_size=batch_size, shuffle=False)
    mae_mlp = evaluate(mlp, test_loader, device, is_gcn=False)
    mae_gcn = evaluate(gcn, test_loader, device, is_gcn=True)
    results[noise_name] = {'MLP': mae_mlp, 'GCN': mae_gcn}
    print(f"{noise_name}: MLP MAE={mae_mlp:.4f}, GCN MAE={mae_gcn:.4f}, gain={(mae_mlp-mae_gcn)/mae_mlp*100:.1f}%")

print("\nSummary:")
for k, v in results.items():
    print(f"{k:20} MLP={v['MLP']:.4f}, GCN={v['GCN']:.4f}, Δ={v['MLP']-v['GCN']:.4f}")