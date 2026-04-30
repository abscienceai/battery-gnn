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
# 1. DANN Bileşenleri
# -------------------------------
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha):
    return GradientReversalLayer.apply(x, alpha)

class GCNEncoder(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, num_domains=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_domains)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.fc(x).squeeze()

class DANN(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, num_layers=3, num_domains=4):
        super().__init__()
        self.encoder = GCNEncoder(node_in_dim, hidden_dim, num_layers)
        self.regressor = Regressor(hidden_dim)
        self.domain_classifier = DomainClassifier(hidden_dim, num_domains)
    def forward(self, x, edge_index, alpha=1.0):
        features = self.encoder(x, edge_index)
        soh = self.regressor(features)
        reversed_features = grad_reverse(features, alpha)
        domain_logits = self.domain_classifier(reversed_features)
        return soh, domain_logits

# -------------------------------
# 2. Veri Yükleme ve Normalizasyon
# -------------------------------
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    all_train = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    all_test = pickle.load(f)

print(f"Toplam eğitim graf: {len(all_train)}, test graf: {len(all_test)}")

domain_names = {0: 'MATR', 1: 'CALCE', 2: 'HUST', 3: 'RWTH'}
domain_to_id = {v: k for k, v in domain_names.items()}

# Tüm grafları domain bazında grupla (eğitim+test karışık)
graphs_by_domain = {d: [] for d in domain_names.values()}
for g in all_train + all_test:
    if isinstance(g.domain, torch.Tensor):
        dom_id = g.domain.item()
    else:
        dom_id = int(g.domain)
    dom_name = domain_names[dom_id]
    graphs_by_domain[dom_name].append(g)

print("Domain bazlı toplam hücre sayıları:")
for dom, glist in graphs_by_domain.items():
    print(f"{dom}: {len(glist)}")

# Eğitim ve test setlerini domain bazında ayır (hücre bazlı split zaten yapılmış)
train_by_domain = {d: [] for d in domain_names.values()}
test_by_domain = {d: [] for d in domain_names.values()}
for g in all_train:
    dom_id = g.domain.item() if isinstance(g.domain, torch.Tensor) else int(g.domain)
    dom_name = domain_names[dom_id]
    train_by_domain[dom_name].append(g)
for g in all_test:
    dom_id = g.domain.item() if isinstance(g.domain, torch.Tensor) else int(g.domain)
    dom_name = domain_names[dom_id]
    test_by_domain[dom_name].append(g)

print("\nEğitim seti domain dağılımı:")
for d in domain_names.values():
    print(f"{d}: {len(train_by_domain[d])} hücre")
print("Test seti domain dağılımı:")
for d in domain_names.values():
    print(f"{d}: {len(test_by_domain[d])} hücre")

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
# 3. DANN Eğitim ve Değerlendirme
# -------------------------------
def train_dann(source_graphs, target_train_graphs, target_test_graphs, epochs=50, batch_size=32, device='cpu'):
    # Eğitim seti = source + target_train
    train_graphs = source_graphs + target_train_graphs
    # Node feature normalizasyonu (eğitim setine göre)
    scaler = normalize_features(train_graphs, target_test_graphs)
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(target_test_graphs, batch_size=batch_size, shuffle=False)
    
    model = DANN(node_in_dim=train_graphs[0].x.shape[1], num_domains=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_reg = nn.MSELoss()
    criterion_domain = nn.CrossEntropyLoss()
    
    best_mae = float('inf')
    best_state = None
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            pred_soh, domain_logits = model(data.x, data.edge_index, alpha=1.0)
            loss_reg = criterion_reg(pred_soh, data.y[:, 0].view(-1))
            # Domain labels: her düğüm için aynı domain etiketi (data.domain)
            # data.batch ile her düğümün hangi grafa ait olduğunu bulup domain_labels'ı genişlet
            batch = data.batch  # (total_nodes,)
            domain_labels = data.domain.long()  # (batch_size,)
            domain_labels_expanded = domain_labels[batch]  # (total_nodes,)
            loss_domain = criterion_domain(domain_logits, domain_labels_expanded)
            loss = loss_reg + 0.1 * loss_domain
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Değerlendirme
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                pred_soh, _ = model(data.x, data.edge_index, alpha=1.0)
                preds.append(pred_soh.cpu().numpy())
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
            pred_soh, _ = model(data.x, data.edge_index, alpha=1.0)
            preds.append(pred_soh.cpu().numpy())
            trues.append(data.y[:, 0].cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    final_mae = mean_absolute_error(true, pred)
    final_r2 = r2_score(true, pred)
    return final_mae, final_r2

def source_only(source_graphs, target_test_graphs, device='cpu'):
    train_graphs = source_graphs
    test_graphs = target_test_graphs
    normalize_features(train_graphs, test_graphs)
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    model = DANN(node_in_dim=train_graphs[0].x.shape[1], num_domains=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_mae = float('inf')
    best_state = None
    for epoch in range(1, 51):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred_soh, _ = model(data.x, data.edge_index, alpha=0.0)
            loss = criterion(pred_soh, data.y[:, 0].view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                pred_soh, _ = model(data.x, data.edge_index, alpha=0.0)
                preds.append(pred_soh.cpu().numpy())
                trues.append(data.y[:, 0].cpu().numpy())
        pred = np.concatenate(preds)
        true = np.concatenate(trues)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: MAE={mae:.4f}, R²={r2:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred_soh, _ = model(data.x, data.edge_index, alpha=0.0)
            preds.append(pred_soh.cpu().numpy())
            trues.append(data.y[:, 0].cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    final_mae = mean_absolute_error(true, pred)
    final_r2 = r2_score(true, pred)
    return final_mae, final_r2

# -------------------------------
# 4. Ana Döngü
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = {}

domains = list(domain_names.values())

for target_domain in domains:
    print(f"\n{'='*60}")
    print(f"Hedef domain: {target_domain}")
    source_domains = [d for d in domains if d != target_domain]
    source_train_graphs = []
    for d in source_domains:
        source_train_graphs.extend(train_by_domain[d])
    target_train_graphs = train_by_domain[target_domain]
    target_test_graphs = test_by_domain[target_domain]
    print(f"Kaynak eğitim hücre sayısı: {len(source_train_graphs)}")
    print(f"Hedef eğitim hücre sayısı: {len(target_train_graphs)} (DANN için kullanılacak)")
    print(f"Hedef test hücre sayısı: {len(target_test_graphs)}")
    
    print("Source-only eğitiliyor...")
    mae_so, r2_so = source_only(source_train_graphs, target_test_graphs, device)
    print("DANN eğitiliyor...")
    mae_dann, r2_dann = train_dann(source_train_graphs, target_train_graphs, target_test_graphs, epochs=50, batch_size=32, device=device)
    
    results[target_domain] = {
        'source_only_MAE': mae_so,
        'source_only_R2': r2_so,
        'DANN_MAE': mae_dann,
        'DANN_R2': r2_dann
    }
    print(f"{target_domain}: Source-only MAE={mae_so:.4f}, R²={r2_so:.4f} | DANN MAE={mae_dann:.4f}, R²={r2_dann:.4f}")

# Tablo yazdır
print("\n" + "="*80)
print("Tablo: Domain Adaptation (DANN) Sonuçları")
print("="*80)
print(f"{'Hedef Seti':<10} {'Source-Only MAE':<15} {'Source-Only R²':<15} {'DANN MAE':<15} {'DANN R²':<15}")
for target, res in results.items():
    print(f"{target:<10} {res['source_only_MAE']:<15.4f} {res['source_only_R2']:<15.4f} {res['DANN_MAE']:<15.4f} {res['DANN_R2']:<15.4f}")

# CSV kaydet
import pandas as pd
df = pd.DataFrame([
    {'Hedef': k, 'KaynakOnly_MAE': v['source_only_MAE'], 'KaynakOnly_R2': v['source_only_R2'],
     'DANN_MAE': v['DANN_MAE'], 'DANN_R2': v['DANN_R2']}
    for k, v in results.items()
])
df.to_csv('/home/msoylu/alper/analysis/results/dann_results.csv', index=False)
print("\nSonuçlar /home/msoylu/alper/analysis/results/dann_results.csv kaydedildi.")