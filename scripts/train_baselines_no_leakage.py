import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from tqdm import tqdm

# Yeni sızıntısız grafları yükle
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    train_graphs = pickle.load(f)
with open(os.path.join(data_dir, 'val_graphs_no_target.pkl'), 'rb') as f:
    val_graphs = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    test_graphs = pickle.load(f)

def extract_sequences(graphs):
    X = []
    y_soh = []
    for g in graphs:
        X.append(g.x.numpy())
        y_soh.append(g.y[:, 0].numpy())
    X = np.concatenate(X, axis=0)
    y_soh = np.concatenate(y_soh)
    return X, y_soh

X_train, y_train = extract_sequences(train_graphs)
X_val, y_val = extract_sequences(val_graphs)
X_test, y_test = extract_sequences(test_graphs)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print(f"Ridge -> MAE: {mean_absolute_error(y_test, y_pred):.4f}, R²: {r2_score(y_test, y_pred):.4f}")

# XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print(f"XGBoost -> MAE: {mean_absolute_error(y_test, y_pred):.4f}, R²: {r2_score(y_test, y_pred):.4f}")

# MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(input_dim=X_train.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def train_epoch(loader):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader, desc='MLP Training'):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x)
            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    return mean_absolute_error(true, pred), r2_score(true, pred)

epochs = 50
best_val_mae = float('inf')
for epoch in range(1, epochs+1):
    train_loss = train_epoch(train_loader)
    val_mae, val_r2 = evaluate(val_loader)
    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} R²: {val_r2:.4f}")
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), '/home/msoylu/alper/analysis/results/mlp_no_leakage.pth')
model.load_state_dict(torch.load('/home/msoylu/alper/analysis/results/mlp_no_leakage.pth'))
test_mae, test_r2 = evaluate(test_loader)
print(f"MLP -> MAE: {test_mae:.4f}, R²: {test_r2:.4f}")