import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs.pkl'), 'rb') as f:
    train_graphs = pickle.load(f)
with open(os.path.join(data_dir, 'val_graphs.pkl'), 'rb') as f:
    val_graphs = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs.pkl'), 'rb') as f:
    test_graphs = pickle.load(f)

def create_sequences(graphs):
    X = []
    y_soh = []
    y_rul = []
    for g in graphs:
        features = g.x.numpy()
        soh_seq = g.y[:, 0].numpy()
        rul_seq = g.y[:, 1].numpy()
        X.append(features)
        y_soh.append(soh_seq[-1])
        y_rul.append(rul_seq[-1])
    return X, np.array(y_soh), np.array(y_rul)

X_train, y_soh_train, y_rul_train = create_sequences(train_graphs)
X_val, y_soh_val, y_rul_val = create_sequences(val_graphs)
X_test, y_soh_test, y_rul_test = create_sequences(test_graphs)

print(f"Train hücre: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

all_train_features = np.vstack(X_train)
scaler = StandardScaler()
scaler.fit(all_train_features)

def normalize_sequences(X_list):
    return [scaler.transform(seq) for seq in X_list]

X_train = normalize_sequences(X_train)
X_val = normalize_sequences(X_val)
X_test = normalize_sequences(X_test)

max_len = max(max(seq.shape[0] for seq in X_train),
              max(seq.shape[0] for seq in X_val),
              max(seq.shape[0] for seq in X_test))

def pad_sequence(seq, max_len):
    pad_len = max_len - seq.shape[0]
    if pad_len > 0:
        seq = np.vstack([seq, np.zeros((pad_len, seq.shape[1]))])
    return seq

X_train_pad = np.array([pad_sequence(seq, max_len) for seq in X_train])
X_val_pad = np.array([pad_sequence(seq, max_len) for seq in X_val])
X_test_pad = np.array([pad_sequence(seq, max_len) for seq in X_test])

class CellLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.soh_head = nn.Linear(hidden_dim, 1)
        self.rul_head = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        soh = self.soh_head(out).squeeze()
        rul = self.rul_head(out).squeeze()
        return soh, rul

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(torch.tensor(X_train_pad, dtype=torch.float),
                              torch.tensor(y_soh_train, dtype=torch.float),
                              torch.tensor(y_rul_train, dtype=torch.float))
val_dataset = TensorDataset(torch.tensor(X_val_pad, dtype=torch.float),
                            torch.tensor(y_soh_val, dtype=torch.float),
                            torch.tensor(y_rul_val, dtype=torch.float))
test_dataset = TensorDataset(torch.tensor(X_test_pad, dtype=torch.float),
                             torch.tensor(y_soh_test, dtype=torch.float),
                             torch.tensor(y_rul_test, dtype=torch.float))

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = CellLSTM(input_dim=X_train_pad.shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
w_soh, w_rul = 1.0, 0.01

def train_epoch(loader):
    model.train()
    total_loss = 0
    for x, y_soh, y_rul in tqdm(loader, desc='Training'):
        x, y_soh, y_rul = x.to(device), y_soh.to(device), y_rul.to(device)
        optimizer.zero_grad()
        soh_pred, rul_pred = model(x)
        loss = w_soh * criterion(soh_pred, y_soh) + w_rul * criterion(rul_pred, y_rul)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader):
    model.eval()
    soh_preds, soh_trues = [], []
    with torch.no_grad():
        for x, y_soh, _ in loader:
            x = x.to(device)
            soh_pred, _ = model(x)
            soh_preds.append(soh_pred.cpu().numpy())
            soh_trues.append(y_soh.numpy())
    soh_pred = np.concatenate(soh_preds)
    soh_true = np.concatenate(soh_trues)
    return {
        'soh_mae': mean_absolute_error(soh_true, soh_pred),
        'soh_r2': r2_score(soh_true, soh_pred)
    }

epochs = 50
best_val_mae = float('inf')
for epoch in range(1, epochs+1):
    train_loss = train_epoch(train_loader)
    val_metrics = evaluate(val_loader)
    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val SOH MAE: {val_metrics['soh_mae']:.4f} R²: {val_metrics['soh_r2']:.4f}")
    if val_metrics['soh_mae'] < best_val_mae:
        best_val_mae = val_metrics['soh_mae']
        torch.save(model.state_dict(), '/home/msoylu/alper/analysis/results/lstm_cell_based.pth')

model.load_state_dict(torch.load('/home/msoylu/alper/analysis/results/lstm_cell_based.pth'))
test_metrics = evaluate(test_loader)
print("\n✅ LSTM (Cell-Based) Test Sonuçları:")
print(f"  SOH MAE: {test_metrics['soh_mae']:.4f}, R²: {test_metrics['soh_r2']:.4f}")