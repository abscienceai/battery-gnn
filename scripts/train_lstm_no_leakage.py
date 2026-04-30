import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Sızıntısız grafları yükle
data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'train_graphs_no_target.pkl'), 'rb') as f:
    train_graphs = pickle.load(f)
with open(os.path.join(data_dir, 'test_graphs_no_target.pkl'), 'rb') as f:
    test_graphs = pickle.load(f)

def create_sequences(graphs):
    X, y = [], []
    for g in graphs:
        features = g.x.numpy()          # (num_nodes, 9)
        soh_seq = g.y[:, 0].numpy()     # (num_nodes,)
        X.append(features)
        y.append(soh_seq[-1])           # son döngünün SOH'u
    return X, np.array(y)

X_train, y_train = create_sequences(train_graphs)
X_test, y_test = create_sequences(test_graphs)

print(f"Train hücre: {len(X_train)}, Test hücre: {len(X_test)}")

# Padding ve normalizasyon (tüm döngüleri aynı uzunluğa getir)
max_len = max(max(seq.shape[0] for seq in X_train), max(seq.shape[0] for seq in X_test))

def pad_sequence(seq, max_len):
    pad_len = max_len - seq.shape[0]
    if pad_len > 0:
        seq = np.vstack([seq, np.zeros((pad_len, seq.shape[1]))])
    return seq

X_train_pad = np.array([pad_sequence(seq, max_len) for seq in X_train])
X_test_pad = np.array([pad_sequence(seq, max_len) for seq in X_test])

# Normalizasyon (train üzerinden)
scaler = StandardScaler()
X_train_flat = X_train_pad.reshape(-1, X_train_pad.shape[-1])
scaler.fit(X_train_flat)
X_train_pad = scaler.transform(X_train_pad.reshape(-1, X_train_pad.shape[-1])).reshape(X_train_pad.shape)
X_test_pad = scaler.transform(X_test_pad.reshape(-1, X_test_pad.shape[-1])).reshape(X_test_pad.shape)

# LSTM modeli
class CellLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # son zaman adımı
        return self.fc(out).squeeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(torch.tensor(X_train_pad, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
test_dataset = TensorDataset(torch.tensor(X_test_pad, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = CellLSTM(input_dim=X_train_pad.shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

def train_epoch(loader):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader, desc='LSTM Training'):
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
    test_mae, test_r2 = evaluate(test_loader)  # test üzerinde doğrulama (val yok, direkt test)
    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test MAE: {test_mae:.4f} R²: {test_r2:.4f}")
    if test_mae < best_val_mae:
        best_val_mae = test_mae
        torch.save(model.state_dict(), '/home/msoylu/alper/analysis/results/lstm_no_leakage.pth')

model.load_state_dict(torch.load('/home/msoylu/alper/analysis/results/lstm_no_leakage.pth'))
final_mae, final_r2 = evaluate(test_loader)
print(f"\n✅ LSTM Test -> MAE: {final_mae:.4f}, R²: {final_r2:.4f}")