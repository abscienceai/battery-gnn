import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
import os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Veriyi yükle
# -------------------------------
df = pd.read_csv('/home/msoylu/alper/analysis/battery_data_all_complete.csv')
print(f"Toplam satır: {len(df)}")

# Her hücre için max döngü sayısını bul
cell_max_cycle = df.groupby('cell_id')['cycle_number'].max().to_dict()
df['RUL'] = df.apply(lambda row: cell_max_cycle[row['cell_id']] - row['cycle_number'], axis=1)

# Her hücre için ilk döngünün peak voltajını bul (SEI proxy için) – sadece RWTH için anlamlı
df['SEI_proxy'] = 0.0  # varsayılan, sadece RWTH için doldurulacak
rwth_cells = df[df['source'] == 'RWTH']['cell_id'].unique()
for cell in rwth_cells:
    cell_df = df[df['cell_id'] == cell].sort_values('cycle_number')
    first_peak = cell_df['dqdv_peak_voltage_1'].iloc[0] if not cell_df.empty else 0
    mask = df['cell_id'] == cell
    df.loc[mask, 'SEI_proxy'] = df.loc[mask, 'dqdv_peak_voltage_1'] - first_peak
df['SEI_proxy'] = df['SEI_proxy'].clip(lower=0)

# Node özellikleri
node_features_cols = [
    'discharge_capacity_Ah', 'temperature_C', 'current_A',
    'dqdv_peak_voltage_1', 'dqdv_peak_height_1',
    'dqdv_peak_voltage_2', 'dqdv_peak_height_2',
    'nominal_capacity_Ah', 'max_voltage_V', 'min_voltage_V'
]
target_cols = ['discharge_capacity_Ah', 'RUL', 'SEI_proxy']
domain_map = {'MATR': 0, 'CALCE': 1, 'HUST': 2, 'RWTH': 3}
df['domain'] = df['source'].map(domain_map)

# Eksik değerleri doldur
df[node_features_cols] = df[node_features_cols].fillna(0)

# Normalizasyon scaler'ı
scaler = StandardScaler()
all_feats = df[node_features_cols].values
scaler.fit(all_feats)

graphs = []
for cell_id, group in tqdm(df.groupby('cell_id'), desc="Graf oluşturma"):
    group = group.sort_values('cycle_number')
    node_feats = scaler.transform(group[node_features_cols].values)
    x = torch.tensor(node_feats, dtype=torch.float)
    
    y_soh = torch.tensor(group['discharge_capacity_Ah'].values, dtype=torch.float)
    y_rul = torch.tensor(group['RUL'].values, dtype=torch.float)
    y_sei = torch.tensor(group['SEI_proxy'].values, dtype=torch.float)
    y = torch.stack([y_soh, y_rul, y_sei], dim=1)
    
    domain = torch.tensor([group['domain'].iloc[0]], dtype=torch.long)
    
    # Kenar oluştur (ardışık bağlantı)
    num_nodes = len(group)
    edge_index = []
    for i in range(num_nodes - 1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])
    if not edge_index:
        edge_index = [[], []]
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    data = Data(x=x, edge_index=edge_index, y=y, domain=domain, cell_id=cell_id)
    graphs.append(data)

print(f"Toplam graf sayısı: {len(graphs)}")

# Train/val/test ayır (hücre bazında)
cell_ids = [g.cell_id for g in graphs]
train_ids, temp_ids = train_test_split(cell_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

train_graphs = [g for g in graphs if g.cell_id in train_ids]
val_graphs = [g for g in graphs if g.cell_id in val_ids]
test_graphs = [g for g in graphs if g.cell_id in test_ids]

print(f"Train hücre: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

# Kaydet
output_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'train_graphs.pkl'), 'wb') as f:
    pickle.dump(train_graphs, f)
with open(os.path.join(output_dir, 'val_graphs.pkl'), 'wb') as f:
    pickle.dump(val_graphs, f)
with open(os.path.join(output_dir, 'test_graphs.pkl'), 'wb') as f:
    pickle.dump(test_graphs, f)
with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Graflar kaydedildi: {output_dir}")