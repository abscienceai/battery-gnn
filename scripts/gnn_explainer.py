import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import MaskType, ModelConfig, ModelMode
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

class NodeLevelMultiTaskGNN(torch.nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.soh_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        self.rul_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        self.sei_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        soh = self.soh_head(x).squeeze()
        rul = self.rul_head(x).squeeze()
        sei = self.sei_head(x).squeeze()
        return soh, rul, sei

class SOHOnlyWrapper(torch.nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.full_model = full_model
    def forward(self, x, edge_index):
        soh, _, _ = self.full_model(x, edge_index)
        return soh

data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'test_graphs.pkl'), 'rb') as f:
    test_graphs = pickle.load(f)

test_cell = None
for g in test_graphs:
    if g.domain.item() == 3:  # RWTH
        test_cell = g
        break
if test_cell is None:
    raise ValueError("RWTH test hücresi bulunamadı.")

node_in_dim = test_cell.x.shape[1]
full_model = NodeLevelMultiTaskGNN(node_in_dim)
full_model.load_state_dict(torch.load('/home/msoylu/alper/analysis/results/nodelevel_best.pth', map_location='cpu'), strict=False)
full_model.eval()

model = SOHOnlyWrapper(full_model)
model.eval()

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type=MaskType.attributes,
    edge_mask_type=MaskType.object,
    model_config=ModelConfig(mode=ModelMode.regression, task_level='node'),
)

explanation = explainer(test_cell.x, test_cell.edge_index)
node_importance = explanation.node_mask.mean(dim=1).detach().numpy()
edge_importance = explanation.edge_mask.detach().numpy()

topk_idx = np.argsort(node_importance)[-5:][::-1]
print("En önemli döngü indeksleri (SOH için):", topk_idx)
print("İlgili SOH değerleri:", [test_cell.y[idx, 0].item() for idx in topk_idx])

os.makedirs('/home/msoylu/alper/analysis/figures', exist_ok=True)
num_nodes = test_cell.num_nodes
cycles = list(range(1, num_nodes+1))

plt.figure(figsize=(10, 4))
plt.bar(cycles, node_importance, width=0.8, color='steelblue')
plt.xlabel('Cycle Number')
plt.ylabel('Importance Score')
plt.title('GNNExplainer: Cycle Importance for SOH Prediction (RWTH Cell)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/home/msoylu/alper/analysis/figures/gnn_explainer_importance.png', dpi=300)
print("Düğüm önem grafiği kaydedildi: /home/msoylu/alper/analysis/figures/gnn_explainer_importance.png")

plt.figure(figsize=(10, 4))
plt.plot(range(len(edge_importance)), edge_importance, marker='o', linestyle='-', color='coral')
plt.xlabel('Edge Index (consecutive cycles)')
plt.ylabel('Edge Importance Score')
plt.title('GNNExplainer: Edge Importance (consecutive cycles)')
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/msoylu/alper/analysis/figures/gnn_explainer_edge_importance.png', dpi=300)
print("Kenar önem grafiği kaydedildi: /home/msoylu/alper/analysis/figures/gnn_explainer_edge_importance.png")