import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.data import DataLoader
import os

data_dir = '/home/msoylu/alper/analysis/cycle_graphs/'
with open(os.path.join(data_dir, 'test_graphs.pkl'), 'rb') as f:
    test_graphs = pickle.load(f)

class SimpleGCN(torch.nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sample = test_graphs[0]
model = SimpleGCN(sample.x.shape[1]).to(device)
model.load_state_dict(torch.load('/home/msoylu/alper/analysis/results/nodelevel_best.pth', map_location=device), strict=False)
model.eval()

embeddings = []
domains = []
with torch.no_grad():
    for data in test_graphs:
        data = data.to(device)
        emb = model(data.x, data.edge_index)
        embeddings.append(emb.cpu().numpy())
        domains.extend([data.domain.item()] * emb.shape[0])
embeddings = np.concatenate(embeddings, axis=0)
domains = np.array(domains)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'orange']
labels = ['MATR', 'CALCE', 'HUST', 'RWTH']
for d in range(4):
    mask = domains == d
    plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=colors[d], label=labels[d], alpha=0.6, s=10)
plt.legend()
plt.title('t-SNE Visualization of Node Embeddings (Test Set)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
os.makedirs('/home/msoylu/alper/analysis/figures', exist_ok=True)
plt.savefig('/home/msoylu/alper/analysis/figures/tsne_domains.png', dpi=300)
print("t-SNE grafiği kaydedildi: /home/msoylu/alper/analysis/figures/tsne_domains.png")