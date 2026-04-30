import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

colors = {
    'input': '#d9ead3',
    'gnn': '#cfe2f3',
    'head': '#fce5cd',
    'domain': '#f4cccc',
    'output': '#d5e8d4'
}

nodes = [
    ('Cycle Graph', 1, 6, 2, 1, colors['input']),
    ('GCN Layer 1', 4, 6, 2, 1, colors['gnn']),
    ('GCN Layer 2', 4, 4.5, 2, 1, colors['gnn']),
    ('GCN Layer 3', 4, 3, 2, 1, colors['gnn']),
    ('SOH Head', 7, 6, 2, 1, colors['head']),
    ('RUL Head', 7, 4.5, 2, 1, colors['head']),
    ('Domain Classifier (GRL)', 7, 3, 2, 1, colors['domain']),
    ('SOH Output', 9.5, 6, 1, 0.8, colors['output']),
    ('RUL Output', 9.5, 4.5, 1, 0.8, colors['output']),
]

for name, x, y, w, h, color in nodes:
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                  facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, name, ha='center', va='center', fontsize=10, fontweight='bold')

arrows = [
    ((2, 6.5), (4, 6.5)),
    ((5, 6), (5, 5.5), (5, 5.5), (5, 5)),
    ((5, 4), (5, 3.5), (5, 3.5), (5, 3)),
    ((6, 6.5), (7, 6.5)),
    ((6, 5), (7, 5)),
    ((6, 3.5), (7, 3.5)),
    ((8, 6.5), (9.5, 6.5)),
    ((8, 5), (9.5, 5)),
]

for arrow in arrows:
    if len(arrow) == 2:
        ax.annotate('', xy=arrow[1], xytext=arrow[0],
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    else:
        verts = [(arrow[0][0], arrow[0][1]),
                 (arrow[1][0], arrow[1][1]),
                 (arrow[2][0], arrow[2][1]),
                 (arrow[3][0], arrow[3][1])]
        path = patches.Path(verts, [patches.Path.MOVETO, patches.Path.LINETO, patches.Path.LINETO, patches.Path.LINETO])
        patch = patches.PathPatch(path, facecolor='none', edgecolor='gray', lw=1.5)
        ax.add_patch(patch)
        # manuel ok başı eklemek için basit bir yol
        ax.annotate('', xy=(arrow[3][0], arrow[3][1]), xytext=(arrow[2][0], arrow[2][1]),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

ax.text(0.5, 7.5, "Multi-Task GNN Architecture with Domain Adaptation",
        ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
os.makedirs('/home/msoylu/alper/analysis/figures', exist_ok=True)
plt.savefig('/home/msoylu/alper/analysis/figures/architecture.png', dpi=300, bbox_inches='tight')
print("Şekil 2 kaydedildi: /home/msoylu/alper/analysis/figures/architecture.png")