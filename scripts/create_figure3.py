import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Sol: Fizik bilgili zamansal graf
ax1.set_title('Proposed: Physics-Informed Temporal Graph', fontsize=12)
nodes = [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8), (0.5, 0.5), (0.8, 0.5)]
radius = 0.08
for i, (x, y) in enumerate(nodes):
    circle = plt.Circle((x, y), radius, color='lightblue', ec='black', lw=1.5)
    ax1.add_patch(circle)
    ax1.text(x, y, f'Cycle {i+1}', ha='center', va='center', fontsize=9)

edges = [(0,1), (1,2), (2,4), (1,3), (3,4)]
weights = ['w=0.95', 'w=0.87', 'w=0.62', 'w=0.78', 'w=0.53']

for (i,j), w in zip(edges, weights):
    x1, y1 = nodes[i]
    x2, y2 = nodes[j]
    # Vektör hesapla
    dx = x2 - x1
    dy = y2 - y1
    dist = np.hypot(dx, dy)
    if dist == 0:
        continue
    # Ok başlangıcını dairenin kenarına al (başlangıç noktası + yarıçap kadar vektör)
    start_x = x1 + (radius / dist) * dx
    start_y = y1 + (radius / dist) * dy
    # Ok bitişini dairenin kenarına al (bitiş noktası - yarıçap)
    end_x = x2 - (radius / dist) * dx
    end_y = y2 - (radius / dist) * dy
    ax1.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, shrinkA=0, shrinkB=0))
    # Etiketi ortaya koy
    mx, my = (start_x + end_x)/2, (start_y + end_y)/2
    ax1.text(mx, my+0.03, w, fontsize=8, ha='center')

ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
ax1.axis('off')

# Sağ: Geleneksel zincir graf
ax2.set_title('Conventional: Sequential Chain Graph', fontsize=12)
for i, (x, y) in enumerate(nodes):
    circle = plt.Circle((x, y), radius, color='lightgray', ec='black', lw=1.5)
    ax2.add_patch(circle)
    ax2.text(x, y, f'Cycle {i+1}', ha='center', va='center', fontsize=9)

for (i,j) in edges:
    x1, y1 = nodes[i]
    x2, y2 = nodes[j]
    dx = x2 - x1
    dy = y2 - y1
    dist = np.hypot(dx, dy)
    if dist == 0:
        continue
    start_x = x1 + (radius / dist) * dx
    start_y = y1 + (radius / dist) * dy
    end_x = x2 - (radius / dist) * dx
    end_y = y2 - (radius / dist) * dy
    ax2.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, shrinkA=0, shrinkB=0))
    mx, my = (start_x + end_x)/2, (start_y + end_y)/2
    ax2.text(mx, my+0.03, 'w=1', fontsize=8, ha='center')

ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax2.axis('off')

plt.tight_layout()
plt.savefig('/home/msoylu/alper/analysis/results/figures/graph_construction_paradigm.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure saved: /home/msoylu/alper/analysis/results/figures/graph_construction_paradigm.png")