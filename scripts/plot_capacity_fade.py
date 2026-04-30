import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('/home/msoylu/alper/analysis/battery_data_all_complete.csv')

sources = ['MATR', 'CALCE', 'HUST', 'RWTH']
colors = {'MATR': 'blue', 'CALCE': 'orange', 'HUST': 'green', 'RWTH': 'red'}
selected_cells = {}

for src in sources:
    cells = df[df['source'] == src]['cell_id'].unique()
    if len(cells) > 0:
        selected_cells[src] = cells[0]

plt.figure(figsize=(10, 6))
for src, cell_id in selected_cells.items():
    subset = df[(df['source'] == src) & (df['cell_id'] == cell_id)].sort_values('cycle_number')
    plt.plot(subset['cycle_number'], subset['discharge_capacity_Ah'], 
             label=f"{src} ({cell_id})", color=colors[src], linewidth=1.5)

plt.xlabel('Cycle Number', fontsize=12)
plt.ylabel('Discharge Capacity (Ah)', fontsize=12)
plt.title('Capacity Fade Curves for Example Cells from Each Dataset', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

os.makedirs('/home/msoylu/alper/analysis/figures', exist_ok=True)
plt.savefig('/home/msoylu/alper/analysis/figures/capacity_fade_example.png', dpi=300, bbox_inches='tight')
print("Şekil 1 kaydedildi: /home/msoylu/alper/analysis/figures/capacity_fade_example.png")