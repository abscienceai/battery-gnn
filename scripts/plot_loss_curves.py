import json
import matplotlib.pyplot as plt
import os

with open('/home/msoylu/alper/analysis/training_history.json', 'r') as f:
    history = json.load(f)

epochs = range(1, len(history['train_loss'])+1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', linewidth=1.5)
plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='s', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
os.makedirs('/home/msoylu/alper/analysis/figures', exist_ok=True)
plt.savefig('/home/msoylu/alper/analysis/figures/loss_curves.png', dpi=300)
print("Şekil 4 kaydedildi: /home/msoylu/alper/analysis/figures/loss_curves.png")