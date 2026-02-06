"""
Script de debug para verificar se os pixels vermelhos estão nos lugares corretos
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

# Carregar dados do JSON
data_peab = json.load(open('json/peab/mnist.json', 'r'))
config = data_peab['config']

# Recarregar dataset
from data.datasets import carregar_dataset, set_mnist_options

set_mnist_options(config['mnist_feature_mode'], tuple(config['mnist_digit_pair']))
X_full, y_full, class_names = carregar_dataset('mnist')
_, X_test, _, _ = train_test_split(
    X_full, y_full, 
    test_size=config['test_size'], 
    random_state=config['random_state']
)

# Pegar primeira instância
inst = data_peab['per_instance'][0]
inst_idx = 0
img_shape = (14, 14)
num_features = 196

# Obter imagem original
x_vals = X_test[inst_idx].values if hasattr(X_test, 'values') else X_test[inst_idx]
if x_vals.max() > 1.0:
    x_vals = x_vals / 255.0
img_original = x_vals.reshape(img_shape)

# Criar máscara da explicação
explanation = inst['explanation']
mask = np.zeros((14, 14), dtype=float)

print(f"Instância {inst_idx}:")
print(f"  Classe verdadeira: {inst['y_true']}")
print(f"  Classe predita: {inst['y_pred']}")
print(f"  Rejeitada: {inst['rejected']}")
print(f"  Total features na explicação: {len(explanation)}")
print(f"  Primeiras 10 features: {explanation[:10]}")

# Marcar pixels da explicação
coords_marcadas = []
for feat in explanation:
    if feat.startswith('bin_'):
        parts = feat.replace('bin_', '').split('_')
        row = int(parts[0])
        col = int(parts[1])
        mask[row, col] = 1.0
        coords_marcadas.append((row, col))

print(f"\nPrimeiras 10 coordenadas marcadas: {coords_marcadas[:10]}")
print(f"Coordenadas no canto superior esquerdo (0-2, 0-2):")
canto = [(r, c) for r, c in coords_marcadas if r < 3 and c < 3]
print(f"  {canto if canto else 'NENHUMA - canto está vazio (correto!)'}")

print(f"\nMáscara[0:3, 0:3] (canto superior esquerdo):")
print(mask[0:3, 0:3])

# Criar visualização
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Painel 1: Imagem original
axes[0].imshow(img_original, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Imagem Original', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Painel 2: Máscara da explicação
axes[1].imshow(mask, cmap='Reds', vmin=0, vmax=1)
axes[1].set_title(f'Máscara ({np.sum(mask)} pixels)', fontsize=12, fontweight='bold')
axes[1].axis('off')

# Painel 3: Overlay
img_rgb = np.stack([img_original, img_original, img_original], axis=-1)
for i in range(14):
    for j in range(14):
        if mask[i, j] > 0:
            alpha = 0.7
            img_rgb[i, j, 0] = min(1.0, img_rgb[i, j, 0] + alpha)
            img_rgb[i, j, 1] = img_rgb[i, j, 1] * (1 - alpha * 0.5)
            img_rgb[i, j, 2] = img_rgb[i, j, 2] * (1 - alpha * 0.5)

axes[2].imshow(img_rgb)
axes[2].set_title('Overlay Vermelho', fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('debug_visualizacao.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Imagem salva em: debug_visualizacao.png")
print("\nVERIFIQUE: Os pixels vermelhos deveriam estar concentrados no CENTRO, não no canto!")
