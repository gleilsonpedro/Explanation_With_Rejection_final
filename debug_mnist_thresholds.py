"""
Script de debug para investigar o problema dos thresholds no MNIST
"""
import sys
sys.path.insert(0, '.')

from data.datasets import set_mnist_options, carregar_dataset
from peab_2 import treinar_e_avaliar_modelo, DATASET_CONFIG, DEFAULT_LOGREG_PARAMS

# Configurar MNIST: pooling, dígitos 3 vs 8 (mais ambíguos)
set_mnist_options('pool2x2', (3, 8))

# Carregar dataset
X, y, nomes_classes = carregar_dataset('mnist')
print(f"\n{'='*60}")
print(f"Dataset MNIST carregado: {X.shape[0]} instâncias, {X.shape[1]} features")
print(f"Classes: {nomes_classes}")
print(f"Distribuição: {dict(zip(*np.unique(y, return_counts=True)))}")

# Configuração do experimento
config = DATASET_CONFIG['mnist']
rejection_cost = config['rejection_cost']
test_size = config['test_size']
subsample_size = config.get('subsample_size', None)

print(f"\nParâmetros:")
print(f"  rejection_cost: {rejection_cost}")
print(f"  test_size: {test_size}")
print(f"  subsample_size: {subsample_size}")

# Aplicar subsampling se configurado
if subsample_size:
    import numpy as np
    from sklearn.model_selection import train_test_split
    X, _, y, _ = train_test_split(X, y, train_size=subsample_size, random_state=42, stratify=y)
    print(f"\nApós subsampling: {X.shape[0]} instâncias")

# Treinar e ver os thresholds
print(f"\n{'='*60}")
print("Iniciando treino e otimização de thresholds...")
print(f"{'='*60}")

import numpy as np
pipeline, t_plus, t_minus, model_params = treinar_e_avaliar_modelo(
    X=X, y=y, 
    test_size=test_size, 
    rejection_cost=rejection_cost, 
    logreg_params=DEFAULT_LOGREG_PARAMS
)

print(f"\n{'='*60}")
print("RESULTADO FINAL:")
print(f"  t_plus (limite superior): {t_plus:.4f}")
print(f"  t_minus (limite inferior): {t_minus:.4f}")
print(f"  Largura da zona de rejeição: {t_plus - t_minus:.4f}")
print(f"{'='*60}\n")
