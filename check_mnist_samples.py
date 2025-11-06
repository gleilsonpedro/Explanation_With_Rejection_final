"""Verificar quantas amostras sobram no MNIST após subsampling"""
import sys
sys.path.insert(0, '.')

from data.datasets import set_mnist_options, carregar_dataset
from sklearn.model_selection import train_test_split
import numpy as np

# Configurar MNIST
set_mnist_options('pool2x2', (3, 8))
X, y, nomes_classes = carregar_dataset('mnist')

print(f"Dataset original: {X.shape[0]} instâncias")
print(f"Distribuição: {dict(zip(*np.unique(y, return_counts=True)))}")

# Aplicar subsampling de 5%
subsample_size = 0.05
X_sub, _, y_sub, _ = train_test_split(X, y, train_size=subsample_size, random_state=42, stratify=y)
print(f"\nApós subsampling (5%): {X_sub.shape[0]} instâncias")
print(f"Distribuição: {dict(zip(*np.unique(y_sub, return_counts=True)))}")

# Split train/test (30% test)
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=test_size, random_state=42, stratify=y_sub)
print(f"\nApós split (30% test):")
print(f"  Train: {X_train.shape[0]} instâncias")
print(f"  Test: {X_test.shape[0]} instâncias")
print(f"  Train distribuição: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"  Test distribuição: {dict(zip(*np.unique(y_test, return_counts=True)))}")
