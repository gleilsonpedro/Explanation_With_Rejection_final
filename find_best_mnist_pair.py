"""
Encontra o melhor par de dígitos MNIST para maximizar rejeições.
Testa todos os pares e mostra a separabilidade de cada um.
"""
import sys
sys.path.insert(0, '.')

from data.datasets import set_mnist_options, carregar_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

RANDOM_STATE = 42

def avaliar_separabilidade(digit_a, digit_b):
    """Treina LogReg e retorna métricas de separabilidade"""
    set_mnist_options('raw', (digit_a, digit_b))
    X, y, _ = carregar_dataset('mnist')
    
    # Subsample 5% para acelerar
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=0.05, random_state=RANDOM_STATE, stratify=y)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.3, random_state=RANDOM_STATE, stratify=y_sub
    )
    
    # Treinar modelo
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    
    # Métricas de separabilidade
    scores = pipeline.decision_function(X_train)
    acuracia = pipeline.score(X_test, y_test)
    
    # Quanto menor a distância entre classes, mais ambíguo
    scores_class0 = scores[y_train == 0]
    scores_class1 = scores[y_train == 1]
    
    mean_diff = abs(scores_class1.mean() - scores_class0.mean())
    overlap = len(scores[(scores > scores_class0.min()) & (scores < scores_class1.max())])
    std_combined = (scores_class0.std() + scores_class1.std()) / 2
    
    # Score de ambiguidade (menor = mais ambíguo = mais rejeições esperadas)
    ambiguity_score = mean_diff / (std_combined + 1e-6)
    
    return {
        'acuracia': acuracia,
        'mean_diff': mean_diff,
        'std_combined': std_combined,
        'ambiguity_score': ambiguity_score,
        'n_samples': len(X_sub),
        'overlap': overlap
    }

print("="*80)
print("BUSCA DO MELHOR PAR DE DÍGITOS MNIST PARA MAXIMIZAR REJEIÇÕES")
print("="*80)
print("\nCritério: Quanto MENOR o 'Ambiguity Score', MAIS rejeições esperadas")
print("(baixa separação entre classes = mais incerteza = mais rejeições)\n")

# Testar todos os pares
pares_mais_ambiguos = []

print(f"{'Par':<10} {'Acurácia':<12} {'Sep. Média':<15} {'Std':<12} {'Ambiguity':<12} {'Samples'}")
print("-"*80)

for a in range(10):
    for b in range(a+1, 10):
        try:
            metrics = avaliar_separabilidade(a, b)
            pares_mais_ambiguos.append((a, b, metrics))
            
            print(f"{a} vs {b:<5} {metrics['acuracia']*100:>6.2f}%     "
                  f"{metrics['mean_diff']:>8.3f}        "
                  f"{metrics['std_combined']:>6.3f}      "
                  f"{metrics['ambiguity_score']:>6.3f}      "
                  f"{metrics['n_samples']:>6}")
        except Exception as e:
            print(f"{a} vs {b:<5} ERRO: {str(e)[:40]}")

# Ordenar por ambiguity_score (menor = mais ambíguo)
pares_mais_ambiguos.sort(key=lambda x: x[2]['ambiguity_score'])

print("\n" + "="*80)
print("TOP 5 PARES MAIS AMBÍGUOS (melhor para gerar rejeições):")
print("="*80)

for i, (a, b, m) in enumerate(pares_mais_ambiguos[:5], 1):
    print(f"\n{i}. Dígitos {a} vs {b}")
    print(f"   - Acurácia: {m['acuracia']*100:.2f}%")
    print(f"   - Separação média: {m['mean_diff']:.3f}")
    print(f"   - Ambiguity Score: {m['ambiguity_score']:.3f} ⭐")
    print(f"   - Amostras: {m['n_samples']}")

print("\n" + "="*80)
print("TOP 5 PARES MAIS FÁCEIS (menos rejeições esperadas):")
print("="*80)

for i, (a, b, m) in enumerate(pares_mais_ambiguos[-5:][::-1], 1):
    print(f"\n{i}. Dígitos {a} vs {b}")
    print(f"   - Acurácia: {m['acuracia']*100:.2f}%")
    print(f"   - Separação média: {m['mean_diff']:.3f}")
    print(f"   - Ambiguity Score: {m['ambiguity_score']:.3f}")

print("\n" + "="*80)
print("RECOMENDAÇÃO FINAL:")
print("="*80)
melhor_par = pares_mais_ambiguos[0]
print(f"Use o par: {melhor_par[0]} vs {melhor_par[1]} (Ambiguity Score: {melhor_par[2]['ambiguity_score']:.3f})")
print("Este é o par mais ambíguo e deve gerar o MAIOR NÚMERO de rejeições!")
print("="*80)
