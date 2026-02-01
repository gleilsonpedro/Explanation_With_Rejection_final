"""
Opcoes de pooling para testar no MNIST:

ATUALMENTE IMPLEMENTADO:
- 'raw': 28x28 = 784 features (original, mais lento)
- 'pool2x2': 14x14 = 196 features (~4x mais rapido)

PARA IMPLEMENTAR OUTRAS OPCOES DE POOL:
Se quiser adicionar mais opcoes de pooling, aqui estao sugestoes:

1. pool4x4: 7x7 = 49 features (~16x mais rapido)
2. pool7x7: 4x4 = 16 features (~50x mais rapido)

Mas CUIDADO: muito pooling pode perder informacao importante!

RECOMENDACAO: Use combinacoes ao inves de mais pooling
"""

import numpy as np
import pandas as pd

def _pool4x4(arr28: np.ndarray) -> np.ndarray:
    """Aplica pooling 4x4 por media em imagem 28x28, resultando em 7x7 (49)."""
    out = np.zeros((7, 7), dtype=float)
    for r in range(7):
        for c in range(7):
            block = arr28[4*r:4*r+4, 4*c:4*c+4]
            out[r, c] = float(block.mean())
    return out


def _pool7x7(arr28: np.ndarray) -> np.ndarray:
    """Aplica pooling 7x7 por media em imagem 28x28, resultando em 4x4 (16)."""
    out = np.zeros((4, 4), dtype=float)
    for r in range(4):
        for c in range(4):
            block = arr28[7*r:7*r+7, 7*c:7*c+7]
            out[r, c] = float(block.mean())
    return out


# ============================================================================
# COMO TESTAR DIFERENTES CONFIGURACOES (SEM IMPLEMENTAR NOVOS POOLS)
# ============================================================================

OPCOES_DE_TESTE = {
    # OPCAO 1: Manter resolucao alta, reduzir features
    "alta_resolucao": {
        'feature_mode': 'raw',           # 784 features
        'top_k_features': 100,           # Seleciona 100 mais importantes
        'subsample_size': 0.01
    },
    
    # OPCAO 2: Reduzir resolucao (mais rapido)
    "media_resolucao": {
        'feature_mode': 'pool2x2',       # 196 features (14x14)
        'top_k_features': None,          # Usa todas
        'subsample_size': 0.01
    },
    
    # OPCAO 3: Reduzir resolucao + selecionar features (muito rapido)
    "baixa_resolucao": {
        'feature_mode': 'pool2x2',       # 196 features (14x14)
        'top_k_features': 50,            # Seleciona 50 mais importantes
        'subsample_size': 0.01
    },
    
    # OPCAO 4: Aumentar dados de treino (melhor modelo)
    "mais_dados": {
        'feature_mode': 'pool2x2',       # 196 features
        'top_k_features': None,
        'subsample_size': 0.05           # 5% ao inves de 1%
    },
    
    # OPCAO 5: Balanceado (bom compromisso)
    "balanceado": {
        'feature_mode': 'pool2x2',       # 196 features
        'top_k_features': 100,           # 100 features mais importantes
        'subsample_size': 0.02           # 2% dos dados
    }
}

# ============================================================================
# VELOCIDADE ESPERADA (estimativas)
# ============================================================================

"""
CONFIGURACAO                  | FEATURES | VELOCIDADE RELATIVA | PRECISAO
------------------------------|----------|---------------------|----------
raw + todas                   | 784      | 1x (baseline)       | 100%
raw + top_k=100              | 100      | 2-3x                | ~98%
pool2x2 + todas              | 196      | 4x                  | ~95%
pool2x2 + top_k=100          | 100      | 6-8x                | ~93%
pool2x2 + top_k=50           | 50       | 10-12x              | ~90%

PARA ANCHOR (que demora 8+ horas):
- pool2x2 + todas: ~2 horas
- pool2x2 + top_k=100: ~1 hora
- pool2x2 + top_k=50: ~30-40 minutos

RECOMENDACAO:
1. Teste inicial: pool2x2 + top_k=50 (para ver se funciona)
2. Validacao: pool2x2 + todas (para validar resultados)
3. Experimento final: raw + todas (para publicacao)
"""

if __name__ == "__main__":
    print("OPCOES DE POOLING DISPONIVEIS NO CODIGO ATUAL:")
    print("  1. 'raw' - 28x28 = 784 features (original)")
    print("  2. 'pool2x2' - 14x14 = 196 features (recomendado)")
    print()
    print("PARA ACELERAR AINDA MAIS, USE COMBINACOES:")
    print("  - feature_mode='pool2x2' + top_k_features=100")
    print("  - feature_mode='pool2x2' + top_k_features=50")
    print()
    print("Veja as configuracoes prontas em OPCOES_DE_TESTE acima")
