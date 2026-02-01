"""
Script de teste para diferentes configuracoes do MNIST.
Este script demonstra como testar diferentes combinacoes de parametros
para acelerar o processamento sem invalidar o experimento.
"""

# Exemplo de configuracoes para testar velocidade vs precisao

# CONFIGURACAO 1: ORIGINAL (mais lento, mais preciso)
CONFIG_ORIGINAL = {
    'feature_mode': 'raw',           # 784 features (28x28)
    'digit_pair': (3, 8),
    'top_k_features': None,          # Usa todas as features
    'test_size': 0.3,
    'rejection_cost': 0.10,
    'subsample_size': 0.01,
    'use_resize': False
}

# CONFIGURACAO 2: POOLING (4x mais rapido, boa precisao)
CONFIG_POOLING = {
    'feature_mode': 'pool2x2',       # 196 features (14x14) - reduz de 28x28 para 14x14
    'digit_pair': (3, 8),
    'top_k_features': None,
    'test_size': 0.3,
    'rejection_cost': 0.10,
    'subsample_size': 0.01,
    'use_resize': False
}

# CONFIGURACAO 3: TOP-K FEATURES (mais rapido, foca nas features mais importantes)
CONFIG_TOPK = {
    'feature_mode': 'raw',           # 784 features originais
    'digit_pair': (3, 8),
    'top_k_features': 100,           # Seleciona apenas as 100 features mais relevantes
    'test_size': 0.3,
    'rejection_cost': 0.10,
    'subsample_size': 0.01,
    'use_resize': False
}

# CONFIGURACAO 4: POOLING + TOP-K (muito mais rapido)
CONFIG_ULTRA_RAPIDO = {
    'feature_mode': 'pool2x2',       # 196 features
    'digit_pair': (3, 8),
    'top_k_features': 50,            # Seleciona apenas 50 features
    'test_size': 0.3,
    'rejection_cost': 0.10,
    'subsample_size': 0.05,          # Aumenta subsample para 5%
    'use_resize': False
}

# CONFIGURACAO 5: RESIZE CUSTOMIZADO (quando use_resize for implementado)
CONFIG_RESIZE_14 = {
    'feature_mode': 'raw',
    'digit_pair': (3, 8),
    'top_k_features': None,
    'test_size': 0.3,
    'rejection_cost': 0.10,
    'subsample_size': 0.01,
    'use_resize': 14                 # Redimensiona para 14x14 (196 features)
}

CONFIG_RESIZE_21 = {
    'feature_mode': 'raw',
    'digit_pair': (3, 8),
    'top_k_features': None,
    'test_size': 0.3,
    'rejection_cost': 0.10,
    'subsample_size': 0.01,
    'use_resize': 21                 # Redimensiona para 21x21 (441 features)
}

# ============================================================================
# RECOMENDACOES DE USO
# ============================================================================

"""
PARA TESTES RAPIDOS (desenvolvimento):
- Use CONFIG_POOLING ou CONFIG_ULTRA_RAPIDO
- Tempo esperado: 25-50% do tempo original

PARA VALIDACAO (pre-experimento final):
- Use CONFIG_POOLING ou CONFIG_TOPK
- Tempo esperado: 40-60% do tempo original
- Precisao esperada: 95-98% da precisao original

PARA EXPERIMENTO FINAL (publicacao):
- Use CONFIG_ORIGINAL
- Garante maxima precisao e comparabilidade

ENTENDENDO OS PARAMETROS:

1. feature_mode:
   - 'raw': usa pixels originais 28x28 = 784 features
   - 'pool2x2': aplica pooling 2x2, reduz para 14x14 = 196 features
   
2. top_k_features:
   - None: usa todas features disponiveis
   - N > 0: seleciona apenas N features mais relevantes (ANOVA F-test)
   - Valores sugeridos: 50, 100, 150, 200
   
3. use_resize:
   - False/None: nao redimensiona
   - N > 0: redimensiona imagem para NxN pixels
   - Valores sugeridos: 14 (rapido), 21 (intermediario)
   
4. subsample_size:
   - Fracao do dataset: 0.01 = 1%, 0.05 = 5%, 0.1 = 10%
   - Quanto menor, mais rapido, mas menos dados para treino

IMPACTO NA VELOCIDADE (estimativas):

feature_mode='pool2x2': ~4x mais rapido
top_k_features=100: ~2-3x mais rapido (com 784 features originais)
top_k_features=50: ~5-6x mais rapido (com 784 features originais)
subsample_size=0.05: ~5x mais rapido (mas afeta qualidade)

COMBINACOES RECOMENDADAS PARA ANCHOR (que demora 8+ horas):

1. Para testar se funciona:
   CONFIG_ULTRA_RAPIDO (esperado: ~30-60 minutos)

2. Para validar resultados:
   CONFIG_POOLING (esperado: ~2-3 horas)

3. Para experimento final:
   CONFIG_ORIGINAL (8+ horas, mas resultados oficiais)
"""

if __name__ == "__main__":
    print("Este arquivo contem apenas configuracoes de exemplo.")
    print("Para usar, copie uma das configuracoes acima para MNIST_CONFIG em peab.py")
    print("\nConfiguracoes disponiveis:")
    print("  - CONFIG_ORIGINAL: maxima precisao, mais lento")
    print("  - CONFIG_POOLING: 4x mais rapido, boa precisao")
    print("  - CONFIG_TOPK: 2-3x mais rapido, foca em features relevantes")
    print("  - CONFIG_ULTRA_RAPIDO: 10-15x mais rapido, para testes rapidos")
    print("  - CONFIG_RESIZE_14: usa redimensionamento para 14x14")
    print("  - CONFIG_RESIZE_21: usa redimensionamento para 21x21")
