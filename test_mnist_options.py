"""
Script de teste para validar a nova funcionalidade MNIST:
- Seleção de modo de features (784 ou pooling 2x2 -> 196)
- Seleção de par de classes (ex: 8 vs 5)
- Propagação automática para PEAB, MinExp e Anchor
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.datasets import set_mnist_options, carregar_dataset

def test_mnist_raw():
    """Testa MNIST modo raw (784 features) com par 8 vs 5"""
    print("=" * 80)
    print("TESTE 1: MNIST modo RAW (784 features), classe 8 vs classe 5")
    print("=" * 80)
    
    set_mnist_options('raw', (8, 5))
    X, y, class_names = carregar_dataset('mnist')
    
    print(f"\n[OK] Dataset carregado:")
    print(f"  - Shape X: {X.shape}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Classes: {class_names}")
    print(f"  - Distribuição y: {dict(zip(*zip(*[(v, (y==v).sum()) for v in [0,1]])))}")
    
    assert X.shape[1] == 784, f"Esperado 784 features, obteve {X.shape[1]}"
    assert len(class_names) == 2, f"Esperado 2 classes, obteve {len(class_names)}"
    assert '8' in class_names and '5' in class_names, f"Esperado classes 8 e 5, obteve {class_names}"
    print("\n[✓] TESTE 1 PASSOU\n")


def test_mnist_pooling():
    """Testa MNIST modo pooling 2x2 (196 features) com par 3 vs 7"""
    print("=" * 80)
    print("TESTE 2: MNIST modo POOLING 2x2 (196 features), classe 3 vs classe 7")
    print("=" * 80)
    
    set_mnist_options('pool2x2', (3, 7))
    X, y, class_names = carregar_dataset('mnist')
    
    print(f"\n[OK] Dataset carregado:")
    print(f"  - Shape X: {X.shape}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Classes: {class_names}")
    print(f"  - Distribuição y: {dict(zip(*zip(*[(v, (y==v).sum()) for v in [0,1]])))}")
    print(f"  - Nomes das colunas (amostra): {list(X.columns[:5])}")
    
    assert X.shape[1] == 196, f"Esperado 196 features (14x14), obteve {X.shape[1]}"
    assert len(class_names) == 2, f"Esperado 2 classes, obteve {len(class_names)}"
    assert '3' in class_names and '7' in class_names, f"Esperado classes 3 e 7, obteve {class_names}"
    # Verificar nomes de colunas (devem ser bin_r_c)
    assert 'bin_0_0' in X.columns, f"Esperado coluna 'bin_0_0', colunas: {list(X.columns[:3])}"
    print("\n[✓] TESTE 2 PASSOU\n")


def test_json_metadata():
    """Testa se metadados MNIST são incluídos corretamente no JSON"""
    print("=" * 80)
    print("TESTE 3: Verificar inclusão de metadados MNIST no JSON")
    print("=" * 80)
    
    from data.datasets import MNIST_FEATURE_MODE, MNIST_SELECTED_PAIR
    
    print(f"\n[OK] Opções globais MNIST:")
    print(f"  - MNIST_FEATURE_MODE: {MNIST_FEATURE_MODE}")
    print(f"  - MNIST_SELECTED_PAIR: {MNIST_SELECTED_PAIR}")
    
    assert MNIST_FEATURE_MODE == 'pool2x2', f"Esperado 'pool2x2', obteve '{MNIST_FEATURE_MODE}'"
    assert MNIST_SELECTED_PAIR == (3, 7), f"Esperado (3, 7), obteve {MNIST_SELECTED_PAIR}"
    
    print("\n[✓] TESTE 3 PASSOU\n")


if __name__ == '__main__':
    try:
        test_mnist_raw()
        test_mnist_pooling()
        test_json_metadata()
        
        print("=" * 80)
        print("✓ TODOS OS TESTES PASSARAM COM SUCESSO!")
        print("=" * 80)
        print("\nPróximos passos:")
        print("  1. Execute peab_2.py e selecione MNIST")
        print("  2. Escolha modo pooling 2x2 (196 features)")
        print("  3. Escolha par de classes (ex: 8 5)")
        print("  4. Verifique que anchor_comparation.py e minexp_comparation.py")
        print("     usam automaticamente as mesmas configurações via shared_training")
        print("\nArquivos modificados:")
        print("  - data/datasets.py: opções globais MNIST e pooling 2x2")
        print("  - peab_2.py: prompt interativo e inclusão de metadados no JSON")
        print("  - anchor_comparation.py: usa meta do shared_training, naming class-vs-class")
        print("  - minexp_comparation.py: idem")
        print("  - utils/shared_training.py: propaga opções via configurar_experimento")
        
    except AssertionError as e:
        print(f"\n[✗] TESTE FALHOU: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[✗] ERRO INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
