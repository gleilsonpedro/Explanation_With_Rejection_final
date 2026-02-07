"""
Script de teste para verificar a correção do subsample_size.

ANTES DA CORREÇÃO (ERRADO):
- subsample_size=0.10 aplicado no dataset completo
- 10% do dataset total = 100 instâncias
- Divisão 70/30 → 70 treino + 30 teste
- Resultado: Treino em 70 e teste em 30 (MUITO POUCO!)

DEPOIS DA CORREÇÃO (CORRETO):
- Dataset completo = 1000 instâncias
- Divisão 70/30 → 700 treino + 300 teste
- subsample_size=0.10 aplicado só no teste → 30 instâncias de teste
- Resultado: Treino em 700 e teste em 30
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def simular_pipeline_antigo(total_instancias=1000, subsample_size=0.10, test_size=0.3):
    """Simula o comportamento ANTIGO (ERRADO)."""
    print("\n" + "="*80)
    print("SIMULAÇÃO DO PIPELINE ANTIGO (ERRADO)")
    print("="*80)
    
    # 1. Dataset completo
    print(f"1. Dataset original: {total_instancias} instâncias")
    
    # 2. Aplicar subsample ANTES da divisão (ERRADO!)
    n_apos_subsample = int(total_instancias * subsample_size)
    print(f"2. Aplicar subsample ({subsample_size*100:.0f}%): {n_apos_subsample} instâncias")
    
    # 3. Dividir em treino/teste
    n_treino = int(n_apos_subsample * (1 - test_size))
    n_teste = n_apos_subsample - n_treino
    print(f"3. Dividir em treino/teste ({(1-test_size)*100:.0f}%/{test_size*100:.0f}%):")
    print(f"   - Treino: {n_treino} instâncias ({n_treino/total_instancias*100:.1f}% do original)")
    print(f"   - Teste: {n_teste} instâncias ({n_teste/total_instancias*100:.1f}% do original)")
    print(f"\n❌ PROBLEMA: Treino com apenas {n_treino/total_instancias*100:.1f}% do dataset!")
    print(f"❌ PROBLEMA: Teste com apenas {n_teste/total_instancias*100:.1f}% do dataset!")
    
    return n_treino, n_teste


def simular_pipeline_corrigido(total_instancias=1000, subsample_size=0.10, test_size=0.3):
    """Simula o comportamento CORRIGIDO (CORRETO)."""
    print("\n" + "="*80)
    print("SIMULAÇÃO DO PIPELINE CORRIGIDO (CORRETO)")
    print("="*80)
    
    # 1. Dataset completo
    print(f"1. Dataset original: {total_instancias} instâncias")
    
    # 2. Dividir em treino/teste PRIMEIRO
    n_treino_full = int(total_instancias * (1 - test_size))
    n_teste_full = total_instancias - n_treino_full
    print(f"2. Dividir em treino/teste ({(1-test_size)*100:.0f}%/{test_size*100:.0f}%):")
    print(f"   - Treino: {n_treino_full} instâncias")
    print(f"   - Teste: {n_teste_full} instâncias")
    
    # 3. Aplicar subsample APENAS no teste
    n_teste_subsampled = int(n_teste_full * subsample_size)
    print(f"3. Aplicar subsample apenas no TESTE ({subsample_size*100:.0f}%):")
    print(f"   - Treino: {n_treino_full} instâncias ({n_treino_full/total_instancias*100:.0f}% do original)")
    print(f"   - Teste: {n_teste_subsampled} instâncias ({n_teste_subsampled/total_instancias*100:.1f}% do original)")
    print(f"\n✅ CORRETO: Treino com {n_treino_full/total_instancias*100:.0f}% do dataset (COMPLETO)")
    print(f"✅ CORRETO: Teste reduzido apenas para acelerar explicações")
    
    return n_treino_full, n_teste_subsampled


def testar_codigo_real():
    """Testa o código real do peab.py."""
    print("\n" + "="*80)
    print("TESTE COM CÓDIGO REAL")
    print("="*80)
    
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from peab import configurar_experimento, aplicar_subsample_teste, DATASET_CONFIG
        
        # Testar com breast_cancer (se não tiver subsample configurado)
        dataset_name = 'breast_cancer'
        
        print(f"\nTestando dataset: {dataset_name}")
        
        # Carregar dataset
        X, y, nomes_classes, rejection_cost, test_size = configurar_experimento(dataset_name)
        print(f"Dataset carregado: {len(y)} instâncias totais")
        
        # Dividir treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        print(f"Após split: {len(y_train)} treino + {len(y_test)} teste")
        
        # Aplicar subsample no teste (se configurado)
        cfg = DATASET_CONFIG.get(dataset_name, {})
        subsample_size = cfg.get('subsample_size', None)
        
        if subsample_size:
            X_test_sub, y_test_sub = aplicar_subsample_teste(X_test, y_test, subsample_size)
            print(f"Após subsample: {len(y_train)} treino + {len(y_test_sub)} teste")
            print(f"\n✅ Treino mantém {len(y_train)} instâncias (COMPLETO)")
            print(f"✅ Teste reduzido para {len(y_test_sub)} instâncias")
        else:
            print(f"\nℹ️  Dataset {dataset_name} não tem subsample configurado")
            print(f"   Treino: {len(y_train)} instâncias")
            print(f"   Teste: {len(y_test)} instâncias")
        
    except Exception as e:
        print(f"\n⚠️  Erro ao testar código real: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Exemplo: creditcard com subsample_size=0.02 (2%)
    print("\n" + "#"*80)
    print("EXEMPLO: Dataset creditcard com subsample_size=0.02 (2%)")
    print("#"*80)
    
    total = 284807  # Tamanho real do creditcard
    subsample = 0.02
    test_size = 0.3
    
    n_treino_old, n_teste_old = simular_pipeline_antigo(total, subsample, test_size)
    n_treino_new, n_teste_new = simular_pipeline_corrigido(total, subsample, test_size)
    
    print("\n" + "="*80)
    print("COMPARAÇÃO FINAL")
    print("="*80)
    print(f"{'':30} | {'ANTIGO (ERRADO)':^20} | {'CORRIGIDO':^20}")
    print("-"*80)
    print(f"{'Instâncias de TREINO':30} | {n_treino_old:^20} | {n_treino_new:^20}")
    print(f"{'Instâncias de TESTE':30} | {n_teste_old:^20} | {n_teste_new:^20}")
    print(f"{'% dataset original (treino)':30} | {n_treino_old/total*100:^20.2f}% | {n_treino_new/total*100:^20.0f}%")
    print("="*80)
    
    print(f"\n💡 DIFERENÇA:")
    print(f"   Treino: {n_treino_new - n_treino_old:+,} instâncias ({(n_treino_new/n_treino_old - 1)*100:+.0f}%)")
    print(f"   O treino agora usa {n_treino_new/total*100:.0f}% do dataset ao invés de {n_treino_old/total*100:.1f}%")
    
    # Testar com código real
    testar_codigo_real()
