"""
Script para reprocessar datasets MinExp em batch (automatizado).

Este script reexecuta o MinExp para datasets que não possuem 'per_instance',
permitindo que a validação funcione corretamente.
"""
import sys
import os

# Adicionar o diretório raiz ao path para importar módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def reprocessar_dataset_minexp(dataset_name: str):
    """
    Reprocessa um dataset específico com MinExp.
    
    Args:
        dataset_name: Nome do dataset para reprocessar
    
    Returns:
        bool: True se sucesso, False se erro
    """
    print(f"\n{'='*80}")
    print(f"REPROCESSANDO: {dataset_name}")
    print(f"{'='*80}\n")
    
    try:
        # Importar módulos necessários
        from data.datasets import selecionar_dataset_e_classe
        import utils.svm_explainer
        import utils.utility
        from utils.shared_training import get_shared_pipeline
        import importlib
        
        # Recarregar o módulo minexp para garantir código atualizado
        minexp_module = importlib.import_module('minexp')
        
        # Verificar se existe função principal
        if not hasattr(minexp_module, 'executar_minexp'):
            print("❌ Função executar_minexp não encontrada no módulo minexp")
            
            # Tentar importar componentes necessários
            print("   Tentando executar MinExp diretamente...")
            
            # Aqui você precisaria adaptar conforme a estrutura exata do minexp.py
            # Por ora, vamos orientar o usuário a executar manualmente
            print(f"\n⚠️  AVISO: Reprocessamento automático não disponível")
            print(f"   Execute manualmente: python minexp.py")
            print(f"   E selecione: {dataset_name}")
            return False
        
        # Executar MinExp para este dataset
        print(f"Executando MinExp para {dataset_name}...")
        resultado = minexp_module.executar_minexp(dataset_name)
        
        if resultado:
            print(f"\n✓ {dataset_name} reprocessado com sucesso!")
            return True
        else:
            print(f"\n✗ Erro ao reprocessar {dataset_name}")
            return False
    
    except Exception as e:
        print(f"\n❌ ERRO ao reprocessar {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Função principal do script."""
    
    import json
    
    JSON_DIR = "json"
    minexp_dir = os.path.join(JSON_DIR, 'minexp')
    
    print("\n" + "="*80)
    print("REPROCESSAMENTO EM BATCH - MinExp")
    print("="*80 + "\n")
    
    # Identificar datasets que precisam ser reprocessados
    datasets_problema = []
    
    if not os.path.exists(minexp_dir):
        print("❌ Pasta json/minexp/ não encontrada!")
        return
    
    json_files = [f for f in os.listdir(minexp_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        dataset_name = json_file.replace('.json', '')
        json_path = os.path.join(minexp_dir, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'per_instance' not in data:
                datasets_problema.append(dataset_name)
        except:
            datasets_problema.append(dataset_name)
    
    if not datasets_problema:
        print("✓ Todos os datasets já possuem 'per_instance'!")
        print("  Nenhum reprocessamento necessário.")
        return
    
    print(f"Datasets que precisam ser reprocessados: {len(datasets_problema)}")
    print()
    
    for i, dataset in enumerate(datasets_problema, 1):
        print(f"  {i}. {dataset}")
    
    print("\n" + "─"*80)
    print("AVISO IMPORTANTE:")
    print("─"*80)
    print("  O reprocessamento automático via script Python pode não funcionar")
    print("  corretamente devido à estrutura do código MinExp atual.")
    print()
    print("  RECOMENDAÇÃO: Execute manualmente para cada dataset:")
    print("    1. python minexp.py")
    print("    2. Selecione o dataset (ex: vertebral_column)")
    print("    3. Aguarde conclusão")
    print("    4. Repita para os outros datasets")
    print()
    print("  Se preferir reprocessar apenas vertebral_column (para teste):")
    print("    - É só executar uma vez para este dataset específico")
    print()
    
    resposta = input("Deseja continuar com tentativa automática? (s/N): ").strip().lower()
    
    if resposta != 's':
        print("\n✓ Operação cancelada. Execute manualmente conforme orientação acima.")
        return
    
    print(f"\n{'='*80}")
    print("INICIANDO REPROCESSAMENTO...")
    print(f"{'='*80}\n")
    
    sucesso = 0
    erro = 0
    
    for dataset in datasets_problema:
        if reprocessar_dataset_minexp(dataset):
            sucesso += 1
        else:
            erro += 1
    
    print(f"\n{'='*80}")
    print("RESUMO DO REPROCESSAMENTO:")
    print(f"{'='*80}")
    print(f"  ✓ Sucesso: {sucesso}/{len(datasets_problema)}")
    print(f"  ✗ Erro: {erro}/{len(datasets_problema)}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
