"""
Script para identificar quais datasets MinExp precisam ser reprocessados.
"""
import json
import os

JSON_DIR = "json"

def verificar_minexp_datasets():
    """Verifica quais datasets MinExp precisam ser reprocessados."""
    
    print("\n" + "="*80)
    print("VERIFICAÇÃO DE DATASETS MINEXP - Status de 'per_instance'")
    print("="*80 + "\n")
    
    minexp_dir = os.path.join(JSON_DIR, 'minexp')
    
    if not os.path.exists(minexp_dir):
        print("❌ Pasta json/minexp/ não encontrada!")
        return
    
    # Listar todos os arquivos JSON
    json_files = [f for f in os.listdir(minexp_dir) if f.endswith('.json') and f != '.gitkeep']
    
    if not json_files:
        print("❌ Nenhum arquivo JSON encontrado em json/minexp/")
        return
    
    print(f"Total de datasets encontrados: {len(json_files)}\n")
    
    datasets_ok = []
    datasets_problema = []
    
    for json_file in sorted(json_files):
        dataset_name = json_file.replace('.json', '')
        json_path = os.path.join(minexp_dir, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            has_per_instance = 'per_instance' in data
            
            if has_per_instance:
                num_explicacoes = len(data['per_instance'])
                print(f"✓ {dataset_name:30s} - OK ({num_explicacoes} explicações)")
                datasets_ok.append(dataset_name)
            else:
                print(f"✗ {dataset_name:30s} - FALTANDO per_instance")
                datasets_problema.append(dataset_name)
        
        except Exception as e:
            print(f"⚠ {dataset_name:30s} - ERRO ao ler: {e}")
            datasets_problema.append(dataset_name)
    
    # Resumo
    print("\n" + "─"*80)
    print("RESUMO:")
    print("─"*80)
    print(f"  ✓ Datasets OK (com per_instance):        {len(datasets_ok)}")
    print(f"  ✗ Datasets com problema (sem per_instance): {len(datasets_problema)}")
    
    if datasets_problema:
        print("\n" + "─"*80)
        print("DATASETS QUE PRECISAM SER REPROCESSADOS:")
        print("─"*80)
        for i, dataset in enumerate(datasets_problema, 1):
            print(f"  {i}. {dataset}")
        
        print("\n" + "─"*80)
        print("COMO REPROCESSAR:")
        print("─"*80)
        print("  Opção 1 - Manual (um por vez):")
        print("    1. Execute: python minexp.py")
        print("    2. Selecione o dataset")
        print("    3. Aguarde conclusão")
        print("    4. Repita para cada dataset")
        print()
        print("  Opção 2 - Automatizado (todos de uma vez):")
        print("    1. Execute: python temporarios/reprocessar_minexp_batch.py")
        print("    2. Escolha quais datasets reprocessar")
        print()
    else:
        print("\n✓ Todos os datasets estão OK! Nenhum reprocessamento necessário.")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    verificar_minexp_datasets()
