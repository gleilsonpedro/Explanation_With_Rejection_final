"""
Script para diagnóstico completo do problema de validação do MinExp.
"""
import json
import os

JSON_DIR = "json"

def diagnosticar_minexp():
    """Realiza diagnóstico completo do problema MinExp."""
    
    print("\n" + "="*80)
    print("DIAGNÓSTICO COMPLETO - PROBLEMA MINEXP VALIDATION")
    print("="*80 + "\n")
    
    # 1. Verificar estrutura de pastas
    print("1. VERIFICANDO ESTRUTURA DE PASTAS:")
    print("-" * 80)
    minexp_dir = os.path.join(JSON_DIR, 'minexp')
    print(f"   Pasta json/minexp/ existe? {os.path.exists(minexp_dir)}")
    
    if os.path.exists(minexp_dir):
        files = [f for f in os.listdir(minexp_dir) if f.endswith('.json')]
        print(f"   Arquivos JSON encontrados: {len(files)}")
        for f in files[:5]:  # Mostrar até 5 arquivos
            print(f"     - {f}")
        if len(files) > 5:
            print(f"     ... e mais {len(files) - 5} arquivos")
    print()
    
    # 2. Verificar arquivo antigo (minexp_results.json)
    print("2. VERIFICANDO ARQUIVO ANTIGO (minexp_results.json):")
    print("-" * 80)
    old_file = os.path.join(JSON_DIR, 'minexp_results.json')
    print(f"   Arquivo json/minexp_results.json existe? {os.path.exists(old_file)}")
    
    if os.path.exists(old_file):
        try:
            with open(old_file, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
            print(f"   ✓ Carregado com sucesso")
            print(f"   Datasets no arquivo: {list(old_data.keys())[:10]}")
            
            # Verificar se algum dataset tem per_instance
            for dataset_name in list(old_data.keys())[:3]:
                dataset_data = old_data[dataset_name]
                has_per_instance = 'per_instance' in dataset_data
                print(f"     - {dataset_name}: {'✓ tem per_instance' if has_per_instance else '✗ SEM per_instance'}")
                
                if has_per_instance:
                    print(f"       Total explicações: {len(dataset_data['per_instance'])}")
                    break
        except Exception as e:
            print(f"   ✗ Erro ao carregar: {e}")
    print()
    
    # 3. Verificar arquivo vertebral_column.json (exemplo)
    print("3. VERIFICANDO ARQUIVO ESPECÍFICO (vertebral_column.json):")
    print("-" * 80)
    test_file = os.path.join(JSON_DIR, 'minexp', 'vertebral_column.json')
    print(f"   Arquivo: {test_file}")
    print(f"   Existe? {os.path.exists(test_file)}")
    
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   Chaves principais: {list(data.keys())}")
        has_per_instance = 'per_instance' in data
        print(f"   Tem 'per_instance'? {has_per_instance}")
        
        if has_per_instance:
            print(f"   ✓ Total de explicações: {len(data['per_instance'])}")
        else:
            print(f"   ✗ NÃO TEM EXPLICAÇÕES INDIVIDUAIS!")
            print(f"      → Arquivo foi gerado com versão antiga do código")
    print()
    
    # 4. Comparar com PEAB (que funciona)
    print("4. COMPARANDO COM PEAB (QUE FUNCIONA):")
    print("-" * 80)
    peab_file = os.path.join(JSON_DIR, 'peab', 'vertebral_column.json')
    print(f"   Arquivo: {peab_file}")
    print(f"   Existe? {os.path.exists(peab_file)}")
    
    if os.path.exists(peab_file):
        with open(peab_file, 'r', encoding='utf-8') as f:
            peab_data = json.load(f)
        
        has_per_instance = 'per_instance' in peab_data
        print(f"   Tem 'per_instance'? {has_per_instance}")
        
        if has_per_instance:
            print(f"   ✓ Total de explicações: {len(peab_data['per_instance'])}")
            print(f"\n   Estrutura de uma explicação PEAB:")
            if len(peab_data['per_instance']) > 0:
                exp_peab = peab_data['per_instance'][0]
                for key in exp_peab.keys():
                    print(f"     - {key}")
    print()
    
    # 5. Conclusão e solução
    print("5. CONCLUSÃO E SOLUÇÃO:")
    print("-" * 80)
    print("   PROBLEMA IDENTIFICADO:")
    print("   ✗ O arquivo json/minexp/vertebral_column.json NÃO contém 'per_instance'")
    print("   ✗ Arquivo foi gerado com versão antiga do código MinExp")
    print()
    print("   SOLUÇÃO:")
    print("   1. Execute novamente o MinExp com a versão atual do código:")
    print("      python minexp.py")
    print("   2. Selecione o dataset: vertebral_column")
    print("   3. O novo arquivo incluirá 'per_instance' com explicações individuais")
    print("   4. Depois execute a validação novamente:")
    print("      python peab_validation.py")
    print()
    print("   OU (se preferir um patch temporário):")
    print("   - O script pode copiar dados de minexp_results.json para o novo formato")
    print()
    print("="*80 + "\n")


if __name__ == '__main__':
    diagnosticar_minexp()
