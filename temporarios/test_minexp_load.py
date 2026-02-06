"""
Script de teste para verificar carregamento de dados MinExp.
"""
import json
import os

JSON_DIR = "json"

def testar_carregamento_minexp(dataset='vertebral_column'):
    """Testa o carregamento de dados do MinExp."""
    
    # Caminho do arquivo MinExp
    minexp_path = os.path.join(JSON_DIR, 'minexp', f'{dataset}.json')
    
    print(f"\n{'='*70}")
    print(f"TESTE DE CARREGAMENTO - MinExp")
    print(f"{'='*70}\n")
    
    print(f"Arquivo: {minexp_path}")
    print(f"Existe? {os.path.exists(minexp_path)}\n")
    
    if not os.path.exists(minexp_path):
        print(f"❌ Arquivo não encontrado!")
        return
    
    # Carregar JSON
    with open(minexp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Verificar estrutura
    print("Chaves principais no JSON:")
    for key in data.keys():
        print(f"  - {key}")
    
    # Verificar explicações individuais
    print(f"\n{'─'*70}")
    print("VERIFICANDO EXPLICAÇÕES INDIVIDUAIS:")
    print(f"{'─'*70}\n")
    
    # Tentar diferentes formas de acesso
    explicacoes_1 = data.get('explicacoes', None)
    explicacoes_2 = data.get('per_instance', None)
    
    print(f"data.get('explicacoes'): {'✓ ENCONTRADO' if explicacoes_1 else '✗ NÃO ENCONTRADO'}")
    print(f"data.get('per_instance'): {'✓ ENCONTRADO' if explicacoes_2 else '✗ NÃO ENCONTRADO'}")
    
    if explicacoes_2:
        print(f"\n✓ Explicações individuais encontradas em 'per_instance'")
        print(f"  Total de explicações: {len(explicacoes_2)}")
        
        if len(explicacoes_2) > 0:
            print(f"\n  Exemplo da primeira explicação:")
            exemplo = explicacoes_2[0]
            for key, value in exemplo.items():
                print(f"    - {key}: {value}")
    else:
        print(f"\n❌ NÃO há explicações individuais no JSON!")
        print(f"   O JSON contém apenas estatísticas agregadas.")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    testar_carregamento_minexp('vertebral_column')
