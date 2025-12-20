"""
Script de debug para investigar incompatibilidade de índices entre PEAB e PuLP
"""
import json
import os

dataset_name = "mnist_3_vs_8"

print("="*80)
print(f"DEBUG: Investigando índices para {dataset_name}")
print("="*80)

# Carregar JSONs
peab_file = f"json/peab/{dataset_name}.json"
pulp_file = f"json/pulp/{dataset_name}.json"

print(f"\n📂 Carregando arquivos:")
print(f"   PEAB: {peab_file}")
print(f"   PuLP: {pulp_file}")

# Verificar se existem
if not os.path.exists(peab_file):
    print(f"\n❌ Arquivo PEAB não encontrado: {peab_file}")
    exit(1)

if not os.path.exists(pulp_file):
    print(f"\n❌ Arquivo PuLP não encontrado: {pulp_file}")
    exit(1)

# Carregar
with open(peab_file, 'r', encoding='utf-8') as f:
    peab_data = json.load(f)

with open(pulp_file, 'r', encoding='utf-8') as f:
    pulp_data = json.load(f)

print("\n✅ Arquivos carregados com sucesso")

# Analisar estrutura PEAB
print("\n" + "="*80)
print("ESTRUTURA DO JSON PEAB")
print("="*80)
print(f"Chaves principais: {list(peab_data.keys())}")

# Verificar se tem per_instance
if 'per_instance' in peab_data:
    per_instance = peab_data['per_instance']
    print(f"\n✅ Tem 'per_instance': {len(per_instance)} instâncias")
    
    # Mostrar primeiras 5
    print("\nPrimeiras 5 instâncias do PEAB:")
    for i, exp in enumerate(per_instance[:5]):
        print(f"  [{i}] id={exp.get('id', 'N/A')}, tamanho={exp.get('explanation_size', 'N/A')}")
    
    # Coletar todos os IDs
    peab_ids = set()
    for exp in per_instance:
        idx = str(exp.get('id', ''))
        peab_ids.add(idx)
    
    print(f"\nTotal de IDs únicos no PEAB: {len(peab_ids)}")
    print(f"Amostra de IDs: {sorted(list(peab_ids))[:10]}")
else:
    print("\n❌ NÃO tem 'per_instance' - formato antigo!")
    print("   Não é possível fazer comparação instância-por-instância")
    exit(1)

# Analisar estrutura PuLP
print("\n" + "="*80)
print("ESTRUTURA DO JSON PULP")
print("="*80)
print(f"Chaves principais: {list(pulp_data.keys())}")

if 'explicacoes' in pulp_data:
    explicacoes = pulp_data['explicacoes']
    print(f"\n✅ Tem 'explicacoes': {len(explicacoes)} instâncias")
    
    # Mostrar primeiras 5
    print("\nPrimeiras 5 instâncias do PuLP:")
    for i, exp in enumerate(explicacoes[:5]):
        print(f"  [{i}] indice={exp.get('indice', 'N/A')}, tamanho={exp.get('tamanho', 'N/A')}")
    
    # Coletar todos os índices
    pulp_ids = set()
    for exp in explicacoes:
        idx = str(exp.get('indice', ''))
        pulp_ids.add(idx)
    
    print(f"\nTotal de IDs únicos no PuLP: {len(pulp_ids)}")
    print(f"Amostra de IDs: {sorted(list(pulp_ids))[:10]}")
else:
    print("\n❌ NÃO tem 'explicacoes'!")
    exit(1)

# Comparar conjuntos
print("\n" + "="*80)
print("COMPARAÇÃO DE ÍNDICES")
print("="*80)

print(f"\nPEAB: {len(peab_ids)} IDs")
print(f"PuLP: {len(pulp_ids)} IDs")

comuns = peab_ids & pulp_ids
apenas_peab = peab_ids - pulp_ids
apenas_pulp = pulp_ids - peab_ids

print(f"\n✅ IDs comuns: {len(comuns)}")
print(f"⚠️  Apenas no PEAB: {len(apenas_peab)}")
print(f"⚠️  Apenas no PuLP: {len(apenas_pulp)}")

if len(comuns) == 0:
    print("\n" + "="*80)
    print("🔍 PROBLEMA IDENTIFICADO: Nenhum ID comum!")
    print("="*80)
    
    print("\nTipo de IDs no PEAB:")
    amostra_peab = sorted(list(peab_ids))[:5]
    for idx in amostra_peab:
        print(f"   - '{idx}' (tipo: {type(idx).__name__}, len: {len(idx)})")
    
    print("\nTipo de IDs no PuLP:")
    amostra_pulp = sorted(list(pulp_ids))[:5]
    for idx in amostra_pulp:
        print(f"   - '{idx}' (tipo: {type(idx).__name__}, len: {len(idx)})")
    
    print("\n💡 POSSÍVEIS CAUSAS:")
    print("   1. PEAB está usando índice sequencial (0, 1, 2, ...)")
    print("   2. PuLP está usando índice original do DataFrame")
    print("   3. Splits de treino/teste diferentes")
    print("   4. Ordem diferente de processamento")
    
    print("\n🔧 SOLUÇÕES:")
    print("   1. Re-executar PEAB e PuLP para garantir mesmos splits")
    print("   2. Modificar código para usar índice consistente")
    print("   3. Usar apenas tamanho dos datasets para comparação agregada")
else:
    print(f"\n✅ Comparação é possível! {len(comuns)} instâncias em comum")
    
    if apenas_peab:
        print(f"\n⚠️  {len(apenas_peab)} instâncias apenas no PEAB:")
        print(f"     Amostra: {sorted(list(apenas_peab))[:10]}")
    
    if apenas_pulp:
        print(f"\n⚠️  {len(apenas_pulp)} instâncias apenas no PuLP:")
        print(f"     Amostra: {sorted(list(apenas_pulp))[:10]}")

print("\n" + "="*80)
