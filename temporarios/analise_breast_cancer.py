"""
ANÁLISE DETALHADA: Breast Cancer
=================================
Por que PULP tem 0 positivas enquanto PEAB tem 84?
"""
import json
from pathlib import Path

print("="*80)
print("ANÁLISE: BREAST CANCER - Por que PULP não tem positivas?")
print("="*80)

# Carregar JSONs
with open("json/peab/breast_cancer.json") as f:
    peab = json.load(f)

with open("json/pulp/breast_cancer.json") as f:
    pulp = json.load(f)

# Thresholds
print("\n1️⃣ THRESHOLDS:")
print(f"   PEAB: t+ = {peab['thresholds']['t_plus']:.6f}, t- = {peab['thresholds']['t_minus']:.6f}")
print(f"   PULP: t+ = {pulp['t_plus']:.6f}, t- = {pulp['t_minus']:.6f}")
print(f"   Diferença t+: {abs(peab['thresholds']['t_plus'] - pulp['t_plus']):.8f}")

# Buscar instâncias que PEAB classifica como POSITIVA
print("\n2️⃣ INSTÂNCIAS POSITIVAS NO PEAB:")
peab_positivas = [e for e in peab.get('explicacoes', []) if e.get('tipo_predicao') == 'POSITIVA']
print(f"   Total: {len(peab_positivas)} instâncias")

if peab_positivas:
    print("\n   Primeiras 10 instâncias positivas no PEAB:")
    for i, exp in enumerate(peab_positivas[:10], 1):
        idx = exp['indice']
        features = exp['features_selecionadas']
        print(f"   {i:2d}. Instância {idx}: {len(features)} features -> {features}")

# Verificar as mesmas instâncias no PULP
print("\n3️⃣ COMO O PULP CLASSIFICOU ESSAS MESMAS INSTÂNCIAS:")

# Criar dicionário de explicações PULP por índice
pulp_por_indice = {e['indice']: e for e in pulp.get('explicacoes', [])}

divergencias = []
for exp_peab in peab_positivas[:20]:  # Analisar primeiras 20
    idx = exp_peab['indice']
    
    if idx in pulp_por_indice:
        exp_pulp = pulp_por_indice[idx]
        tipo_pulp = exp_pulp['tipo_predicao']
        
        if tipo_pulp != 'POSITIVA':
            divergencias.append({
                'indice': idx,
                'peab_tipo': 'POSITIVA',
                'pulp_tipo': tipo_pulp,
                'peab_features': exp_peab['features_selecionadas'],
                'pulp_features': exp_pulp['features_selecionadas'],
                'peab_tam': len(exp_peab['features_selecionadas']),
                'pulp_tam': len(exp_pulp['features_selecionadas'])
            })

print(f"\n   Divergências encontradas: {len(divergencias)}/{len(peab_positivas[:20])}")

if divergencias:
    print("\n   Detalhes das divergências:")
    for i, div in enumerate(divergencias[:5], 1):
        print(f"\n   {i}. Instância {div['indice']}:")
        print(f"      PEAB: {div['peab_tipo']} ({div['peab_tam']} features)")
        print(f"            Features: {div['peab_features']}")
        print(f"      PULP: {div['pulp_tipo']} ({div['pulp_tam']} features)")
        print(f"            Features: {div['pulp_features']}")

# Estatísticas gerais de conversão
print("\n4️⃣ DISTRIBUIÇÃO DAS CONVERSÕES:")
conversoes = {}
for exp_peab in peab_positivas:
    idx = exp_peab['indice']
    if idx in pulp_por_indice:
        tipo_pulp = pulp_por_indice[idx]['tipo_predicao']
        conversoes[tipo_pulp] = conversoes.get(tipo_pulp, 0) + 1

print(f"\n   Das {len(peab_positivas)} positivas do PEAB:")
for tipo, count in conversoes.items():
    pct = (count / len(peab_positivas)) * 100
    print(f"   → {count:3d} ({pct:5.1f}%) viraram {tipo} no PULP")

print("\n" + "="*80)
print("CONCLUSÃO:")
print("="*80)
print("""
O problema é que o PULP está sendo MUITO CONSERVADOR!

Possíveis razões:
1. As restrições do PULP são mais estritas (EPSILON pode estar pequeno demais)
2. A normalização dos scores pode estar diferente
3. O solver CBC pode estar aplicando tolerâncias conservadoras

SOLUÇÃO POTENCIAL (sem re-executar tudo):
- Relaxar o EPSILON no código do PULP
- Ajustar tolerâncias do solver CBC
- Verificar se a normalização está consistente

MAS CUIDADO: Isso requer re-executar PULP em todos os datasets!
""")
print("="*80)
