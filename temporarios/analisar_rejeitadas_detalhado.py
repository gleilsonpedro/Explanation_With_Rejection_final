"""
Script para analisar TODAS as instâncias rejeitadas no MNIST
Mostra todas as opções, independente de qual método tem explicação menor
"""

import json
import numpy as np

print("="*80)
print("ANÁLISE DETALHADA DAS INSTÂNCIAS REJEITADAS - MNIST")
print("="*80)

# Carregar dados
with open('json/peab/mnist_3_vs_8.json', 'r') as f:
    peab_data = json.load(f)

with open('json/minexp/mnist.json', 'r') as f:
    minexp_data = json.load(f)

peab_instances = peab_data['per_instance']
minexp_instances = minexp_data['per_instance']

# Encontrar todas as rejeitadas
rejeitadas = []

for idx in range(len(peab_instances)):
    peab_inst = peab_instances[idx]
    minexp_inst = minexp_instances[idx]
    
    # Verificar se ambos rejeitaram
    if peab_inst['rejected'] and minexp_inst['rejected']:
        peab_size = peab_inst['explanation_size']
        minexp_size = minexp_inst['explanation_size']
        
        diferenca = peab_size - minexp_size  # Positivo = PEAB maior, Negativo = PEAB menor
        
        rejeitadas.append({
            'idx': idx,
            'y_true': peab_inst['y_true'],
            'peab_size': peab_size,
            'minexp_size': minexp_size,
            'diferenca': diferenca,
            'peab_melhor': peab_size < minexp_size,
            'decision_score': peab_inst['decision_score'],
            'abs_score': abs(peab_inst['decision_score'])
        })

print(f"\n✓ Total de instâncias rejeitadas por AMBOS os métodos: {len(rejeitadas)}")

if not rejeitadas:
    print("\n⚠️ Nenhuma instância foi rejeitada por ambos os métodos!")
else:
    # Ordenar por quem tem explicação menor
    print("\n" + "="*80)
    print("📊 TODAS AS INSTÂNCIAS REJEITADAS (ordenadas por tamanho total)")
    print("="*80)
    print(f"{'IDX':<6} {'Classe':<8} {'PEAB':<6} {'MinExp':<6} {'Dif':<6} {'Melhor':<12} {'Score':<10}")
    print("-"*80)
    
    # Ordenar por soma total (menor explicação total = mais simples)
    rejeitadas_sorted = sorted(rejeitadas, key=lambda x: x['peab_size'] + x['minexp_size'])
    
    for rej in rejeitadas_sorted:
        classe_str = "3" if rej['y_true'] == 0 else "8"
        melhor = "PEAB" if rej['peab_melhor'] else "MinExp"
        dif_str = f"{rej['diferenca']:+d}"
        
        print(f"{rej['idx']:<6} {classe_str:<8} {rej['peab_size']:<6} "
              f"{rej['minexp_size']:<6} {dif_str:<6} {melhor:<12} "
              f"{rej['decision_score']:>9.3f}")
    
    # Análise por quem é melhor
    print("\n" + "="*80)
    print("📈 ANÁLISE POR MÉTODO")
    print("="*80)
    
    peab_melhores = [r for r in rejeitadas if r['peab_melhor']]
    minexp_melhores = [r for r in rejeitadas if not r['peab_melhor']]
    
    print(f"\nPEAB tem explicação menor: {len(peab_melhores)} instâncias")
    if peab_melhores:
        print("\nMelhores casos para PEAB (ordenados por diferença):")
        peab_melhores.sort(key=lambda x: x['diferenca'])  # Mais negativo = maior vantagem PEAB
        print(f"{'IDX':<6} {'Classe':<8} {'PEAB':<6} {'MinExp':<6} {'Economia':<10} {'Score':<10}")
        print("-"*60)
        for rej in peab_melhores[:10]:
            classe_str = "3" if rej['y_true'] == 0 else "8"
            economia = abs(rej['diferenca'])
            print(f"{rej['idx']:<6} {classe_str:<8} {rej['peab_size']:<6} "
                  f"{rej['minexp_size']:<6} {economia:<10} "
                  f"{rej['decision_score']:>9.3f}")
    
    print(f"\nMinExp tem explicação menor: {len(minexp_melhores)} instâncias")
    if minexp_melhores:
        print("\nMelhores casos para MinExp (ordenados por diferença):")
        minexp_melhores.sort(key=lambda x: x['diferenca'], reverse=True)  # Mais positivo = maior vantagem MinExp
        print(f"{'IDX':<6} {'Classe':<8} {'PEAB':<6} {'MinExp':<6} {'Desvantagem':<12} {'Score':<10}")
        print("-"*60)
        for rej in minexp_melhores[:10]:
            classe_str = "3" if rej['y_true'] == 0 else "8"
            desvantagem = rej['diferenca']
            print(f"{rej['idx']:<6} {classe_str:<8} {rej['peab_size']:<6} "
                  f"{rej['minexp_size']:<6} {desvantagem:<12} "
                  f"{rej['decision_score']:>9.3f}")
    
    # Recomendações
    print("\n" + "="*80)
    print("💡 RECOMENDAÇÕES")
    print("="*80)
    
    if peab_melhores:
        melhor_peab = peab_melhores[0]
        print(f"\n🏆 MELHOR para mostrar PEAB ganhando:")
        print(f"   IDX_REJEITADA = {melhor_peab['idx']}")
        print(f"   - PEAB: {melhor_peab['peab_size']} pixels")
        print(f"   - MinExp: {melhor_peab['minexp_size']} pixels")
        print(f"   - PEAB economiza {abs(melhor_peab['diferenca'])} pixels!")
    
    # Opções com explicações mais curtas (independente de quem ganhou)
    print(f"\n📌 TOP 5 com MENOR TOTAL de pixels (mais simples de visualizar):")
    print(f"{'IDX':<6} {'Classe':<8} {'PEAB':<6} {'MinExp':<6} {'Total':<7} {'Melhor':<10}")
    print("-"*60)
    for rej in rejeitadas_sorted[:5]:
        classe_str = "3" if rej['y_true'] == 0 else "8"
        total = rej['peab_size'] + rej['minexp_size']
        melhor = "PEAB" if rej['peab_melhor'] else "MinExp"
        print(f"{rej['idx']:<6} {classe_str:<8} {rej['peab_size']:<6} "
              f"{rej['minexp_size']:<6} {total:<7} {melhor:<10}")
    
    print("\n" + "="*80)
    print("✅ Use os índices acima para escolher a melhor visualização!")
    print("="*80)
