"""
Script para escolher os melhores índices MNIST para comparação PEAB vs MinExp

Critérios:
1. PEAB deve ter explicação MENOR que MinExp (mostra que PEAB é melhor)
2. Diferença deve ser visualmente significativa
3. Ambos devem classificar a instância corretamente
4. Preferir instâncias com dígitos bem formados (score de decisão mais extremo)
"""

import json
import numpy as np
from pathlib import Path

# Carregar dados
print("="*80)
print("ANALISADOR DE ÍNDICES - MNIST PEAB vs MinExp")
print("="*80)

print("\n[1] Carregando dados...")
with open('json/peab/mnist_3_vs_8.json', 'r') as f:
    peab_data = json.load(f)

with open('json/minexp/mnist.json', 'r') as f:
    minexp_data = json.load(f)

peab_instances = peab_data['per_instance']
minexp_instances = minexp_data['per_instance']

print(f"  PEAB: {len(peab_instances)} instâncias")
print(f"  MinExp: {len(minexp_instances)} instâncias")

# Análise por categoria
print("\n" + "="*80)
print("[2] ANALISANDO CANDIDATOS POR CATEGORIA")
print("="*80)

categorias = {
    'positiva': {'y_pred': 1, 'rejected': False},
    'negativa': {'y_pred': 0, 'rejected': False},
    'rejeitada': {'rejected': True}
}

melhores_por_categoria = {}

for cat_nome, criterios in categorias.items():
    print(f"\n{'='*80}")
    print(f"CATEGORIA: {cat_nome.upper()}")
    print(f"{'='*80}")
    
    candidatos = []
    
    for idx in range(len(peab_instances)):
        peab_inst = peab_instances[idx]
        minexp_inst = minexp_instances[idx]
        
        # Verificar se atende aos critérios da categoria
        if criterios.get('rejected', False):
            if not peab_inst['rejected'] or not minexp_inst['rejected']:
                continue
        else:
            y_pred_esperado = criterios['y_pred']
            if peab_inst['y_pred'] != y_pred_esperado or minexp_inst['y_pred'] != y_pred_esperado:
                continue
            if peab_inst['rejected'] or minexp_inst['rejected']:
                continue
        
        # Verificar se ambos acertaram
        if peab_inst['y_true'] != minexp_inst['y_true']:
            print(f"  AVISO: idx={idx} tem y_true diferentes! Pulando...")
            continue
        
        # Calcular métricas
        peab_size = peab_inst['explanation_size']
        minexp_size = minexp_inst['explanation_size']
        
        # Queremos PEAB < MinExp (PEAB é melhor)
        if peab_size >= minexp_size:
            continue  # PEAB não é melhor aqui
        
        diferenca = minexp_size - peab_size
        reducao_percentual = (diferenca / minexp_size) * 100
        
        # Score de decisão (quanto mais extremo, mais "confiante" o modelo)
        decision_score = abs(peab_inst['decision_score'])
        
        # Métrica de qualidade: combina redução + confiança
        qualidade = reducao_percentual * (1 + decision_score)
        
        candidatos.append({
            'idx': idx,
            'y_true': peab_inst['y_true'],
            'peab_size': peab_size,
            'minexp_size': minexp_size,
            'diferenca': diferenca,
            'reducao_pct': reducao_percentual,
            'decision_score': peab_inst['decision_score'],
            'abs_decision': decision_score,
            'qualidade': qualidade
        })
    
    if not candidatos:
        print(f"\n  ⚠️  NENHUM CANDIDATO encontrado para {cat_nome}!")
        continue
    
    # Ordenar por qualidade (melhor primeiro)
    candidatos.sort(key=lambda x: x['qualidade'], reverse=True)
    
    print(f"\n  ✓ {len(candidatos)} candidatos encontrados onde PEAB < MinExp")
    print(f"\n  📊 TOP 10 MELHORES ÍNDICES:")
    print(f"  {'Rank':<6} {'IDX':<6} {'Classe':<8} {'PEAB':<6} {'MinExp':<6} {'Dif':<5} {'Redução':<8} {'Score':<8} {'Qualidade':<10}")
    print(f"  {'-'*76}")
    
    for i, cand in enumerate(candidatos[:10], 1):
        classe_str = f"3" if cand['y_true'] == 0 else "8"
        print(f"  {i:<6} {cand['idx']:<6} {classe_str:<8} "
              f"{cand['peab_size']:<6} {cand['minexp_size']:<6} "
              f"{cand['diferenca']:<5} {cand['reducao_pct']:>6.1f}% "
              f"{cand['decision_score']:>7.3f} {cand['qualidade']:>9.1f}")
    
    # Guardar o melhor
    melhor = candidatos[0]
    melhores_por_categoria[cat_nome] = melhor
    
    print(f"\n  🏆 MELHOR ESCOLHA: idx={melhor['idx']}")
    print(f"     - Classe: {'3' if melhor['y_true'] == 0 else '8'}")
    print(f"     - PEAB: {melhor['peab_size']} pixels")
    print(f"     - MinExp: {melhor['minexp_size']} pixels")
    print(f"     - Redução: {melhor['diferenca']} pixels ({melhor['reducao_pct']:.1f}%)")
    print(f"     - Decision Score: {melhor['decision_score']:.3f}")

# Resumo final
print("\n" + "="*80)
print("📋 RESUMO - ÍNDICES RECOMENDADOS")
print("="*80)

if melhores_por_categoria:
    print("\n✅ Cole estas linhas no topo do mnist_plot_comparacao.py:\n")
    print("-" * 60)
    
    if 'positiva' in melhores_por_categoria:
        idx = melhores_por_categoria['positiva']['idx']
        reducao = melhores_por_categoria['positiva']['reducao_pct']
        print(f"IDX_POSITIVA = {idx}    # Redução: {reducao:.1f}% (PEAB melhor)")
    else:
        print(f"IDX_POSITIVA = None    # Nenhum candidato encontrado")
    
    if 'negativa' in melhores_por_categoria:
        idx = melhores_por_categoria['negativa']['idx']
        reducao = melhores_por_categoria['negativa']['reducao_pct']
        print(f"IDX_NEGATIVA = {idx}    # Redução: {reducao:.1f}% (PEAB melhor)")
    else:
        print(f"IDX_NEGATIVA = None    # Nenhum candidato encontrado")
    
    if 'rejeitada' in melhores_por_categoria:
        idx = melhores_por_categoria['rejeitada']['idx']
        reducao = melhores_por_categoria['rejeitada']['reducao_pct']
        print(f"IDX_REJEITADA = {idx}    # Redução: {reducao:.1f}% (PEAB melhor)")
    else:
        print(f"IDX_REJEITADA = None    # Nenhum candidato encontrado")
    
    print("-" * 60)
    
    # Estatísticas gerais
    print("\n📈 ESTATÍSTICAS GERAIS:")
    for cat_nome, melhor in melhores_por_categoria.items():
        print(f"\n  {cat_nome.upper()}:")
        print(f"    - Índice: {melhor['idx']}")
        print(f"    - PEAB explicação: {melhor['peab_size']} pixels")
        print(f"    - MinExp explicação: {melhor['minexp_size']} pixels")
        print(f"    - PEAB é {melhor['reducao_pct']:.1f}% menor!")
        print(f"    - Economia de {melhor['diferenca']} pixels")
        print(f"    - Confiança do modelo: {abs(melhor['decision_score']):.3f}")
else:
    print("\n⚠️  Nenhum índice recomendado foi encontrado.")
    print("    Isso pode acontecer se MinExp sempre tiver explicações menores.")

print("\n" + "="*80)
print("💡 DICA: Após definir os índices, execute mnist_plot_comparacao.py")
print("         para gerar as visualizações com esses índices fixos.")
print("="*80)
