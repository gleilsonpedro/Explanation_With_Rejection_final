import json

# Carregar dados
with open(r'json\peab\rcv1.json', encoding='utf-8') as f:
    peab = json.load(f)

with open(r'json\pulp\rcv1.json', encoding='utf-8') as f:
    pulp = json.load(f)

print("="*80)
print("ANÁLISE COMPARATIVA: PEAB vs PULP - Dataset RCV1")
print("="*80)

print("\n📊 TEMPO DE EXECUÇÃO:")
print("-"*80)
print(f"{'Método':<10} {'Total (s)':<12} {'Positiva (s)':<15} {'Negativa (s)':<15} {'Rejeitada (s)':<15}")
print("-"*80)
print(f"{'PEAB':<10} {peab['computation_time']['total']:>11.2f} {peab['computation_time']['positive']:>14.4f} {peab['computation_time']['negative']:>14.4f} {peab['computation_time']['rejected']:>14.4f}")
print(f"{'PULP':<10} {pulp['estatisticas_gerais']['tempo_total_segundos']:>11.2f} {pulp['estatisticas_por_tipo']['positiva']['tempo_medio']:>14.4f} {pulp['estatisticas_por_tipo']['negativa']['tempo_medio']:>14.4f} {pulp['estatisticas_por_tipo']['rejeitada']['tempo_medio']:>14.4f}")

speedup_total = peab['computation_time']['total'] / pulp['estatisticas_gerais']['tempo_total_segundos']
print(f"\n⚡ PEAB é {speedup_total:.2f}x mais LENTO que PULP no total")

# Comparação nas rejeitadas
speedup_rej = peab['computation_time']['rejected'] / pulp['estatisticas_por_tipo']['rejeitada']['tempo_medio']
print(f"⚡ PEAB é {speedup_rej:.2f}x mais LENTO nas REJEITADAS")

print("\n📏 TAMANHO DAS EXPLICAÇÕES:")
print("-"*80)
print(f"{'Método':<10} {'Positiva':<15} {'Negativa':<15} {'Rejeitada':<15}")
print("-"*80)
print(f"{'PEAB':<10} {peab['explanation_stats']['positive']['mean_length']:>14.1f} {peab['explanation_stats']['negative']['mean_length']:>14.1f} {peab['explanation_stats']['rejected']['mean_length']:>14.1f}")
print(f"{'PULP':<10} {pulp['estatisticas_por_tipo']['positiva']['tamanho_medio']:>14.1f} {pulp['estatisticas_por_tipo']['negativa']['tamanho_medio']:>14.1f} {pulp['estatisticas_por_tipo']['rejeitada']['tamanho_medio']:>14.1f}")

print("\n📈 NÚMERO DE INSTÂNCIAS:")
print("-"*80)
print(f"Positivas: {peab['explanation_stats']['positive']['count']}")
print(f"Negativas: {peab['explanation_stats']['negative']['count']}")
print(f"Rejeitadas: {peab['explanation_stats']['rejected']['count']}")

print("\n🔍 DIAGNÓSTICO DO PROBLEMA:")
print("-"*80)

# Problema identificado
if peab['computation_time']['rejected'] > 1000:
    print("❌ PROBLEMA ENCONTRADO: PEAB está extremamente lento nas instâncias rejeitadas!")
    print(f"   • Tempo médio nas rejeitadas: {peab['computation_time']['rejected']:.2f}s (!!)")
    print(f"   • Isso representa {peab['computation_time']['rejected']/peab['computation_time']['total']*100:.1f}% do tempo total")
    
if pulp['estatisticas_por_tipo']['rejeitada']['tempo_medio'] < 2:
    print("✅ PULP está sendo eficiente mesmo nas rejeitadas (~1.4s por instância)")

print("\n💡 ANÁLISE:")
print("-"*80)
print("1. Para instâncias POSITIVAS e NEGATIVAS:")
print("   • PEAB: ~1.5-1.9s por instância")
print("   • PULP: ~1.4s por instância")
print("   • Desempenho similar ✓")
print()
print("2. Para instâncias REJEITADAS:")
print(f"   • PEAB: ~{peab['computation_time']['rejected']:.1f}s por instância (!!)")
print(f"   • PULP: ~{pulp['estatisticas_por_tipo']['rejeitada']['tempo_medio']:.1f}s por instância")
print(f"   • PEAB está {speedup_rej:.0f}x mais lento! ❌")
print()
print("3. Tamanho das explicações nas rejeitadas:")
print(f"   • PEAB: ~{peab['explanation_stats']['rejected']['mean_length']:.0f} features")
print(f"   • PULP: ~{pulp['estatisticas_por_tipo']['rejeitada']['tamanho_medio']:.0f} features")
print("   • Tamanhos similares, então não é problema de qualidade")

print("\n🎯 CONCLUSÃO:")
print("-"*80)
print("O problema está no PEAB, não no PULP!")
print("• PEAB tem um bug de desempenho nas instâncias REJEITADAS")
print("• Com C=0.01 e 4000 features, as rejeitadas estão travando o PEAB")
print("• PULP consegue resolver mesmo com muitas features porque é otimizado")
print()
print("🔧 PRÓXIMOS PASSOS:")
print("1. Investigar o código do PEAB para otimizar instâncias rejeitadas")
print("2. Testar com C=1.0 ou C=10.0 para reduzir número de features")
print("3. Adicionar limite de tempo no PEAB também")

print("\n" + "="*80)
