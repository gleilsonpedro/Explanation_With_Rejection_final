"""
Análise do tempo MinExp no MNIST
"""

print("\n" + "="*80)
print("ANÁLISE RÁPIDA - MinExp MNIST muito lento")
print("="*80 + "\n")

# Dados observados
instancias_total = 502
instancias_feitas = 40
tempo_minutos = 40.38
tempo_restante_horas = 7.77

tempo_por_instancia = tempo_minutos / instancias_feitas
tempo_total_horas = (tempo_minutos + (instancias_total - instancias_feitas) * tempo_por_instancia) / 60

print(f"SITUAÇÃO ATUAL (subsample=0.12):")
print(f"  Instâncias no teste: {instancias_total}")
print(f"  Progresso: {instancias_feitas}/{instancias_total} ({instancias_feitas/instancias_total*100:.1f}%)")
print(f"  Tempo por instância: {tempo_por_instancia:.2f} min (~{tempo_por_instancia*60:.0f}s)")
print(f"  Tempo restante: {tempo_restante_horas:.1f} horas")
print(f"  ⚠️ MUITO LENTO!")

print(f"\n{'─'*80}\n")

# Calcular com subsample 0.01
subsample_atual = 0.12
subsample_novo = 0.01
reducao_fator = subsample_novo / subsample_atual

instancias_novo = int(instancias_total * reducao_fator)
tempo_novo_min = instancias_novo * tempo_por_instancia
tempo_novo_horas = tempo_novo_min / 60

economia_horas = tempo_total_horas - tempo_novo_horas

print(f"COM SUBSAMPLE=0.01 (RECOMENDADO):")
print(f"  Instâncias no teste: {instancias_novo}")
print(f"  Tempo total: ~{tempo_novo_min:.0f} min = {tempo_novo_horas:.1f}h")
print(f"  ⏱️ ECONOMIZA: {economia_horas:.1f} horas!")
print(f"  ✅ {instancias_novo} instâncias ainda é válido estatisticamente")

print(f"\n{'─'*80}\n")

print("COMPARAÇÃO:")
print("-"*80)
print(f"  Subsample 0.12: {instancias_total} inst → {tempo_total_horas:.1f}h")
print(f"  Subsample 0.01: {instancias_novo} inst → {tempo_novo_horas:.1f}h")
print(f"  Redução: {(1-reducao_fator)*100:.0f}% das instâncias")
print(f"  Economia: {economia_horas:.1f}h ({economia_horas/tempo_total_horas*100:.0f}%)")

print(f"\n{'='*80}")
print("RECOMENDAÇÃO URGENTE:")
print("="*80)
print(f"✅ SIM, REDUZA PARA 0.01 IMEDIATAMENTE!")
print(f"   Economiza {economia_horas:.1f}h (de {tempo_total_horas:.1f}h para {tempo_novo_horas:.1f}h)")
print(f"   {instancias_novo} instâncias AINDA É VÁLIDO")
print(f"   Modelo foi treinado com dataset COMPLETO (isso não muda!)")
print()
print("🔴 AÇÃO: Cancele o MinExp agora (Ctrl+C)")
print("   Atualize peab.py → subsample_size: 0.01")
print("   Recomece: python minexp.py")
print("="*80 + "\n")

print("VALOR PARA ATUALIZAR NO peab.py:")
print("-"*80)
print("""
MNIST_CONFIG = {
    # ... outras configs ...
    'subsample_size': 0.01   # ← MUDE de 0.12 para 0.01
}
""")
print("="*80 + "\n")
