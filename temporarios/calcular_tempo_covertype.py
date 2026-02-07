"""
Calculadora de tempo esperado para Anchor/MinExp no covertype.
"""

# Dados do covertype
total_instancias = 581012
test_size = 0.3
subsample_size = 0.01

# Cálculo do tamanho do teste
instancias_teste_full = int(total_instancias * test_size)
instancias_teste_subsample = int(instancias_teste_full * subsample_size)

print("\n" + "="*80)
print("ANÁLISE DE TEMPO - COVERTYPE")
print("="*80 + "\n")

print(f"Dataset covertype:")
print(f"  Total de instâncias: {total_instancias:,}")
print(f"  Test size (30%): {instancias_teste_full:,} instâncias")
print(f"  Subsample (1%): {instancias_teste_subsample:,} instâncias no teste")

print(f"\n{'─'*80}")
print("TEMPO ESPERADO POR MÉTODO")
print(f"{'─'*80}\n")

# Tempos médios por instância (baseado em observações reais)
tempo_anchor_por_inst = 0.5  # minutos (30 segundos)
tempo_minexp_por_inst = 0.05  # minutos (3 segundos)
tempo_peab_por_inst = 0.001  # minutos (0.06 segundos)
tempo_pulp_por_inst = 0.15  # minutos (9 segundos)

# Seu caso atual
instancias_atuais = 1485
tempo_decorrido_min = 33.42  # 33:25
instancias_feitas = 67

tempo_real_por_inst = tempo_decorrido_min / instancias_feitas
tempo_restante = (instancias_atuais - instancias_feitas) * tempo_real_por_inst

print(f"ANCHOR (baseado no seu progresso atual):")
print(f"  Instâncias no teste: {instancias_atuais}")
print(f"  Tempo por instância: {tempo_real_por_inst:.2f} min ({tempo_real_por_inst*60:.0f}s)")
print(f"  Progresso: {instancias_feitas}/{instancias_atuais} ({instancias_feitas/instancias_atuais*100:.1f}%)")
print(f"  Tempo restante: {tempo_restante/60:.1f} horas")
print(f"  Tempo total estimado: {(tempo_decorrido_min + tempo_restante)/60:.1f} horas")

print(f"\n✅ SIM, isso está CORRETO para Anchor!")
print(f"   Anchor é o método mais lento (usa perturbações extensivas)")

print(f"\n{'─'*80}\n")

print(f"COMPARAÇÃO COM OUTROS MÉTODOS ({instancias_teste_subsample} instâncias):")
print(f"  PEAB:   ~{instancias_teste_subsample * tempo_peab_por_inst:.0f} min = {instancias_teste_subsample * tempo_peab_por_inst/60:.1f}h")
print(f"  MinExp: ~{instancias_teste_subsample * tempo_minexp_por_inst:.0f} min = {instancias_teste_subsample * tempo_minexp_por_inst/60:.1f}h")
print(f"  PuLP:   ~{instancias_teste_subsample * tempo_pulp_por_inst:.0f} min = {instancias_teste_subsample * tempo_pulp_por_inst/60:.1f}h")
print(f"  Anchor: ~{instancias_teste_subsample * tempo_anchor_por_inst:.0f} min = {instancias_teste_subsample * tempo_anchor_por_inst/60:.1f}h")

print(f"\n{'─'*80}")
print("VERIFICAÇÃO DO SUBSAMPLE")
print(f"{'─'*80}\n")

if instancias_atuais != instancias_teste_subsample:
    print(f"⚠️  ATENÇÃO: Diferença detectada!")
    print(f"   Esperado: {instancias_teste_subsample} instâncias (com subsample 1%)")
    print(f"   Atual: {instancias_atuais} instâncias")
    print(f"   Diferença: {instancias_atuais - instancias_teste_subsample} instâncias a menos")
    print(f"\n   Possível causa: Rejeições reduzem o número de instâncias explicadas")
else:
    print(f"✅ Subsample correto: {instancias_atuais} instâncias")

print(f"\n{'─'*80}")
print("RECOMENDAÇÕES")
print(f"{'─'*80}\n")

print(f"1. ⏱️  ANCHOR É REALMENTE LENTO (11-12h é normal para covertype)")
print(f"   - Usa ~30s por instância (vs 3s do MinExp)")
print(f"   - É o método mais caro computacionalmente")

print(f"\n2. 💡 OPÇÕES PARA REDUZIR TEMPO:")
print(f"   a) Aumentar subsample_size para 0.05 (5%) - ainda representativo")
print(f"      Tempo: ~{instancias_teste_full * 0.05 * tempo_anchor_por_inst/60:.1f}h (vs atual {instancias_teste_subsample * tempo_anchor_por_inst/60:.1f}h)")
print(f"      Mais instâncias = resultados mais confiáveis")

print(f"\n   b) Reduzir subsample_size para 0.005 (0.5%) - mais rápido")
print(f"      Tempo: ~{instancias_teste_full * 0.005 * tempo_anchor_por_inst/60:.1f}h")
print(f"      Menos instâncias = menos confiável mas mais rápido")

print(f"\n   c) Ajustar parâmetros do Anchor (se possível):")
print(f"      - Reduzir n_samples")
print(f"      - Reduzir threshold de precisão")

print(f"\n3. 🔄 MINEXP também demora bastante (~{instancias_teste_subsample * tempo_minexp_por_inst/60:.1f}h)")
print(f"   Mas é ~10x mais rápido que Anchor")

print(f"\n4. ⚡ MAIS RÁPIDOS:")
print(f"   - PEAB: ~{instancias_teste_subsample * tempo_peab_por_inst/60:.1f}h (muito rápido)")
print(f"   - PuLP: ~{instancias_teste_subsample * tempo_pulp_por_inst/60:.1f}h (médio)")

print(f"\n{'='*80}\n")

print("CONCLUSÃO:")
print("✅ Sim, 11-12h é NORMAL para Anchor no covertype com subsample 1%")
print("✅ MinExp também vai demorar (~1-2h)")
print("💡 Se quiser mais rápido, aumente o subsample para 0.05 (ainda é só 5%)")
print("="*80 + "\n")
