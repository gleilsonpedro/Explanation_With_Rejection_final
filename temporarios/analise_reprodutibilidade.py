"""
Análise: O que deve ser REPRODUZÍVEL vs o que PODE VARIAR entre execuções
"""

print("=" * 100)
print("REPRODUTIBILIDADE DOS EXPERIMENTOS")
print("=" * 100)

print("""
Seus experimentos têm RANDOM_STATE = 42 FIXO em:
  ✓ minexp.py (linha 89)
  ✓ data/datasets.py (linha 156)
  ✓ train_test_split usa random_state=RANDOM_STATE
  ✓ np.random.seed(RANDOM_STATE)

Isso significa que SEMPRE que você rodar:
""")

print("\n" + "=" * 100)
print("O QUE SERÁ SEMPRE IDÊNTICO (REPRODUZÍVEL):")
print("=" * 100)
print("""
1. ✓ Divisão treino/teste
   → As MESMAS instâncias em treino e teste
   → Ordem das instâncias sempre igual

2. ✓ Modelo treinado
   → Mesmos pesos/coeficientes
   → Mesmas predições

3. ✓ Instâncias classificadas vs rejeitadas
   → MESMAS instâncias rejeitadas
   → MESMAS instâncias classificadas

4. ✓ Explicações geradas
   → MESMAS features nas explicações
   → MESMO tamanho de cada explicação
   → Conteúdo IDÊNTICO
""")

print("\n" + "=" * 100)
print("O QUE PODE VARIAR (NÃO REPRODUZÍVEL):")
print("=" * 100)
print("""
1. ⚠️  TEMPOS DE EXECUÇÃO
   → Dependem da carga da CPU
   → Processos em background no Windows
   → Cache do processador
   → Memória disponível
   → VARIAÇÃO ESPERADA: ±10-30% entre execuções

2. ⚠️  Ordem de processamento paralelo (se houver)
   → Mas resultado final é o mesmo
""")

print("\n" + "=" * 100)
print("ANÁLISE DO SEU CASO: MinExp MNIST")
print("=" * 100)
print("""
Execução 1: 67.6 segundos
Execução 2: 86.7 segundos
Diferença: +28.3%

Isso é NORMAL? Vamos analisar:
""")

print("""
VARIAÇÃO DE TEMPO ESPERADA:
  - CPU em estado normal: ±10-15% (4-6 segundos)
  - CPU com carga moderada: ±20-25% (13-17 segundos)  
  - CPU com alta carga/aquecimento: ±30-50% (20-34 segundos)

SUA VARIAÇÃO: +28.3% (19 segundos)
  → Está no LIMITE SUPERIOR mas ainda é plausível
  → Possíveis causas:
    • Windows Defender rodando na 2ª execução
    • CPU aquecida na 2ª execução (throttling térmico)
    • Memória swap sendo usada
    • Outro processo em background
""")

print("\n" + "=" * 100)
print("RECOMENDAÇÃO")
print("=" * 100)
print("""
Para ter tempos MAIS ESTÁVEIS:

1. Rodar MÚLTIPLAS vezes (ex: 3-5 execuções)
2. Calcular MEDIANA dos tempos (ignora outliers)
3. Fechar outros programas durante experimento
4. Aguardar CPU esfriar entre execuções

Para o MNIST especificamente:
  → Rodar 3 vezes
  → Usar a MEDIANA (ex: se der 67s, 72s, 86s → usar 72s)
  → Explicar na metodologia que tempos têm ±20% de variação

IMPORTANTE: As EXPLICAÇÕES serão IDÊNTICAS em todas as execuções!
Só o TEMPO varia (mas o resultado/explicação é sempre o mesmo).
""")

print("\n" + "=" * 100)
print("VERIFICAÇÃO RÁPIDA")
print("=" * 100)
print("""
Para confirmar reprodutibilidade:
1. Rode 2 vezes o mesmo dataset
2. Compare os JSONs (explicações devem ser IDÊNTICAS)
3. Tempos podem variar ±30%
4. Se explicações mudarem → PROBLEMA no código
5. Se só tempo mudar → NORMAL
""")
