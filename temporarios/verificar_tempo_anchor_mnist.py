print('=' * 80)
print('VERIFICAÇÃO: Tempo Anchor MNIST - Relatório vs Tabela')
print('=' * 80)

# DADOS DO RELATÓRIO
print('\n📄 RELATÓRIO (anchor_mnist_3_vs_8.txt):')
print('-' * 80)
tempo_pos = 206.2062  # segundos
tempo_neg = 315.2780  # segundos
tempo_rej = 300.5903  # segundos

n_pos = 20
n_neg = 18
n_rej = 3

print(f'  Positivas Aceitas: {tempo_pos:.4f} segundos ({n_pos} instâncias)')
print(f'  Negativas Aceitas: {tempo_neg:.4f} segundos ({n_neg} instâncias)')
print(f'  Rejeitadas:        {tempo_rej:.4f} segundos ({n_rej} instâncias)')

# CÁLCULO DA TABELA
print('\n\n📊 TABELA (mnist_runtime_unified.tex):')
print('-' * 80)

# A tabela combina positivas + negativas em "Classificadas"
tempo_classif_ms = 257871.79  # milissegundos
tempo_rej_ms = 300590.33      # milissegundos

print(f'  Classificadas: {tempo_classif_ms:.2f} ms = {tempo_classif_ms/1000:.4f} segundos')
print(f'  Rejeitadas:    {tempo_rej_ms:.2f} ms = {tempo_rej_ms/1000:.4f} segundos')

# VERIFICAÇÃO: Classificadas deve ser a média ponderada
print('\n\n✅ VERIFICAÇÃO - CLASSIFICADAS (Positivas + Negativas):')
print('-' * 80)

tempo_classif_calculado = (tempo_pos * n_pos + tempo_neg * n_neg) / (n_pos + n_neg)

print(f'\nFórmula: (tempo_pos × n_pos + tempo_neg × n_neg) / (n_pos + n_neg)')
print(f'       = ({tempo_pos} × {n_pos} + {tempo_neg} × {n_neg}) / ({n_pos} + {n_neg})')
print(f'       = ({tempo_pos * n_pos:.2f} + {tempo_neg * n_neg:.2f}) / {n_pos + n_neg}')
print(f'       = {tempo_pos * n_pos + tempo_neg * n_neg:.2f} / {n_pos + n_neg}')
print(f'       = {tempo_classif_calculado:.4f} segundos')
print(f'       = {tempo_classif_calculado * 1000:.2f} milissegundos')

diferenca_classif = abs(tempo_classif_ms - tempo_classif_calculado * 1000)

print(f'\nComparação:')
print(f'  Tabela:    {tempo_classif_ms:.2f} ms')
print(f'  Calculado: {tempo_classif_calculado * 1000:.2f} ms')
print(f'  Diferença: {diferenca_classif:.4f} ms')

if diferenca_classif < 0.01:
    print(f'  ✅ CORRETO: Valores batem perfeitamente!')
else:
    print(f'  ⚠️ ATENÇÃO: Diferença de {diferenca_classif:.2f} ms')

# VERIFICAÇÃO: Rejeitadas
print('\n\n✅ VERIFICAÇÃO - REJEITADAS:')
print('-' * 80)

diferenca_rej = abs(tempo_rej_ms - tempo_rej * 1000)

print(f'\nComparação:')
print(f'  Tabela:     {tempo_rej_ms:.2f} ms')
print(f'  Relatório:  {tempo_rej * 1000:.2f} ms')
print(f'  Diferença:  {diferenca_rej:.4f} ms')

if diferenca_rej < 0.01:
    print(f'  ✅ CORRETO: Valores batem perfeitamente!')
else:
    print(f'  ⚠️ ATENÇÃO: Diferença de {diferenca_rej:.2f} ms')

# CONCLUSÃO
print('\n\n' + '=' * 80)
print('CONCLUSÃO:')
print('=' * 80)

print('''
✅ OS VALORES ESTÃO CORRETOS!

A diferença que você notou é porque:

1. RELATÓRIO mostra tempos SEPARADOS:
   • Positivas Aceitas: 206.21 s (20 instâncias)
   • Negativas Aceitas: 315.28 s (18 instâncias)
   • Rejeitadas: 300.59 s (3 instâncias)

2. TABELA mostra tempos COMBINADOS:
   • Classificadas = média ponderada de positivas + negativas
     → (206.21×20 + 315.28×18) / (20+18) = 257.87 s ✓
   • Rejeitadas = mantém o valor original
     → 300.59 s ✓

🎯 Por que combinar positivas + negativas?

Na tabela comparativa, queremos comparar:
  • Tempo para explicar instâncias ACEITAS (classificadas)
  • Tempo para explicar instâncias REJEITADAS

Separar positivas/negativas ocuparia 2 colunas extras e não é o foco
da comparação entre métodos.

📊 Este é o procedimento CORRETO para tabelas comparativas!
   Cada método (PEAB, Anchor, MinExp) usa a mesma lógica:
   - Classificadas = média ponderada de pos + neg
   - Rejeitadas = valor direto
''')

print('=' * 80)
