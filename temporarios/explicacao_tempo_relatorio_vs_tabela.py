print('=' * 80)
print('EXPLICAÇÃO DETALHADA: Tempo no Relatório vs Tempo na Tabela')
print('=' * 80)

print('''
🔍 ORIGEM DA CONFUSÃO:

Você está comparando dois formatos diferentes de apresentação:

1. RELATÓRIO (anchor_mnist_3_vs_8.txt):
   • Mostra tempos SEPARADOS por predição:
     - Positivas: 206.21 s
     - Negativas: 315.28 s
     - Rejeitadas: 300.59 s
   
   • É útil para análise detalhada (ver se há diferença entre pos/neg)

2. TABELA (mnist_runtime_unified.tex):
   • Mostra tempos COMBINADOS para comparação entre métodos:
     - Classificadas: 257.87 s (positivas + negativas juntas)
     - Rejeitadas: 300.59 s
   
   • É útil para comparar PEAB vs Anchor vs MinExp no mesmo formato


📊 COMO É CALCULADO NA TABELA:
''')

print('=' * 80)
print('CÓDIGO: gerar_tabelas_mnist.py (linhas 137-220)')
print('=' * 80)

print('''
def extrair_tempo_por_tipo_media_std_ms(data, metodo):
    """
    Extrai tempos dos JSONs, calculando média ponderada para classificadas.
    """
    
    # Para cada instância no JSON
    for pi in per_instance:
        tempo = pi.get("computation_time")  # em segundos
        
        if pi.get("rejected"):
            rej_s.append(tempo)           # Rejeitadas: lista separada
        else:
            classif_s.append(tempo)       # Classificadas: pos + neg juntas
    
    # Calcula média dos tempos combinados
    mean_class_ms = np.mean(classif_s) * 1000   # Converte s → ms
    mean_rej_ms = np.mean(rej_s) * 1000
    
    return (mean_class_ms, std_class_ms), (mean_rej_ms, std_rej_ms)


🎯 RESULTADO:
   • Classificadas = média de TODAS as instâncias aceitas (pos + neg)
   • Rejeitadas = média das instâncias rejeitadas
''')

print('\n' + '=' * 80)
print('VERIFICAÇÃO MATEMÁTICA: MNIST Anchor')
print('=' * 80)

print('''
DADOS DO JSON (per_instance):
  • 20 positivas com tempos individuais → média = 206.21 s
  • 18 negativas com tempos individuais → média = 315.28 s
  • 3 rejeitadas com tempos individuais → média = 300.59 s

CÁLCULO DA TABELA (média ponderada das classificadas):

  classif_mean = (Σ tempos_pos + Σ tempos_neg) / (n_pos + n_neg)
               = (206.21×20 + 315.28×18) / (20 + 18)
               = (4124.12 + 5675.00) / 38
               = 9799.13 / 38
               = 257.87 segundos
               = 257871.79 milissegundos ✓

CÁLCULO DA TABELA (rejeitadas):

  rej_mean = Σ tempos_rej / n_rej
           = 300.59 segundos
           = 300590.33 milissegundos ✓
''')

print('\n' + '=' * 80)
print('POR QUE COMBINAR POSITIVAS + NEGATIVAS?')
print('=' * 80)

print('''
✅ RAZÕES PARA COMBINAR:

1. COMPARABILIDADE entre métodos
   • Todos os métodos (PEAB, Anchor, MinExp) seguem o mesmo formato
   • Fácil ver: "Quanto tempo leva para explicar instâncias aceitas?"

2. COMPACIDADE da tabela
   • 2 colunas por método (Clas. + Rej.)
   • Se separássemos pos/neg: 3 colunas por método (Pos. + Neg. + Rej.)
   • Tabela ficaria muito larga (9 colunas vs. 6 colunas)

3. FOCO CORRETO da análise
   • A decisão de rejeitar é baseada no SCORE, não na classe
   • O que importa: "explicar aceitas" vs "explicar rejeitadas"
   • A diferença entre pos/neg dentro das aceitas é secundária

4. PADRÃO CIENTÍFICO
   • Papers normalmente reportam separado por decisão (aceitar/rejeitar)
   • Análise pos/neg vai em seções específicas de "Class imbalance"


📝 PARA O SEU ARTIGO:

Se quiser mencionar a diferença entre positivas/negativas:
  
  "Para instâncias classificadas, o Anchor levou em média 257.87 ms,
   sendo 206.21 ms para positivas e 315.28 ms para negativas."

Mas na TABELA PRINCIPAL, mantenha o formato combinado para clareza.
''')

print('\n' + '=' * 80)
print('CONCLUSÃO FINAL')
print('=' * 80)

print('''
✅ OS VALORES ESTÃO CORRETOS!

  • Relatório: Mostra detalhes (pos/neg separados)
  • Tabela: Mostra síntese comparativa (pos+neg combinados)

  Ambos são verdadeiros, apenas apresentam granularidades diferentes.

🎯 NÃO PRECISA MUDAR NADA!
   Este é o formato correto para tabelas comparativas em artigos.
''')

print('=' * 80)
