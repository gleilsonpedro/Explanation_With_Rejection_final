print('=' * 80)
print('EXPLICAÇÃO: Std=0 LEGÍTIMO vs Std=0 BUG')
print('=' * 80)

print('''
🔍 DIFERENÇA CRUCIAL:


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ STD = 0 LEGÍTIMO (Breast Cancer Rejeitadas)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Contexto:
  • PEAB Rejeitadas: 2.00 ± 0.00 (34 instâncias)
  • Anchor Rejeitadas: 2.00 ± 0.00 (34 instâncias)
  • MinExp Rejeitadas: 2.00 ± 0.00 (34 instâncias)

Por que é CORRETO?
  • Verificamos os dados: TODAS as 34 rejeitadas têm EXATAMENTE 2 features
  • Não há variação REAL nos dados
  • Std=0 reflete a realidade: tamanhos idênticos

Interpretação:
  • As instâncias rejeitadas no Breast Cancer são MUITO HOMOGÊNEAS
  • Todas precisam das mesmas 2 features para serem explicadas
  • É uma característica interessante do dataset!


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ❌ STD = 0 BUG (Casos Que Corrigimos Hoje)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. TABELA DE TEMPO - Anchor Covertype (ANTES):
   • Classificadas: 34522.68 ± 0.00 ms (742 instâncias)
   • Rejeitadas: 67311.90 ± 0.00 ms (83 instâncias)
   
   Por que era BUG?
     • Tempos de execução NUNCA são idênticos
     • JSON tinha computation_time = 0.0 para TODAS
     • Era um erro de salvamento (JSON antigo)
   
   Correção:
     • Executar Anchor novamente
     • Classificadas: 34569.58 ± 30987.04 ms ✓
     • Rejeitadas: 67035.24 ± 48883.12 ms ✓


2. TABELA DE TEMPO - MinExp Credit Card Rejeitadas (ANTES):
   • Rejeitadas: 1929.72 ± 0.00 ms (9 instâncias)
   
   Por que era BUG?
     • JSON tinha 9 tempos IDÊNTICOS: 1.9297181606292725
     • Bug do chunk processing (distribuía tempo igualmente)
     • Código estava ERRADO
   
   Correção:
     • Código MinExp já corrigido (instância-por-instância)
     • Executar novamente: 1678.53 ± 371.69 ms ✓


3. TABELA DE EXPLICAÇÕES - Banknote Anchor (ANTES):
   • Classificadas: 1.37 ± 0.00 (170 instâncias)
   
   Por que era BUG?
     • Dados tinham variação REAL (21 pos com 4 features, 149 neg com 1 feature)
     • Fórmula pooled std estava ERRADA
     • Código calculava std=0 quando deveria ser std=0.99
   
   Correção:
     • Corrigir fórmula em gerar_tabelas_mnist.py
     • Agora: 1.37 ± 0.99 ✓


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📊 RESUMO: QUANDO STD=0 É OK?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ STD=0 É LEGÍTIMO quando:
   • TAMANHOS DE EXPLICAÇÕES são todos iguais
   • Casos:
     - Breast Cancer Rejeitadas: todas com 2 features
     - Pima Indians Rejeitadas: todas com 8 features (máximo possível)
     - MNIST Anchor Rejeitadas: todas com 0 features (explicações vazias)
   
   • É uma característica REAL dos dados
   • Indica homogeneidade nas explicações


❌ STD=0 É BUG quando:
   • TEMPOS DE EXECUÇÃO são todos iguais
     → Tempos DEVEM variar (mesmo que pouco)
   
   • CÓDIGO calcula errado
     → Pooled std com fórmula simplificada
     → Chunk processing dividindo tempo igualmente
   
   • JSON tem valores zerados/inválidos
     → computation_time = 0.0 para todos


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎯 PARA O SEU ARTIGO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Você pode comentar no texto:

  "Alguns datasets apresentam desvio padrão zero em certas células,
   indicando explicações de tamanho homogêneo. Por exemplo, todas as
   34 instâncias rejeitadas do Breast Cancer são explicadas com
   exatamente 2 features em todos os três métodos, demonstrando alta
   consistência no padrão de rejeição deste dataset."

Isso mostra que você ENTENDE os dados e não é um erro!


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ CONCLUSÃO FINAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Breast Cancer Rejeitadas com 2.00 ± 0.00:
  • ✅ CORRETO!
  • ✅ Verificado nos 3 JSONs
  • ✅ Todas as 34 instâncias têm exatamente 2 features
  • ✅ É uma característica real, não um bug

Não precisa mudar nada!
''')

print('=' * 80)
