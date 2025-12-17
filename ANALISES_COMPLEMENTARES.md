# Scripts de Análise Complementar - PEAB

Este diretório contém scripts para análises complementares dos experimentos do PEAB.

## 📊 Scripts Disponíveis

### 1. `peab_wilcoxon.py` - Testes Estatísticos de Significância

**Objetivo**: Verificar se diferenças entre PEAB e baselines são estatisticamente significativas.

**Testes Implementados**:
- ✅ Teste de Wilcoxon (não-paramétrico)
- ✅ Teste t-pareado (paramétrico)
- ✅ Tamanho de Efeito (Cohen's d)
- ✅ Correção de Bonferroni (múltiplas comparações)

**Como Executar**:
```bash
python peab_wilcoxon.py
```

**Saídas Geradas**:
- `results/statistical_tests/wilcoxon_test_report.txt` - Relatório completo
- `results/statistical_tests/wilcoxon_results.json` - Resultados em JSON
- `results/statistical_tests/comparison_table.tex` - Tabela LaTeX para paper

**Interpretação**:
- **p < 0.05**: Diferença estatisticamente significativa ✓
- **p ≥ 0.05**: Diferença não significativa ✗
- **Cohen's d**:
  - |d| < 0.2: Efeito pequeno
  - 0.2 ≤ |d| < 0.5: Efeito médio
  - 0.5 ≤ |d| < 0.8: Efeito grande
  - |d| ≥ 0.8: Efeito muito grande

---

### 2. `peab_metricaExtra.py` - Métricas Extras de Explicabilidade

**Objetivo**: Avaliar qualidade das explicações além da minimalidade (tamanho).

**Métricas Implementadas**:

1. **Consistência** (0-1)
   - Mede se instâncias similares têm explicações similares
   - Usa Índice de Jaccard
   - Alto = Explicações consistentes ✓

2. **Cobertura de Features** (%)
   - Quantas features diferentes são usadas
   - Alta entropia = Boa diversidade
   - Identifica features mais frequentes

3. **Estabilidade** (CV)
   - Variância do tamanho das explicações
   - CV < 0.3 = Estável ✓
   - CV > 0.5 = Instável ✗

4. **Tempo Computacional** (segundos)
   - Eficiência do método
   - Tempo por instância
   - Separado por tipo (positiva/negativa/rejeitada)

5. **Taxa de Features Únicas** (%)
   - Quão específicas são as explicações
   - Alto = Explicações específicas
   - Baixo = Explicações genéricas

**Como Executar**:
```bash
python peab_metricaExtra.py
```

**Saídas Geradas**:
- `results/extra_metrics/extra_metrics_{dataset}.txt` - Um relatório por dataset
- Rankings comparativos entre métodos
- Análise detalhada de cada métrica

---

## 🚀 Fluxo de Uso Recomendado

### Passo 1: Execute os Experimentos
```bash
# Execute PEAB e baselines primeiro
python peab.py
python minexp.py
python anchor.py
python pulp_experiment.py
```

### Passo 2: Testes Estatísticos
```bash
# Verifica se diferenças são significativas
python peab_wilcoxon.py
```

**O que esperar**:
- Comparações PEAB vs MinExp, Anchor, PULP
- P-values indicando significância
- Tamanho de efeito (magnitude da diferença)

### Passo 3: Métricas Extras
```bash
# Avalia qualidade das explicações
python peab_metricaExtra.py
```

**O que esperar**:
- Análise multidimensional da qualidade
- Rankings por métrica
- Identificação de trade-offs

---

## 📈 Exemplo de Interpretação

### Cenário 1: PEAB Vence Claramente ✓
```
Wilcoxon Test:
  PEAB vs MinExp: p = 0.003, d = -0.82 (muito grande)
  → PEAB é significativamente menor (p < 0.05)

Métricas Extras:
  - Consistência: PEAB = 0.75, MinExp = 0.68 → PEAB mais consistente
  - Estabilidade: PEAB CV = 0.25, MinExp CV = 0.42 → PEAB mais estável
  - Tempo: PEAB = 0.05s, MinExp = 0.15s → PEAB mais rápido
  
Conclusão: PEAB é superior em todas as dimensões!
```

### Cenário 2: Trade-offs 🤔
```
Wilcoxon Test:
  PEAB vs Anchor: p = 0.023, d = -0.35 (médio)
  → PEAB é significativamente menor, mas efeito moderado

Métricas Extras:
  - Consistência: PEAB = 0.72, Anchor = 0.81 → Anchor mais consistente!
  - Estabilidade: PEAB CV = 0.28, Anchor CV = 0.19 → Anchor mais estável!
  - Tempo: PEAB = 0.05s, Anchor = 0.35s → PEAB 7x mais rápido!
  
Conclusão: PEAB é menor e mais rápido, mas Anchor é mais consistente.
Trade-off válido dependendo da aplicação.
```

### Cenário 3: Não Significativo ✗
```
Wilcoxon Test:
  PEAB vs PULP: p = 0.156, d = -0.18 (pequeno)
  → Diferença NÃO é significativa (p ≥ 0.05)

Métricas Extras:
  - Tamanho similar
  - Consistência similar
  - Tempo: PEAB muito mais rápido
  
Conclusão: Métodos comparáveis em qualidade, mas PEAB é mais eficiente.
```

---

## 📝 Para o Paper

### O que Reportar:

**Obrigatório**:
1. ✅ Tamanho médio das explicações (minimalidade)
2. ✅ P-values dos testes de Wilcoxon
3. ✅ Tamanho de efeito (Cohen's d)

**Recomendado**:
4. ⚠️ Tempo computacional
5. ⚠️ Consistência e Estabilidade

**Opcional**:
6. ℹ️ Cobertura de features
7. ℹ️ Taxa de features únicas

### Tabela Sugerida para Paper:

```latex
\begin{table}[h]
\centering
\caption{Comparação Estatística - PEAB vs Baselines}
\begin{tabular}{lcccccc}
\hline
Método & Tamanho & Tempo (s) & Wilcoxon p & Cohen's d & Sig. \\
\hline
PEAB    & 12.3±2.1 & 0.05 & -         & -         & -   \\
MinExp  & 15.7±3.4 & 0.12 & 0.003     & -0.82     & ✓   \\
Anchor  & 14.1±1.8 & 0.35 & 0.023     & -0.35     & ✓   \\
PULP    & 13.2±2.9 & 0.08 & 0.156     & -0.18     & ✗   \\
\hline
\end{tabular}
\end{table}
```

---

## 🔧 Troubleshooting

### Erro: "Arquivo não encontrado"
- Certifique-se de ter executado os experimentos antes
- Verifique se os arquivos `*_results.json` existem em `json/`

### Erro: "Nenhuma comparação possível"
- Verifique se há datasets comuns entre os métodos
- Pode ser que alguns métodos não tenham sido executados

### Resultados Estranhos
- Verifique se os experimentos foram executados com as mesmas configurações
- Seeds diferentes podem causar variações

---

## 📚 Referências

1. **Wilcoxon (1945)**: "Individual Comparisons by Ranking Methods"
2. **Demšar (2006)**: "Statistical Comparisons of Classifiers over Multiple Data Sets"
3. **Cohen (1988)**: "Statistical Power Analysis for the Behavioral Sciences"
4. **Ribeiro et al. (2016)**: "Why Should I Trust You?" - LIME
5. **Lundberg & Lee (2017)**: "A Unified Approach to Interpreting Model Predictions" - SHAP

---

## ⚠️ Limitações

**Problema 4 (Múltiplas Seeds)**: NÃO IMPLEMENTADO
- Motivo: Aumentaria 5-10x o tempo computacional
- Alternativa: Resultados com seed=42 são reportados, mas reconhecemos a limitação

**Problema 7 (Ablation Study)**: NÃO IMPLEMENTADO
- Motivo: Foco é comparação com baselines, não análise interna do PEAB
- Para fazer: Requer implementar variantes do método (sem reforço, sem minimização, etc.)

---

**Dúvidas?** Consulte a documentação inline nos scripts ou o código fonte.
