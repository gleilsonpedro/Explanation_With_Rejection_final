# PuLP Experiment - Solver de Otimização Inteira

## 📋 Visão Geral

O `pulp_experiment.py` implementa um solver de **otimização inteira** usando a biblioteca PuLP (CBC solver) para calcular explicações **matematicamente ótimas** (cardinalidade mínima).

Este método serve como **GROUND TRUTH** (baseline) para avaliar a qualidade das heurísticas (PEAB, Anchor, MinExp).

---

## 🎯 Objetivo Acadêmico

### Por que PuLP?
- ✅ **Garante solução ÓTIMA** (menor número de features possível)
- ✅ **Baseline científico** para calcular GAP das heurísticas
- ✅ **Rigor matemático** para publicações acadêmicas
- ⚠️ **Trade-off**: Lento mas preciso

### Comparação:
| Método | Velocidade | Qualidade | Uso |
|--------|-----------|-----------|-----|
| **PuLP** | 🐌 Lento | ⭐⭐⭐⭐⭐ Ótimo | Benchmark offline |
| **PEAB** | 🚀 Rápido | ⭐⭐⭐⭐ Muito bom | Produção |
| **Anchor** | 🐢 Médio | ⭐⭐⭐ Bom | Explicações globais |
| **MinExp** | 🐌 Lento | ⭐⭐⭐ Bom | Explicações locais |

---

## 📁 Estrutura de Saída

### JSON (formato consistente com outros métodos):
```
json/
└── pulp_results.json
    └── {dataset_name}
        ├── dataset: "nome"
        ├── metodo: "pulp"
        ├── num_instancias: 150
        ├── params: {...}
        ├── t_plus: 0.5657
        ├── t_minus: -0.5000
        ├── rejection_cost: 0.24
        ├── metricas_modelo: {...}
        ├── estatisticas_gerais: {...}
        ├── estatisticas_por_tipo: {...}
        └── explicacoes: [
            {
                "indice": 0,
                "classe_real": "Classe1",
                "tipo_predicao": "POSITIVA",
                "features_selecionadas": ["feat1", "feat3", "feat5"],
                "tamanho": 3,
                "tempo_segundos": 0.1234
            },
            ...
        ]
```

### Relatórios TXT:
```
results/
└── report/
    └── pulp/
        └── {dataset_name}/
            └── R_pulp_{dataset_name}.txt
```

---

## 🚀 Como Usar

### 1. Execução Direta:
```bash
python pulp_experiment.py
```

### 2. Via Menu (menu será criado):
```python
# No futuro: main.py
# [4] Executar PuLP (solver exato)
```

### 3. Programático:
```python
from pulp_experiment import executar_experimento_pulp
executar_experimento_pulp()
```

---

## 📊 Exemplo de Saída

```
================================================================================
   PULP EXPERIMENT - Solver de Otimização Inteira (Ground Truth)
================================================================================

🎯 Dataset selecionado: wine
⚠️  AVISO: PuLP é lento mas garante soluções ÓTIMAS.

📊 Hiperparâmetros utilizados:
{
  "penalty": "l2",
  "C": 0.01,
  "solver": "liblinear",
  "max_iter": 1000
}
💰 Rejection cost: 0.24
🔀 Test size: 0.3

🔧 Treinando modelo...
✅ Thresholds: t+ = 0.5657, t- = -0.5000
📏 Zona de rejeição: 1.0657

🔬 Processando 39 instâncias de teste...
[████████████████████████████████████████] 39/39 (100%)

✅ JSON salvo: json/pulp_results.json
✅ Relatório salvo: results/report/pulp/wine/R_pulp_wine.txt

================================================================================
📊 RESUMO DO EXPERIMENTO
================================================================================
Dataset: wine
Instâncias processadas: 39
Tamanho médio: 4.23 features
Tempo total: 12.45s
Tempo médio/instância: 0.3192s

Distribuição por tipo:
  POSITIVA  :   39 (100.0%) - Tam. médio: 4.23
================================================================================
```

---

## 🔬 Formulação Matemática

### Problema de Otimização:
```
Minimizar: Σ z_i  (cardinalidade)

Sujeito a:
- z_i ∈ {0, 1}  (binário: feature i está na explicação?)
- score_worst ≥ t+  (para predições POSITIVAS)
- score_worst ≤ t-  (para predições NEGATIVAS)
- t- ≤ score_worst ≤ t+  (para REJEIÇÕES)
```

### Onde:
- `score_worst = intercept + Σ(z_i × contribuição_i)`
- `contribuição_i` considera pior cenário (adversarial)

---

## 🔗 Integração com Outros Métodos

### PEAB vs PuLP:
```python
# O arquivo benchmark_peab.py compara:
GAP = tamanho_PEAB - tamanho_PuLP
Taxa_Otimalidade = % (GAP == 0)
Speedup = tempo_PuLP / tempo_PEAB
```

### Comparação Múltipla:
```python
# Futuro: compare_all_methods.py
# Compara PEAB vs Anchor vs MinExp vs PuLP
# Calcula GAP de cada heurística vs ground truth
```

---

## ⚙️ Configurações

### Dependências:
```bash
pip install pulp
```

### Solver Backend:
- **Padrão**: CBC (open-source)
- **Opcional**: Gurobi, CPLEX (acadêmico, mais rápido)

### Performance:
- **Wine (39 instâncias)**: ~10-15s
- **MNIST (3000 instâncias)**: ~30-60min ⚠️
- **Spambase (1382 instâncias)**: ~10-20min

**Recomendação**: Execute PuLP UMA VEZ por dataset e cache os resultados.

---

## 📝 Notas de Implementação

### Consistência com PEAB:
1. ✅ Usa mesmos thresholds (t+, t-)
2. ✅ Usa mesmo split (RANDOM_STATE=42)
3. ✅ Usa mesma normalização (MinMaxScaler)
4. ✅ Usa mesmo modelo treinado

### Diferenças vs benchmark_peab.py:
| Aspecto | benchmark_peab.py | pulp_experiment.py |
|---------|-------------------|-------------------|
| **Objetivo** | Comparar PEAB vs PuLP | Gerar ground truth |
| **Saída** | CSV + TXT de comparação | JSON + TXT individual |
| **Executa** | PEAB + PuLP juntos | Apenas PuLP |
| **Uso** | Análise de GAP | Baseline independente |

---

## 🎓 Uso Acadêmico

### Para sua Dissertação:
1. **Capítulo de Metodologia**:
   - "PuLP foi usado como baseline para validar qualidade do PEAB"
   
2. **Tabelas Comparativas**:
   ```latex
   \begin{table}
   \caption{Comparação PEAB vs Solver Ótimo (PuLP)}
   \begin{tabular}{l|cc|c}
   Dataset & PEAB & PuLP & GAP \\
   \hline
   Wine    & 4.5  & 4.2  & 0.3 \\
   ...
   \end{tabular}
   \end{table}
   ```

3. **Análise de Trade-off**:
   - "PEAB obtém 95% de otimalidade com speedup de 50x"

---

## 🐛 Troubleshooting

### Problema: "No module named 'pulp'"
```bash
pip install pulp
```

### Problema: Muito lento
- ✅ Normal para datasets grandes
- ✅ Execute em background overnight
- ✅ Use subsample para testes rápidos

### Problema: Solver não encontra solução
- Verifique se thresholds são válidos (t- < t+)
- Verifique se modelo está treinado corretamente

---

## 📖 Referências

1. **PuLP Documentation**: https://coin-or.github.io/pulp/
2. **CBC Solver**: https://github.com/coin-or/Cbc
3. **Integer Programming**: Wolsey, L.A. (1998). *Integer Programming*

---

## 🔄 Próximos Passos

1. ✅ **CONCLUÍDO**: Criar `pulp_experiment.py`
2. ⏳ **PRÓXIMO**: Criar `peab_vs_pulp.py` (análise comparativa)
3. ⏳ **FUTURO**: Criar `main.py` (menu unificado)
4. ⏳ **FUTURO**: Criar pasta `experiments/` (organização)

---

**Autor**: Gleilson Pedro  
**Data**: 09/12/2025  
**Versão**: 1.0
