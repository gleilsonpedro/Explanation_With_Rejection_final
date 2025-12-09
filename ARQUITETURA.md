# 🏗️ ARQUITETURA DO PROJETO - Estrutura Modular

```
Explanation_With_Rejection_final/
│
├── 📁 experiments/                    ← NOVA ESTRUTURA (FASE 3)
│   ├── peab_experiment.py             ← Experimento PEAB
│   ├── anchor_experiment.py           ← Experimento Anchor
│   ├── minexp_experiment.py           ← Experimento MinExp
│   ├── pulp_experiment.py             ← Experimento PuLP ✅ CRIADO
│   │
│   ├── peab_vs_pulp.py                ← Comparação PEAB vs PuLP (FASE 2)
│   ├── compare_all_methods.py         ← Comparação geral (FASE 5)
│   │
│   ├── main.py                        ← Menu unificado (FASE 4)
│   └── README.md                      ← Documentação da pasta
│
├── 📁 json/                           ← Resultados estruturados
│   ├── peab_results.json              ← Resultados PEAB
│   ├── anchor_results.json            ← Resultados Anchor
│   ├── minexp_results.json            ← Resultados MinExp
│   ├── pulp_results.json              ← Resultados PuLP ✅ FORMATO DEFINIDO
│   ├── comparative_results.json       ← Comparação geral (legacy)
│   └── hiperparametros.json           ← Configurações
│
├── 📁 results/                        ← Relatórios e análises
│   ├── report/
│   │   ├── peab/                      ← Relatórios PEAB
│   │   ├── anchor/                    ← Relatórios Anchor
│   │   ├── minexp/                    ← Relatórios MinExp
│   │   └── pulp/                      ← Relatórios PuLP ✅ ESTRUTURA DEFINIDA
│   │
│   └── benchmark/
│       ├── peab_vs_pulp/              ← Comparação específica (FASE 2)
│       └── all_methods/               ← Comparação geral (FASE 5)
│
├── 📁 data/                           ← Datasets e loaders
├── 📁 utils/                          ← Funções auxiliares
├── 📁 docs/                           ← Documentação
│
├── 📄 PULP_README.md                  ← Documentação PuLP ✅ CRIADO
├── 📄 CHECKLIST_REESTRUTURACAO.md     ← Checklist de tarefas ✅ CRIADO
├── 📄 ARQUITETURA.md                  ← Este arquivo ✅
└── 📄 README.md                       ← Documentação principal
```

---

## 🔄 FLUXO DE EXECUÇÃO

### 1️⃣ Execução Individual de Métodos
```
┌─────────────────┐
│ peab.py         │ → json/peab_results.json    → results/report/peab/
├─────────────────┤
│ anchor.py       │ → json/anchor_results.json  → results/report/anchor/
├─────────────────┤
│ minexp.py       │ → json/minexp_results.json  → results/report/minexp/
├─────────────────┤
│ pulp_experiment │ → json/pulp_results.json    → results/report/pulp/ ✅
└─────────────────┘
```

### 2️⃣ Comparações (Reutiliza JSONs)
```
┌────────────────────────────┐
│  peab_vs_pulp.py           │
├────────────────────────────┤
│  ← json/peab_results.json  │
│  ← json/pulp_results.json  │
│  ↓                          │
│  results/benchmark/        │
│    peab_vs_pulp/           │
│      ├── relatorio.txt     │
│      ├── comparacao.csv    │
│      └── graficos/         │
└────────────────────────────┘

┌────────────────────────────┐
│  compare_all_methods.py    │
├────────────────────────────┤
│  ← json/peab_results.json  │
│  ← json/anchor_results.json│
│  ← json/minexp_results.json│
│  ← json/pulp_results.json  │
│  ↓                          │
│  results/benchmark/        │
│    all_methods/            │
│      ├── tabela_latex.tex  │
│      ├── metricas.csv      │
│      └── graficos/         │
└────────────────────────────┘
```

### 3️⃣ Menu Unificado
```
┌──────────────────────────────────────┐
│         main.py (MENU)               │
├──────────────────────────────────────┤
│ [1] Executar PEAB                    │ → peab_experiment.py
│ [2] Executar Anchor                  │ → anchor_experiment.py
│ [3] Executar MinExp                  │ → minexp_experiment.py
│ [4] Executar PuLP                    │ → pulp_experiment.py ✅
│ [5] ────────────────────────         │
│ [6] Comparar PEAB vs PuLP            │ → peab_vs_pulp.py
│ [7] Comparar PEAB vs Anchor vs MinExp│ → (existente)
│ [8] Comparar TODOS (inclui PuLP)     │ → compare_all_methods.py
│ [9] ────────────────────────         │
│ [10] Gerar Relatório Completo        │ → gera tudo
│ [0] Sair                             │
└──────────────────────────────────────┘
```

---

## 📊 FORMATO DOS DADOS

### Estrutura JSON Padronizada
```json
{
  "dataset_name": {
    "dataset": "wine",
    "metodo": "pulp",  // ou "peab", "anchor", "minexp"
    "num_instancias": 39,
    "params": {
      "C": 0.01,
      "penalty": "l2",
      "solver": "liblinear"
    },
    "t_plus": 0.5657,
    "t_minus": -0.5000,
    "rejection_cost": 0.24,
    "metricas_modelo": {
      "acuracia_sem_rejeicao": 0.95,
      "acuracia_com_rejeicao": 1.0,
      "taxa_rejeicao": 0.0,
      "risco_empirico": 0.05
    },
    "estatisticas_gerais": {
      "tamanho_medio": 4.23,
      "tempo_total_segundos": 12.45,
      "tempo_medio_segundos": 0.3192
    },
    "estatisticas_por_tipo": {
      "positiva": {
        "instancias": 39,
        "tamanho_medio": 4.23,
        "tempo_medio": 0.3192
      }
    },
    "explicacoes": [
      {
        "indice": 0,
        "classe_real": "Classe1",
        "tipo_predicao": "POSITIVA",
        "features_selecionadas": ["feat1", "feat3", "feat5"],
        "tamanho": 3,
        "tempo_segundos": 0.1234
      }
    ]
  }
}
```

---

## 🔍 COMPARAÇÃO: ANTES vs DEPOIS

### ❌ ANTES (benchmark_peab.py)
```python
# Problema: Executa PEAB + PuLP juntos
# Problema: Não salva JSON do PuLP
# Problema: Difícil comparar PuLP com outros métodos
# Problema: Não modular

executar_benchmark()  # Faz tudo de uma vez
├── Treina modelo
├── Executa PEAB
├── Executa PuLP
├── Compara
└── Salva apenas CSV de comparação
```

### ✅ DEPOIS (Arquitetura Modular)
```python
# Solução: Métodos independentes
# Solução: Todos salvam JSON padronizado
# Solução: Comparações reutilizam JSONs
# Solução: Modular e escalável

# Passo 1: Executar métodos (pode ser em momentos diferentes)
python pulp_experiment.py   # Salva json/pulp_results.json
python peab.py              # Salva json/peab_results.json

# Passo 2: Comparar (lê JSONs já salvos)
python peab_vs_pulp.py      # Lê JSONs, calcula GAP, gera relatório

# Passo 3: Comparação geral
python compare_all_methods.py  # Compara TODOS os métodos
```

---

## 🎯 VANTAGENS DA NOVA ARQUITETURA

### 1. **Modularidade** ✅
- Cada método roda independente
- Não precisa reprocessar tudo para comparar
- Fácil adicionar novos métodos

### 2. **Reprodutibilidade** ✅
- JSONs servem como cache
- Mesmos dados para todas as comparações
- Experimentos podem ser refeitos parcialmente

### 3. **Consistência** ✅
- Formato JSON padronizado
- Mesma estrutura de diretórios
- Mesmas métricas calculadas

### 4. **Escalabilidade** ✅
- Fácil adicionar LIME, SHAP, etc.
- Fácil adicionar novas comparações
- Fácil paralelizar execuções

### 5. **Organização Acadêmica** ✅
- Pasta `experiments/` separa código de análise
- Fácil gerar tabelas para dissertação
- Código limpo para revisores

---

## 📝 EXEMPLO DE USO COMPLETO

### Cenário: Testar novo método em 3 datasets

#### Passo 1: Executar métodos individuais
```bash
# PEAB
python peab.py
# Seleciona: wine, sonar, breast_cancer
# Gera: json/peab_results.json

# PuLP (deixar rodando overnight)
python pulp_experiment.py
# Seleciona: wine, sonar, breast_cancer
# Gera: json/pulp_results.json

# Anchor
python anchor.py
# Seleciona: wine, sonar, breast_cancer
# Gera: json/anchor_results.json
```

#### Passo 2: Comparações específicas
```bash
# PEAB vs PuLP (otimalidade)
python peab_vs_pulp.py
# Lê: peab_results.json + pulp_results.json
# Gera: results/benchmark/peab_vs_pulp/

# PEAB vs Anchor vs MinExp
python compare_all_methods.py
# Lê: todos os JSONs
# Gera: results/benchmark/all_methods/
```

#### Passo 3: Relatório final
```bash
python gerar_relatorio_completo.py
# Agrega todas as comparações
# Gera tabelas LaTeX
# Gera gráficos acadêmicos
# Output: results/RELATORIO_FINAL.pdf
```

---

## 🔧 MANUTENÇÃO E EXTENSÃO

### Adicionar Novo Método (ex: LIME)
```python
# 1. Criar experiments/lime_experiment.py
# 2. Implementar seguindo formato padrão
# 3. Salvar em json/lime_results.json
# 4. Atualizar compare_all_methods.py para incluir LIME
# 5. Adicionar opção no menu main.py
```

### Adicionar Nova Comparação
```python
# Criar experiments/peab_vs_lime.py
# Ler json/peab_results.json + json/lime_results.json
# Calcular métricas específicas
# Salvar em results/benchmark/peab_vs_lime/
```

---

## 📚 REFERÊNCIAS DE CÓDIGO

### Código Existente para Reutilizar:
```python
# De utils/results_handler.py
update_method_results()  # Salvar JSONs
_to_builtin()           # Serialização

# De utils/progress_bar.py
ProgressBar()           # Barra de progresso

# De peab.py
treinar_e_avaliar_modelo()  # Treino consistente
_get_lr()                    # Extrair logreg
```

### Código Novo Criado:
```python
# pulp_experiment.py
calcular_explicacao_otima_pulp()  # Solver
executar_experimento_pulp()        # Main
gerar_relatorio_pulp()             # Report
```

---

**Última atualização**: 09/12/2025  
**Status**: FASE 1 ✅ Concluída | FASE 2 ⏳ Próxima
