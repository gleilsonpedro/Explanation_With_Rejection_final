# 📋 PROPOSTA DE ORGANIZAÇÃO PROFISSIONAL DO PROJETO
## Explainable AI with Rejection Option - Dissertação de Mestrado

---

## 🎯 OBJETIVOS DA REORGANIZAÇÃO

1. **Separar** código principal de scripts auxiliares/temporários
2. **Organizar** resultados por tipo (JSON, relatórios, gráficos, benchmarks)
3. **Centralizar** execução através de um menu principal
4. **Facilitar** replicação e compreensão do projeto
5. **Preparar** para publicação/compartilhamento

---

## 📂 ESTRUTURA PROPOSTA

```
Explanation_With_Rejection_final/
│
├── 📄 README.md                          # Documentação principal do projeto
├── 📄 requirements.txt                   # Dependências Python
├── 📄 .gitignore                         # Arquivos ignorados pelo Git
├── 📄 LEIA_ME.txt                        # Instruções em português
│
├── 🎮 main.py                            # ⭐ SCRIPT MESTRE - MENU PRINCIPAL
│
├── 📁 src/                               # Código fonte principal
│   ├── 📄 __init__.py
│   ├── 📄 peab.py                        # Método PEAB (seu método principal)
│   ├── 📄 anchor.py                      # Baseline: Anchor
│   ├── 📄 minexp.py                      # Baseline: MinExp
│   └── 📄 benchmark_peab.py              # Comparação PEAB vs MILP (PuLP)
│
├── 📁 data/                              # Datasets e carregamento
│   ├── 📄 __init__.py
│   ├── 📄 datasets.py                    # Funções de carregamento
│   ├── 📄 pima-indians-diabetes.csv
│   ├── 📄 data_banknote_authentication.txt
│   ├── 📄 sonar.all-data
│   └── 📄 winequality-red.csv
│
├── 📁 utils/                             # Utilitários compartilhados
│   ├── 📄 __init__.py
│   ├── 📄 shared_training.py             # Treino de modelos
│   ├── 📄 rejection_logic.py             # Lógica de rejeição
│   ├── 📄 results_handler.py             # Salvar/carregar resultados
│   ├── 📄 progress_bar.py                # Barra de progresso
│   ├── 📄 svm_explainer.py               # Explicador SVM (se usado)
│   └── 📄 find_best_hyperparameters.py   # Otimização de hiperparâmetros
│
├── 📁 analysis/                          # Scripts de análise e visualização
│   ├── 📄 __init__.py
│   ├── 📄 generate_comparative_plots.py  # Gráficos comparativos (tempo, acurácia, etc)
│   ├── 📄 generate_comparative_tables.py # Tabelas para dissertação
│   ├── 📄 visualize_mnist_explanations.py # Visualização de explicações MNIST
│   ├── 📄 statistical_tests.py           # Testes estatísticos
│   └── 📄 summarize_results.py           # Resumo geral dos experimentos
│
├── 📁 config/                            # Arquivos de configuração
│   ├── 📄 hiperparametros.json           # Hiperparâmetros por dataset
│   └── 📄 experiment_config.json         # Configurações gerais dos experimentos
│
├── 📁 results/                           # 🎯 TODOS OS RESULTADOS AQUI
│   │
│   ├── 📁 json/                          # Resultados brutos em JSON
│   │   ├── 📄 comparative_results.json   # Comparação entre métodos
│   │   ├── 📄 peab_results.json          # Resultados PEAB (separado)
│   │   ├── 📄 anchor_results.json        # Resultados Anchor (separado)
│   │   └── 📄 minexp_results.json        # Resultados MinExp (separado)
│   │
│   ├── 📁 reports/                       # Relatórios textuais
│   │   ├── 📁 peab/
│   │   │   ├── 📄 peab_mnist.txt
│   │   │   ├── 📄 peab_pima_indians_diabetes.txt
│   │   │   └── 📄 ...
│   │   ├── 📁 anchor/
│   │   │   └── 📄 anchor_*.txt
│   │   ├── 📁 minexp/
│   │   │   └── 📄 minexp_*.txt
│   │   └── 📁 benchmark/                 # Comparação PEAB vs MILP
│   │       ├── 📄 benchmark_mnist.txt
│   │       └── 📄 ...
│   │
│   ├── 📁 plots/                         # Gráficos e visualizações
│   │   ├── 📁 comparative/               # Comparações entre métodos
│   │   │   ├── 📄 execution_time_comparison.png
│   │   │   ├── 📄 explanation_size_comparison.png
│   │   │   ├── 📄 accuracy_comparison.png
│   │   │   └── 📄 rejection_rate_comparison.png
│   │   ├── 📁 mnist_explanations/        # Visualizações MNIST
│   │   │   ├── 📄 mnist_positive_examples.png
│   │   │   ├── 📄 mnist_negative_examples.png
│   │   │   └── 📄 mnist_rejected_examples.png
│   │   ├── 📁 score_overlap/             # Análise de sobreposição
│   │   │   └── 📄 ...
│   │   └── 📁 interactive/               # Gráficos interativos (Plotly)
│   │       └── 📄 ...
│   │
│   └── 📁 tables/                        # Tabelas para dissertação
│       ├── 📄 comparative_metrics.csv
│       ├── 📄 comparative_metrics.tex    # LaTeX para dissertação
│       ├── 📄 statistical_tests.csv
│       └── 📄 hyperparameters_table.csv
│
├── 📁 notebooks/                         # Jupyter Notebooks (análises exploratórias)
│   ├── 📄 exploratory_analysis.ipynb
│   ├── 📄 mnist_visualization.ipynb
│   └── 📄 results_analysis.ipynb
│
├── 📁 scripts/                           # Scripts auxiliares/temporários
│   ├── 📄 busca_mnist.py                 # Busca hiperparâmetros MNIST
│   ├── 📄 teste_debug_mnist.py           # Testes e debug
│   └── 📄 diagnostico_*.py               # Scripts de diagnóstico
│
├── 📁 docs/                              # Documentação adicional
│   ├── 📄 methodology.md                 # Descrição da metodologia
│   ├── 📄 datasets.md                    # Descrição dos datasets
│   ├── 📄 setup_guide.md                 # Guia de instalação
│   └── 📁 images/                        # Imagens para documentação
│
├── 📁 tests/                             # Testes unitários (futuro)
│   ├── 📄 __init__.py
│   ├── 📄 test_peab.py
│   └── 📄 test_utils.py
│
└── 📁 env/                               # Ambiente virtual (não versionar)

```

---

## 🎮 MENU PRINCIPAL PROPOSTO (main.py)

```
═══════════════════════════════════════════════════════════════════
    EXPLAINABLE AI WITH REJECTION OPTION - Sistema de Experimentos
═══════════════════════════════════════════════════════════════════

[1] 🔬 EXECUTAR EXPERIMENTOS
    ├── [1.1] PEAB (Método Proposto)
    ├── [1.2] Anchor (Baseline)
    ├── [1.3] MinExp (Baseline)
    ├── [1.4] Executar Todos os Métodos
    └── [1.5] Benchmark PEAB vs MILP (PuLP)

[2] 📊 ANÁLISE E VISUALIZAÇÃO
    ├── [2.1] Gerar Gráficos Comparativos
    ├── [2.2] Gerar Tabelas para Dissertação
    ├── [2.3] Visualizar Explicações MNIST
    ├── [2.4] Análise Estatística Completa
    └── [2.5] Resumo Geral dos Resultados

[3] 🔧 UTILITÁRIOS
    ├── [3.1] Buscar Melhores Hiperparâmetros
    ├── [3.2] Validar Datasets
    ├── [3.3] Limpar Resultados Antigos
    └── [3.4] Exportar Resultados (ZIP)

[4] 📚 DOCUMENTAÇÃO
    ├── [4.1] Ver Descrição dos Métodos
    ├── [4.2] Ver Descrição dos Datasets
    └── [4.3] Abrir Documentação Completa

[0] ❌ SAIR

═══════════════════════════════════════════════════════════════════
```

---

## 📝 MAPEAMENTO: ARQUIVOS ATUAIS → ESTRUTURA PROPOSTA

### ✅ MANTER NA RAIZ
- `README.md` → Manter
- `requirements.txt` → Manter
- `.gitignore` → Manter
- `LEIA_ME.txt` → Manter

### 📦 CRIAR NOVO
- `main.py` → **NOVO** - Menu principal

### 📂 MOVER PARA `src/`
- `peab.py` → `src/peab.py`
- `anchor.py` → `src/anchor.py`
- `minexp.py` → `src/minexp.py`
- `benchmark_peab.py` → `src/benchmark_peab.py`

### 📂 MOVER PARA `analysis/`
- `visualizer.py` → `analysis/visualize_mnist_explanations.py`
- `summarize_results.py` → `analysis/summarize_results.py`
- `benchmark_analysis.py` → `analysis/generate_comparative_plots.py`
- `bench_PLOTLY.py` → `analysis/generate_comparative_plots.py` (fundir)
- `bench_resumo.py` → `analysis/generate_comparative_tables.py`
- Scripts de `audit/` → `analysis/` (detailed_json.py, score_overlap.py)

### 📂 MOVER PARA `scripts/` (auxiliares)
- `busca_mnist.py` → `scripts/busca_mnist.py`
- `diagnostico_*.py` → `scripts/diagnostico_*.py`
- `teste_debug_mnist.py` → `scripts/teste_debug_mnist.py`
- `peab_copy.py` → `scripts/` ou DELETAR
- `benchmark_peab copy.py` → `scripts/` ou DELETAR

### 📂 MOVER PARA `config/`
- `json/hiperparametros.json` → `config/hiperparametros.json`

### 📂 REORGANIZAR `results/`
- `json/comparative_results.json` → `results/json/comparative_results.json`
- `results/report/peab/` → `results/reports/peab/`
- `results/report/anchor/` → `results/reports/anchor/`
- `results/report/minexp/` → `results/reports/minexp/`
- `results/benchmark/` → `results/reports/benchmark/`
- `results/plots/` → Manter (organizar subpastas)
- `results/plots_interativos/` → `results/plots/interactive/`
- `analysis_output/plots/` → `results/plots/score_overlap/`

### 📂 PASTAS PARA DELETAR/ARQUIVAR
- `test_old/` → Arquivar ou deletar
- `analysis_old/` → Arquivar ou deletar
- `__pycache__/` → Ignorar no .gitignore
- `env/` → Manter mas ignorar no .gitignore
- `audit/` → Mover conteúdo para `analysis/`

### 📂 MANTER
- `data/` → Manter estrutura
- `utils/` → Manter estrutura

---

## 🚀 FLUXO DE TRABALHO PROPOSTO

### 1️⃣ **Executar Experimentos**
```bash
python main.py
# Selecionar opção [1.4] - Executar Todos os Métodos
# Resultados salvos automaticamente em results/
```

### 2️⃣ **Gerar Análises**
```bash
python main.py
# Selecionar opção [2] - Análise e Visualização
# Gráficos → results/plots/comparative/
# Tabelas → results/tables/
```

### 3️⃣ **Visualizar MNIST**
```bash
python main.py
# Selecionar opção [2.3] - Visualizar Explicações MNIST
# Imagens → results/plots/mnist_explanations/
```

### 4️⃣ **Benchmark PEAB vs MILP**
```bash
python main.py
# Selecionar opção [1.5] - Benchmark PEAB vs MILP
# Relatórios → results/reports/benchmark/
```

---

## 📋 CHECKLIST DE MIGRAÇÃO

### Fase 1: Preparação
- [ ] Fazer backup completo do projeto
- [ ] Criar branch no Git: `git checkout -b reorganization`
- [ ] Criar estrutura de pastas vazia

### Fase 2: Movimentação de Arquivos
- [ ] Criar `src/` e mover scripts principais
- [ ] Criar `analysis/` e mover scripts de análise
- [ ] Criar `scripts/` e mover auxiliares
- [ ] Criar `config/` e mover JSONs de configuração
- [ ] Reorganizar `results/` com subpastas

### Fase 3: Criação de Novos Arquivos
- [ ] Criar `main.py` com menu principal
- [ ] Criar `__init__.py` em todas as pastas de módulo
- [ ] Atualizar imports nos arquivos movidos
- [ ] Criar `docs/methodology.md`
- [ ] Criar `docs/datasets.md`

### Fase 4: Ajustes e Testes
- [ ] Testar execução do menu principal
- [ ] Verificar se todos os caminhos estão corretos
- [ ] Testar cada opção do menu
- [ ] Atualizar README.md com nova estrutura
- [ ] Atualizar .gitignore

### Fase 5: Limpeza
- [ ] Deletar arquivos `*_copy.py`
- [ ] Arquivar `test_old/` e `analysis_old/`
- [ ] Limpar `__pycache__/`
- [ ] Commit e push das mudanças

---

## 🎨 BENEFÍCIOS DA REORGANIZAÇÃO

✅ **Clareza**: Fácil identificar onde cada arquivo está
✅ **Manutenção**: Simples adicionar novos scripts
✅ **Replicação**: Outros pesquisadores conseguem rodar facilmente
✅ **Publicação**: Pronto para GitHub/GitLab público
✅ **Dissertação**: Estrutura profissional para apêndice
✅ **Backup**: Resultados organizados por tipo
✅ **Menu Central**: Uma única entrada para todas as operações

---

## 📚 DOCUMENTAÇÃO ADICIONAL SUGERIDA

### README.md deve conter:
1. Título e descrição do projeto
2. Requisitos e instalação
3. Estrutura do projeto
4. Como executar (comando `python main.py`)
5. Descrição dos métodos (PEAB, Anchor, MinExp)
6. Datasets utilizados
7. Citação (quando publicado)
8. Licença

### docs/methodology.md deve conter:
- Descrição detalhada do método PEAB
- Algoritmo passo a passo
- Pseudocódigo
- Diferenças para baselines

### docs/datasets.md deve conter:
- Lista de datasets
- Características (instâncias, features, classes)
- Fonte de cada dataset
- Pré-processamento aplicado

---

## 💡 PRÓXIMOS PASSOS

1. **Revisar esta proposta** e ajustar conforme necessário
2. **Fazer backup** completo do projeto
3. **Seguir o checklist** de migração fase por fase
4. **Testar cada fase** antes de prosseguir para a próxima
5. **Documentar** mudanças no README.md

---

## ⚠️ OBSERVAÇÕES IMPORTANTES

- **NÃO deletar nada** antes de ter backup
- **Testar após cada movimentação** para garantir que tudo funciona
- **Atualizar imports** quando mover arquivos entre pastas
- **Manter env/ fora do Git** (.gitignore)
- **Commitar frequentemente** durante a reorganização

---

**Autor da Proposta**: GitHub Copilot  
**Data**: 06/12/2025  
**Projeto**: Explainable AI with Rejection Option - Dissertação de Mestrado
