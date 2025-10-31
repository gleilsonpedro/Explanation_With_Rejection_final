
# XAI com Rejeição: Explicações Abdutivas Mínimas para Regressão Logística

Este projeto implementa e avalia um método para gerar explicações abdutivas mínimas para modelos de Regressão Logística com opção de rejeição. O objetivo é fornecer explicações que sejam mínimas (usando o menor número possível de features) e confiáveis para as predições do modelo.

## Conceitos Fundamentais

* **Explicação Abdutiva:** Um conjunto suficiente de features e seus valores que, por si só, justificam a predição do modelo para uma determinada instância.
* **Classificador com Rejeição:** Um modelo que, além de classificar em Classe 0 ou Classe 1, pode optar por não tomar uma decisão ("rejeitar") quando a instância cai em uma "zona de incerteza", aumentando assim a confiabilidade das predições realizadas.
* **Minimalidade:** Busca-se encontrar a explicação abdutiva que utilize o menor número possível de features, mantendo a qualidade da explicação.
* **Confiabilidade:** As explicações são validadas para garantir que representam fielmente o comportamento do modelo.

## Visão Geral do Projeto

O projeto consiste em três componentes principais:

1. **Gerador de Explicações Abdutivas Mínimas**
   - Implementa um algoritmo para encontrar explicações mínimas
   - Inclui mecanismo de rejeição para casos ambíguos
   - Otimiza explicações para máxima confiabilidade

2. **Sistema de Comparação**
   - ANCHOR: Comparação com explicações baseadas em regras
   - MinExp: Comparação com explicações minimais tradicionais
   - Análise detalhada das diferenças e vantagens

3. **Framework de Análise**
   - Métricas de avaliação: tamanho das explicações, fidelidade
   - Análise estatística dos resultados
   - Visualizações comparativas detalhadas

## Estrutura do Projeto

```
├── utils/                  # Módulos de suporte
│   ├── results_handler.py  # Gerenciamento de resultados
│   ├── svm_explainer.py   # Explicador base
│   ├── utility.py         # Funções utilitárias
│   ├── plot_generator.py  # Geração de gráficos
│   └── find_best_hyperparameters.py # Otimização de parâmetros
├── json/                   # Arquivos de configuração
│   └── hiperparametros.json # Parâmetros otimizados
├── results/               # Resultados e relatórios
├── cache/                # Cache de processamento
└── data/                 # Datasets e dados processados

# Arquivos Principais
├── peab_comparation.py    # Experimento principal
├── minexp_comparation.py  # Comparação com MinExp
├── anchor_comparation.py  # Comparação com ANCHOR
└── detailed_explanation.py # Análise detalhada de explicações
```

## Fluxo de Execução e Estrutura de Arquivos

### 1. Preparação do Ambiente
```
├── json/                   # Configurações e parâmetros
│   └── hiperparametros.json  # Parâmetros otimizados por dataset
├── cache/                  # Armazenamento de processamento
│   └── cache_cumulativo.pkl  # Cache centralizado dos resultados
├── data/                   # Dados de entrada
│   └── datasets/          # Datasets processados
└── results/               # Resultados e relatórios
    └── report/           # Logs detalhados por dataset
```

### 2. Execução Principal (peab_comparation.py)
O script realiza as seguintes etapas para cada dataset:

1. **Carregamento e Configuração**
   - Lê hiperparâmetros de `json/hiperparametros.json`
   - Carrega dataset da pasta `data/`
   - Inicializa cache em `cache/cache_cumulativo.pkl`

2. **Processamento**
   - Pipeline com MinMaxScaler e LogisticRegression
   - Split treino/teste controlado
   - Cálculo de thresholds de rejeição

3. **Geração de Explicações**
   - Processamento por instância
   - Cálculo de features relevantes
   - Validação de minimalidade

4. **Salvamento de Resultados**
   - Log detalhado em `results/report/<dataset>_report.txt`
   - Atualização do cache em `cache/cache_cumulativo.pkl`
   - Métricas e estatísticas em JSON

### 3. Análise Comparativa
Execução dos comparadores:
```
├── anchor_comparation.py   # Comparação com ANCHOR
├── minexp_comparation.py   # Comparação com MinExp
└── analyze_results.py      # Análise consolidada
```

### 4. Visualização e Relatórios
```
results/
├── report/                # Logs detalhados por dataset
│   ├── dataset1_report.txt
│   └── dataset2_report.txt
└── plots/                # Visualizações comparativas
```

## Como Usar

1. **Configuração do Ambiente**
   ```bash
   python -m venv env
   source env/bin/activate  # ou env\Scripts\activate no Windows
   pip install -r requirements.txt
   ```

2. **Execução dos Experimentos**
   ```python
   # Experimento principal com cache cumulativo
   python peab_comparation.py
   
   # Comparações com outros métodos
   python anchor_comparation.py
   python minexp_comparation.py
   
   # Análise dos resultados
   python analyze_results.py
   ```

3. **Estrutura dos Resultados**
   ```
   results/
   ├── report/                      # Logs detalhados
   │   ├── peab_dataset1.txt       # Log PEAB
   │   ├── anchor_dataset1.txt     # Log ANCHOR
   │   └── minexp_dataset1.txt     # Log MinExp
   └── plots/                      # Visualizações
       └── comparisons/            # Gráficos comparativos
   ```

4. **Cache e Reprodutibilidade**
   - Cache cumulativo em `cache/cache_cumulativo.pkl`
   - Hiperparâmetros em `json/hiperparametros.json`
   - Datasets processados em `data/`

## Datasets Suportados

O projeto inclui suporte para diversos datasets de classificação binária:
- Breast Cancer
- Iris (convertido para binário)
- MNIST (convertido para binário: dígito '0' vs resto)
- Pima Indians Diabetes
- Sonar
- Vertebral Column
- Wine (convertido para binário)
- E outros

### Datasets e cache offline

- Alguns datasets são baixados de fontes públicas (UCI, GitHub raw). Para evitar erros temporários de rede (ex.: HTTP 429 Too Many Requests), o carregador salva uma cópia local na pasta `data/` e passa a reutilizá-la nas próximas execuções.
- Exemplo (Pima Indians Diabetes):
   - Arquivo de cache: `data/pima-indians-diabetes.csv`
   - Se o download falhar, você pode baixar manualmente e salvar nesse caminho; o código detectará o arquivo local automaticamente.

## Contribuições Principais

1. **Explicações Mínimas com Rejeição**
   - Nova abordagem para lidar com casos ambíguos
   - Garantia de minimalidade e confiabilidade
   - Otimização do processo de geração de explicações

2. **Framework Comparativo**
   - Metodologia sistemática de avaliação
   - Métricas abrangentes de qualidade
   - Análise detalhada do desempenho

3. **Resultados Empíricos**
   - Análise extensiva em múltiplos datasets
   - Validação estatística dos resultados
   - Demonstração das vantagens da abordagem proposta

## Referências

[A serem adicionadas - artigos relevantes sobre XAI, rejeição e explicações abdutivas]