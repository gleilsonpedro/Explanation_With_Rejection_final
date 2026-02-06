# Scripts para Geração de Pipeline do Artigo

Este diretório contém scripts para gerar automaticamente as informações sobre o pipeline experimental para inclusão no artigo científico.

## Scripts Disponíveis

### 1. `gerar_pipeline_artigo.py` (com emojis - terminal UTF-8)
Gera o pipeline completo com todas as informações detalhadas de cada dataset.
Inclui emojis para melhor visualização no terminal.

**Uso:**
```bash
python temporarios\gerar_pipeline_artigo.py
```

**Inclui:**
- Fonte de cada dataset
- Split treino/teste
- Subsample (se aplicado)
- Número de features
- Configurações especiais do MNIST (feature mode, digit pair)
- Rejection cost
- Métricas de performance (acurácia base, com rejeição, taxa de rejeição, ganho)
- Tabela resumo de todos os datasets
- Estatísticas gerais do experimento

---

### 2. `gerar_pipeline_artigo_txt.py` (sem emojis - compatível Windows)
Versão sem emojis, ideal para redirecionar para arquivo de texto.

**Uso:**
```bash
python temporarios\gerar_pipeline_artigo_txt.py > pipeline_completo.txt
```

**Vantagens:**
- Sem problemas de encoding
- Pode ser salvo em arquivo .txt
- Compatível com qualquer terminal

---

### 3. `gerar_resumo_executivo.py`
Gera um resumo executivo compacto em formato tabular.

**Uso:**
```bash
python temporarios\gerar_resumo_executivo.py
```

**Formato:**
- Tabela compacta com uma linha por dataset
- Estatísticas gerais ao final
- Lista de configurações especiais
- Ideal para referência rápida

---

### 4. `gerar_tabelas_latex.py`
Gera tabelas formatadas em LaTeX prontas para copiar no artigo.

**Uso:**
```bash
python temporarios\gerar_tabelas_latex.py
```

**Gera:**
- Tabela de configuração experimental (formato LaTeX)
- Tabela de resultados de desempenho (formato LaTeX)
- Descrição textual para a seção de metodologia
- Comentários úteis para o artigo

**Importante:** Requer `\usepackage{booktabs}` no preâmbulo do LaTeX.

---

## Informações Geradas

### Para cada dataset, os scripts fornecem:

1. **Identificação:**
   - Nome do dataset
   - Fonte (UCI, Kaggle, etc.)

2. **Configuração:**
   - Split treino/teste (ex: 70%/30%)
   - Subsample aplicado (se houver)
   - Número de features
   - Rejection cost utilizado

3. **Configurações Especiais (MNIST):**
   - Feature mode: raw (784 features, 28x28) ou pool2x2 (196 features, 14x14)
   - Digit pair: par de dígitos classificados (ex: 3 vs 8)
   - Top-K features (se aplicado)

4. **Resultados:**
   - Instâncias de teste
   - Acurácia sem rejeição
   - Acurácia com rejeição
   - Taxa de rejeição
   - Ganho de acurácia (em pontos percentuais)

5. **Estatísticas Gerais:**
   - Total de datasets avaliados
   - Datasets que usaram subsample
   - Total de instâncias de teste
   - Ganho médio/máximo/mínimo de acurácia
   - Taxa média/máxima/mínima de rejeição

---

## Exemplos de Uso

### Gerar e visualizar no terminal:
```bash
python temporarios\gerar_pipeline_artigo.py
```

### Salvar em arquivo de texto:
```bash
python temporarios\gerar_pipeline_artigo_txt.py > meu_pipeline.txt
```

### Gerar apenas resumo executivo:
```bash
python temporarios\gerar_resumo_executivo.py > resumo.txt
```

### Gerar tabelas LaTeX:
```bash
python temporarios\gerar_tabelas_latex.py > tabelas.tex
```

---

## Datasets com Configurações Especiais

### Subsample Aplicado:
- **covertype**: 1% (dataset muito grande)
- **creditcard**: 2% (dataset muito grande)
- **mnist**: 12% (para testes mais rápidos)
- **newsgroups**: 10% (dataset de texto grande)
- **rcv1**: 5% (dataset de texto muito grande)

### MNIST:
- **Feature mode**: raw (28x28 pixels = 784 features)
- **Classificação**: Dígitos 3 vs 8 (classificação binária)
- **Pooling**: Pode usar pool2x2 para reduzir para 14x14 = 196 features (4x mais rápido)

---

## Observações

- Todos os scripts leem os arquivos JSON em `json/peab/`
- Os scripts são independentes e podem ser executados em qualquer ordem
- Para datasets com alta taxa de rejeição (>50%), uma observação é incluída
- Para datasets com alta acurácia final (>95%), uma observação é incluída
- Os scripts detectam automaticamente as configurações do MNIST baseado no número de features

---

## Referências para o Artigo

Use os dados gerados para:
1. **Seção de Metodologia**: Descrever o protocolo experimental
2. **Seção de Resultados**: Apresentar as métricas de performance
3. **Tabelas**: Copiar as tabelas LaTeX geradas
4. **Discussão**: Usar as estatísticas gerais para análise

---

**Criado em:** 05/02/2026
**Autor:** Sistema automatizado de documentação
**Versão:** 1.0
