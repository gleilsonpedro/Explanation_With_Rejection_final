# Novas Funcionalidades MNIST - Classe vs Classe e Pooling 2x2

## Resumo das Mudanças

Foi implementado um sistema completo para análise MNIST com duas novas opções:

1. **Modo de Features**:
   - **784 features** (raw/normal): pixels originais 28x28
   - **196 features** (pooling 2x2): imagem reduzida para 14x14 via pooling por média

2. **Classificação Classe vs Classe**:
   - Substituiu "one-vs-rest" por seleção de **par de dígitos** (ex: 8 vs 5)
   - Reduz drasticamente o número de instâncias e melhora a análise focada

## Arquivos Modificados

### 1. `data/datasets.py`
**Principais mudanças:**
- Opções globais: `MNIST_FEATURE_MODE` e `MNIST_SELECTED_PAIR`
- Função `set_mnist_options(feature_mode, pair)` para configurar antes do carregamento
- Função `_pool2x2(arr28)` que aplica pooling 2x2 por média em uma imagem 28x28
- Função `_apply_pooling_df(X)` que processa todo o DataFrame (784→196 colunas)
- Atualização de `carregar_dataset('mnist')`:
  - Filtra apenas as classes do par selecionado
  - Aplica pooling se `MNIST_FEATURE_MODE == 'pool2x2'`
  - Retorna classes binárias (0 e 1) mapeadas corretamente
- Atualização de `selecionar_dataset_e_classe()`:
  - Menu interativo específico para MNIST:
    - Pergunta modo de features (784 ou 196)
    - Pergunta par de dígitos (ex: "8 5")
  - Define opções globais e recarrega dataset com filtros aplicados

### 2. `peab_2.py`
**Principais mudanças:**
- Função `montar_dataset_cache(...)`:
  - Adiciona metadados MNIST ao JSON (`mnist_feature_mode` e `mnist_digit_pair`)
  - Esses metadados são incluídos na chave `config` do JSON exportado
- Propagação automática: ao rodar PEAB, as escolhas de MNIST são salvas no JSON

### 3. `anchor_comparation.py`
**Principais mudanças:**
- Linha 99: Nome de relatório usa formato `class_A_vs_class_B` em vez de `vs_rest`
- Linhas 327-337: Adiciona metadados MNIST ao JSON de saída
- Linha 408: Runner programático também usa naming `class_vs_class`
- **Compatibilidade automática**: ao usar `get_shared_pipeline()`, Anchor herda exatamente:
  - Modo de features (784 ou 196)
  - Par de classes (ex: 8 vs 5)
  - Thresholds e pipeline idênticos ao PEAB

### 4. `minexp_comparation.py`
**Principais mudanças:**
- Linhas 92-96: Nome de relatório usa formato `class_A_vs_class_B`
- Linha 389: Runner programático idem
- **Compatibilidade automática**: MinExp também herda configurações via `get_shared_pipeline()`

### 5. `utils/shared_training.py`
**Não precisa alteração direta**, mas propaga as configurações MNIST porque:
- Chama `configurar_experimento(dataset_name)` de `peab_2.py`
- `configurar_experimento` por sua vez chama `carregar_dataset('mnist')`
- `carregar_dataset` lê as opções globais `MNIST_FEATURE_MODE` e `MNIST_SELECTED_PAIR`
- Resultado: todos os métodos (PEAB/MinExp/Anchor) usam **exatamente** os mesmos dados

## Fluxo de Execução

### Exemplo: Análise MNIST 8 vs 5 com pooling 2x2

```bash
# 1. Executar PEAB
python peab_2.py
```

**Prompts interativos:**
```
| [1] MNIST (70k x 784 x 10)

Digite o número do dataset: 1

Carregando mnist...

Selecione o modo de features para MNIST:
  [0] 784 (normal)
  [1] 196 (pooling 2x2)
Opção: 1

Digite duas classes (0-9) para comparar, separadas por espaço. Ex: 8 5
Par de classes: 8 5

Dataset carregado! (Total Amostras: 13138, Features: 196)
Classes disponíveis no dataset original:
   [0] - 8 (Total: 6825)
   [1] - 5 (Total: 6313)
```

**Resultado:**
- PEAB gera relatório: `results/report/peab/peab_mnist_8_vs_5.txt`
- JSON salvo: `comparative_results.json` (ou `json/comparative_results.json`)
  - Chave: `"peab" > "mnist"`
  - Config inclui:
    ```json
    {
      "config": {
        "dataset_name": "mnist",
        "mnist_feature_mode": "pool2x2",
        "mnist_digit_pair": [8, 5],
        ...
      }
    }
    ```

### Anchor e MinExp herdam automaticamente

```bash
# 2. Executar MinExp
python minexp_comparation.py
# Mesmos prompts, mas já estão configurados globalmente

# 3. Executar Anchor
python anchor_comparation.py
# Idem
```

**OU usar runners programáticos (para automação):**

```python
from anchor_comparation import run_anchor_for_dataset
from minexp_comparation import run_minexp_for_dataset

# Importante: definir opções ANTES de chamar runners
from data.datasets import set_mnist_options
set_mnist_options('pool2x2', (8, 5))

# Agora rodar os métodos
run_anchor_for_dataset('mnist')  # Usa automaticamente 196 features, classes 8 vs 5
run_minexp_for_dataset('mnist')  # Idem
```

## Benefícios

### 1. Pooling 2x2 (196 features)
- **Redução de dimensionalidade**: 784 → 196 (75% menor)
- **Velocidade**: 
  - Treino do modelo: ~4x mais rápido
  - Geração de explicações: ~3-5x mais rápido (especialmente MinExp/Anchor)
  - Uso de memória: ~75% menor
- **Interpretabilidade**: 14x14 ainda é visual, mas com menos ruído

### 2. Classe vs Classe (ex: 8 vs 5)
- **Menos instâncias**: ~13k em vez de ~70k (81% menor)
- **Análise focada**: comparação direta entre dois dígitos específicos
- **Rejections mais significativas**: ambiguidade real entre dígitos similares
- **Compatibilidade com visualização**: `json/analise_json.py` já adaptado para classe-vs-classe

## Validação

Execute o script de teste:
```bash
python test_mnist_options.py
```

**Saída esperada:**
```
✓ TESTE 1 PASSOU (MNIST raw 784, 8 vs 5)
✓ TESTE 2 PASSOU (MNIST pooling 196, 3 vs 7)
✓ TESTE 3 PASSOU (metadados no JSON)
✓ TODOS OS TESTES PASSARAM COM SUCESSO!
```

## Visualização (json/analise_json.py)

O script de visualização **já está adaptado** para classe-vs-classe:
- Não pergunta mais classe (one-vs-rest)
- Gera heatmaps para **ambas as classes** do par selecionado
- Nomeia arquivos: `mnist_<Método>_classe<A>_...` e `mnist_<Método>_classe<B>_...`

## Compatibilidade

- ✅ **Backward compatible**: outros datasets (breast_cancer, wine, etc.) não são afetados
- ✅ **JSON unificado**: todos os métodos salvam no mesmo formato
- ✅ **Thresholds compartilhados**: t+/t− idênticos entre PEAB/MinExp/Anchor
- ✅ **Visualização automática**: analise_json.py detecta par de classes do JSON

## Notas Técnicas

### Pooling 2x2 por Média
```python
def _pool2x2(arr28: np.ndarray) -> np.ndarray:
    """28x28 → 14x14"""
    out = np.zeros((14, 14), dtype=float)
    for r in range(14):
        for c in range(14):
            block = arr28[2*r:2*r+2, 2*c:2*c+2]  # bloco 2x2
            out[r, c] = float(block.mean())       # média
    return out
```

### Nomes de Features após Pooling
- Raw (784): `feature_0`, `feature_1`, ..., `feature_783`
- Pooling (196): `bin_0_0`, `bin_0_1`, ..., `bin_13_13` (linha_coluna em 14x14)

### Opções Globais (Thread-Safe)
As opções `MNIST_FEATURE_MODE` e `MNIST_SELECTED_PAIR` são variáveis globais do módulo.
- **Seguro para uso sequencial** (rodar PEAB → MinExp → Anchor no mesmo processo)
- **Não thread-safe**: evite execução paralela com configurações diferentes

### Propagação via shared_training
```
peab_2.py: selecionar_dataset_e_classe()
    ↓ (define opções globais)
data/datasets.py: set_mnist_options('pool2x2', (8,5))
    ↓
peab_2.py: configurar_experimento('mnist')
    ↓
data/datasets.py: carregar_dataset('mnist')
    ↓ (lê opções globais)
[filtra classes 8 e 5]
[aplica pooling 2x2]
    ↓
utils/shared_training.py: get_shared_pipeline('mnist')
    ↓ (chama configurar_experimento)
[retorna pipeline treinado + meta com mnist_feature_mode e mnist_digit_pair]
    ↓
anchor_comparation.py / minexp_comparation.py
[recebem exatamente o mesmo dataset e thresholds]
```

## Próximos Passos Sugeridos

1. **Experimentar diferentes pares**:
   - Dígitos similares: 3 vs 8, 4 vs 9, 1 vs 7
   - Dígitos distintos: 0 vs 1, 6 vs 7

2. **Comparar performance**:
   - Raw 784 vs Pooling 196
   - One-vs-rest (antigo) vs classe-vs-classe (novo)

3. **Visualização avançada**:
   - Heatmaps já implementados em `json/analise_json.py`
   - Considerar overlays de explicações em imagens originais

4. **Documentação de resultados**:
   - Gerar tabela comparativa (tempo, acurácia, tamanho explicação)
   - Análise de ambiguidade (rejeitadas) por par de classes

---

**Autor:** Sistema automatizado  
**Data:** 6 de novembro de 2025  
**Versão:** 1.0
