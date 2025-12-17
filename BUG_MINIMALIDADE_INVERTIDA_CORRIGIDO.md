# Bug Corrigido: Padrão Invertido de Minimalidade

## Data: 2025-01-XX

## Problema Identificado

Ao executar a validação de minimalidade, observou-se um **padrão invertido** entre predições positivas e negativas:

| Dataset | Positivas | Negativas | Padrão |
|---------|-----------|-----------|--------|
| Banknote | 94.05% | 4.94% | INVERTIDO! |
| Vertebral | 0.98% | 92.75% | INVERTIDO! |

Quando positivas tinham alta minimalidade, negativas tinham baixa, e vice-versa.

## Causa Raiz

O teste de minimalidade usa perturbações uniformes para verificar se uma feature é necessária.
**Problema**: O espaço de perturbação uniforme não é neutro - ele tende a gerar scores que caem predominantemente de um lado da fronteira de decisão (score=0).

### Diagnóstico por Dataset:

| Dataset | Bias Ratio | %Positivas | %Negativas |
|---------|-----------|-----------|-----------|
| pima | 0.85 | 45.9% | 54.1% | ← balanceado ✅
| breast_cancer | 0.88 | 46.9% | 53.1% | ← balanceado ✅
| **banknote** | 0.17 | 14.6% | **85.4%** | ← BIAS NEGATIVAS
| **vertebral** | 51.63 | **98.1%** | 1.9% | ← BIAS POSITIVAS
| **sonar** | 1000 | **100%** | 0% | ← BIAS EXTREMO

### Impacto no Teste de Minimalidade:

Para **vertebral_column** (98.1% das perturbações são positivas):
- **Instâncias POSITIVAS**: Mesmo removendo qualquer feature, ~98% das perturbações ainda são positivas
  - Se 98% > 95% → todas features parecem "redundantes" → minimalidade ~0%
  
- **Instâncias NEGATIVAS**: Mesmo com features fixas, ~98% das perturbações são positivas
  - Apenas ~2% mantêm classe negativa → <95% → todas features parecem "necessárias" → minimalidade ~100%

## Solução Implementada

### Correção: Threshold Adaptativo com Baseline

Em vez de usar um threshold fixo de 95%, agora calculamos o **baseline** esperado (probabilidade de manter a predição por acaso) e ajustamos o threshold dinamicamente.

```python
def calcular_baseline_predicao(pipeline, X_train, y_pred, rejeitada, t_plus, t_minus, max_abs, n_samples=500):
    """
    Calcula baseline: probabilidade de manter predição com perturbações 100% uniformes.
    """
    # Gerar perturbações sem fixar nada
    perturbacoes = ... # 100% uniforme
    
    if rejeitada:
        acertos = np.sum((scores >= t_minus) & (scores <= t_plus))
    else:
        acertos = np.sum(predicoes == y_pred)
    
    return acertos / n_samples
```

### Cálculo do Threshold Ajustado:

```python
# Margem = metade do ganho possível sobre baseline
margem = (1.0 - baseline) * 0.5 if baseline < 1.0 else 0.0

# Threshold = max(baseline + margem, 0.95)
threshold_ajustado = max(baseline + margem, 0.95)
```

**Exemplos:**
- Se baseline = 50% → threshold = max(50% + 25%, 95%) = **95%** (padrão)
- Se baseline = 85% → threshold = max(85% + 7.5%, 95%) = **95%** (padrão)
- Se baseline = 98% → threshold = max(98% + 1%, 95%) = **99%** (mais exigente)

## Resultados Após Correção

| Dataset | Antes | Depois |
|---------|-------|--------|
| pima_indians_diabetes | ~? | **76.44%** ✅ |
| vertebral_column | ~0.98% / 92.75% (invertido) | **54.70%** ✅ |
| banknote | ~94.05% / 4.94% (invertido) | **58.07%** ✅ |

## Arquivos Modificados

- `peab_validation.py`:
  - Adicionada função `calcular_baseline_predicao()`
  - Modificada função `validar_necessidade_features()` para usar threshold adaptativo
  - Adicionado cache de baseline por tipo de predição

## Impacto Acadêmico

Esta correção é **crítica para a validade acadêmica** do experimento:

1. **Antes**: Resultados de minimalidade eram enviesados pela geometria do dataset
2. **Depois**: Resultados refletem a real contribuição das features na explicação

O teste agora pergunta: "Fixar esta feature melhora a taxa de manutenção da predição **além do que seria esperado por acaso**?"
