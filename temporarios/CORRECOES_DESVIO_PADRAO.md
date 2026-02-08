# Correções para Gerar Desvio Padrão nas Tabelas de Runtime

## 📋 Resumo das Correções

### Problema Identificado
Os scripts **Anchor** e **MinExp** não estavam salvando o tempo de execução individual (`computation_time`) para cada instância no JSON, impossibilitando o cálculo do desvio padrão.

### ✅ Correções Implementadas

#### 1. **anchor.py** (3 alterações)
- **Linha ~189**: Adicionado dicionário `tempos_individuais = {}` para armazenar tempo por instância
- **Linha ~259**: Adicionado `tempos_individuais[i] = runtime` para salvar tempo de cada instância
- **Linha ~690**: Adicionado `'computation_time': float(tempos_ind_local.get(i, 0.0))` no `per_instance`

**Resultado**: Anchor agora salva o tempo de cada instância no JSON.

#### 2. **minexp.py** (2 alterações)
- **Linha ~392**: Renomeado `'tempo_segundos'` para `'computation_time'` (consistência)
- **Linha ~737**: Renomeado `'tempo_segundos'` para `'computation_time'` (consistência)

**Resultado**: MinExp agora usa o mesmo campo que PEAB e Anchor.

#### 3. **temporarios/gerar_tabela_runtime_unificada.py** (1 alteração)
- Removido fallback `or inst.get("tempo_segundos")` já que agora todos usam `computation_time`

**Resultado**: Script simplificado e consistente.

---

## 🔄 Plano de Re-execução

### Datasets que Precisam Ser Re-executados

#### ✅ Alta Prioridade (para tabela principal)
1. **Banknote** - Rápido (~1 min)
2. **Vertebral Column** - Rápido (~1 min)
3. **Pima Indians** - Rápido (~2 min)
4. **Heart Disease** - Rápido (~1 min)
5. **Breast Cancer** - Médio (~5 min)
6. **Sonar** - Médio (~10 min)
7. **Spambase** - Médio (~5 min)

#### ⚠️ Datasets Demorados (opcional, pode rodar depois)
8. **Credit Card** - Anchor lento (~30 min)
9. **Covertype** - Anchor muito lento (~2-3 horas)
10. **MNIST (3 vs 8)** - Anchor extremamente lento (~5-8 horas)

---

## 📝 Comandos para Re-execução

### 1. PEAB (MINABRO) - TODOS OS DATASETS
```bash
# Datasets rápidos (7-10 min total)
python peab.py --dataset banknote
python peab.py --dataset vertebral_column
python peab.py --dataset pima_indians_diabetes
python peab.py --dataset heart_disease
python peab.py --dataset breast_cancer
python peab.py --dataset sonar
python peab.py --dataset spambase

# Datasets demorados (1-2 horas total)
python peab.py --dataset creditcard
python peab.py --dataset covertype
python peab.py --dataset mnist
```

### 2. Anchor - PRIORIZAR RÁPIDOS
```bash
# PRIORIDADE 1: Datasets rápidos (15-30 min total)
python anchor.py --dataset banknote
python anchor.py --dataset vertebral_column
python anchor.py --dataset pima_indians_diabetes
python anchor.py --dataset heart_disease
python anchor.py --dataset breast_cancer
python anchor.py --dataset sonar
python anchor.py --dataset spambase

# PRIORIDADE 2: Datasets demorados (RODAR SEPARADAMENTE, DE PREFERÊNCIA À NOITE)
# python anchor.py --dataset creditcard        # ~30 min
# python anchor.py --dataset covertype         # ~2-3 horas
# python anchor.py --dataset mnist             # ~5-8 horas (!)
```

### 3. MinExp (AbLinRO) - TODOS OS DATASETS
```bash
# Datasets rápidos (10-15 min total)
python minexp.py --dataset banknote
python minexp.py --dataset vertebral_column
python minexp.py --dataset pima_indians_diabetes
python minexp.py --dataset heart_disease
python minexp.py --dataset breast_cancer
python minexp.py --dataset sonar
python minexp.py --dataset spambase

# Datasets demorados (2-3 horas total)
python minexp.py --dataset creditcard
python minexp.py --dataset covertype
python minexp.py --dataset mnist
```

---

## 🚀 Estratégia de Execução Recomendada

### Opção 1: Tabela Rápida (Apenas Datasets Comuns - 7 datasets)
**Tempo Total: ~1-2 horas**

1. Re-executar PEAB, Anchor e MinExp para os 7 datasets rápidos
2. Gerar tabela com valores atualizados e desvio padrão
3. Deixar Credit Card, Covertype e MNIST para depois

```bash
# Executar em sequência (ou em paralelo se tiver GPU/CPU suficiente)
for dataset in banknote vertebral_column pima_indians_diabetes heart_disease breast_cancer sonar spambase; do
    echo "Processando $dataset..."
    python peab.py --dataset $dataset
    python anchor.py --dataset $dataset
    python minexp.py --dataset $dataset
done

# Gerar tabela
python temporarios/gerar_tabela_runtime_unificada.py
```

### Opção 2: Tabela Completa (10 datasets)
**Tempo Total: ~10-15 horas (deixar rodando overnight)**

1. Re-executar todos os datasets
2. Priorizar PEAB e MinExp (mais rápidos)
3. Deixar Anchor (creditcard, covertype, mnist) rodando à noite

```bash
# Dia 1: PEAB e MinExp (todos) + Anchor (rápidos)
python peab.py --dataset banknote
python peab.py --dataset vertebral_column
# ... (todos os datasets PEAB)

python minexp.py --dataset banknote
# ... (todos os datasets MinExp)

python anchor.py --dataset banknote
# ... (apenas datasets rápidos Anchor)

# Dia 2 (overnight): Anchor (demorados)
python anchor.py --dataset creditcard
python anchor.py --dataset covertype
python anchor.py --dataset mnist

# Gerar tabela final
python temporarios/gerar_tabela_runtime_unificada.py
```

---

## ✨ Resultado Esperado

Após re-executar os scripts, os JSONs terão:

```json
{
  "per_instance": [
    {
      "id": "0",
      "y_true": 0,
      "y_pred": 0,
      "rejected": false,
      "decision_score": -0.443,
      "explanation": ["feature1", "feature2"],
      "explanation_size": 2,
      "computation_time": 0.00123  // ← AGORA TODOS TÊM ISSO!
    },
    ...
  ],
  "computation_time": {
    "total": 25.5,
    "mean_per_instance": 0.062,
    "positive": 0.055,
    "negative": 0.048,
    "rejected": 0.089
  }
}
```

A tabela final terá desvio padrão:

```latex
Banknote & 5.6 $\pm$ 1.2 & 40.8 $\pm$ 5.3 & ...
```

---

## 📊 Verificação Rápida

Após re-executar um dataset, verifique se o JSON tem `computation_time` por instância:

```bash
python -c "import json; d=json.load(open('json/anchor/banknote.json')); print('Has per_instance:', 'per_instance' in d); print('Has computation_time:', 'computation_time' in d.get('per_instance', [{}])[0] if d.get('per_instance') else False)"
```

**Esperado**: `Has per_instance: True` e `Has computation_time: True`

---

## 🎯 Próximos Passos

1. ✅ Escolher estratégia (Opção 1 ou 2)
2. ✅ Re-executar scripts conforme escolha
3. ✅ Executar `temporarios/gerar_tabela_runtime_unificada.py`
4. ✅ Verificar tabela gerada em `results/tabelas_latex/runtime_unified_with_std.tex`
5. ✅ Copiar tabela LaTeX para artigo

---

**Data**: 7 de fevereiro de 2026
**Status**: Correções implementadas, pronto para re-execução
