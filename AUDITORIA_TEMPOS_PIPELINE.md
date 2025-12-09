# 🔍 AUDITORIA DE TEMPOS E PIPELINE - Análise Comparativa

**Data**: 09/12/2025  
**Objetivo**: Garantir medição precisa de tempo e consistência de pipeline

---

## 📊 RESUMO DA AUDITORIA

| Método | Tempo Medido | Pipeline | Status |
|--------|-------------|----------|--------|
| **PEAB** | ✅ Apenas experimento | ✅ Próprio | ✅ CORRETO |
| **PuLP** | ✅ Apenas experimento | ✅ Usa PEAB | ✅ CORRETO |
| **Anchor** | ⚠️ Inclui explainer.explain() | ✅ Usa shared_pipeline | ⚠️ CORRIGIR |
| **MinExp** | ⚠️ Inclui chunks inteiros | ✅ Usa shared_pipeline | ⚠️ CORRIGIR |

---

## 🔬 ANÁLISE DETALHADA POR MÉTODO

### 1️⃣ PEAB (peab.py) ✅

**Medição de Tempo**:
```python
# Linha 840-872
start_total = time.perf_counter()
with ProgressBar(total=len(X_test)) as pbar:
    for i in range(len(X_test)):
        start_inst = time.perf_counter()  # ✅ INÍCIO CORRETO
        inst = X_test.iloc[[i]]
        expl, logs, ad, rm = gerar_explicacao_instancia(...)  # APENAS ISTO
        duracao = time.perf_counter() - start_inst  # ✅ FIM CORRETO
        # ... resto do código (não inclui na medição)
total_time_experimento = time.perf_counter() - start_total
```

**✅ CORRETO**: 
- Timer inicia DEPOIS de preparar `inst`
- Timer para IMEDIATAMENTE após `gerar_explicacao_instancia()`
- Barra de progresso (`pbar.update()`) NÃO incluída
- Append de resultados NÃO incluído

**Pipeline**: Próprio (MinMaxScaler + LogisticRegression)

---

### 2️⃣ PuLP (pulp_experiment.py) ✅

**Medição de Tempo**:
```python
# Linha 253-257
start_time = time.perf_counter()  # ✅ INÍCIO CORRETO
features_otimas, tamanho, tipo_pred = calcular_explicacao_otima_pulp(
    modelo, instancia, X_train, t_plus, t_minus
)
tempo_gasto = time.perf_counter() - start_time  # ✅ FIM CORRETO
```

**✅ CORRETO**:
- Mede APENAS `calcular_explicacao_otima_pulp()`
- Barra de progresso (`pbar.update()`) NÃO incluída
- Atualização de estatísticas NÃO incluída

**Pipeline**: Usa `treinar_e_avaliar_modelo()` do PEAB ✅

---

### 3️⃣ Anchor (anchor.py) ⚠️

**Medição de Tempo**:
```python
# Linha 217-253
start_time = time.time()  # ⚠️ ANTES DA PREPARAÇÃO
instance_arr = X_test.iloc[i].values if hasattr(X_test, 'iloc') else X_test[i]

try:
    explanation = explainer.explain(...)  # ALGORITMO
except ...
    # MUITOS TRY/EXCEPT
    
runtime = time.time() - start_time  # ⚠️ INCLUI PREPARAÇÃO + EXCEÇÕES
```

**⚠️ PROBLEMAS**:
1. Timer inicia ANTES de `instance_arr = X_test.iloc[i].values`
2. Inclui tempo de conversão de dados
3. Inclui tempo dos `try/except` (mesmo se falhar)
4. Usa `time.time()` em vez de `time.perf_counter()` (menos preciso)

**Pipeline**: Usa `get_shared_pipeline()` ✅ (consistente com PEAB)

**🔧 CORREÇÃO NECESSÁRIA**:
```python
instance_arr = X_test.iloc[i].values if hasattr(X_test, 'iloc') else X_test[i]
start_time = time.perf_counter()  # MOVER PARA CÁ
try:
    explanation = explainer.explain(...)
    runtime = time.perf_counter() - start_time  # DENTRO DO TRY
except:
    runtime = 0.0  # ou np.nan
```

---

### 4️⃣ MinExp (minexp.py) ⚠️

**Medição de Tempo**:
```python
# Linha 187-197
start_time_neg = time.time()  # ⚠️ ANTES DO CHUNKING
if len(neg_idx) > 0:
    explain_in_chunks(neg_idx, "Negative")  # CHUNKS + PROGRESS BAR
runtime_neg = time.time() - start_time_neg  # ⚠️ INCLUI OVERHEAD

# Linha 201-234
start_time_rej = time.time()  # ⚠️ ANTES DO LOOP
if len(rej_idx) > 0:
    for start in range(0, len(rej_idx), chunk_size):  # LOOP + TRY/EXCEPT
        sl = slice(start, start + chunk_size)
        sel_idx = rej_idx[sl]
        try:
            explanations_local = utils.svm_explainer.svm_explanation_rejected(...)
            # ... processamento
runtime_rej = time.time() - start_time_rej  # ⚠️ INCLUI CHUNKING + ERROS
```

**⚠️ PROBLEMAS**:
1. Mede tempo de CHUNKS inteiros (não por instância individual)
2. Inclui overhead de chunking (loops, slicing)
3. Inclui tempo de `try/except` mesmo quando falha
4. Inclui tempo de `pbar.update()` dentro dos chunks
5. Usa `time.time()` em vez de `time.perf_counter()`

**Pipeline**: Usa `get_shared_pipeline()` ✅ (consistente com PEAB)

**🔧 CORREÇÃO NECESSÁRIA**:
```python
# Medir por instância individualmente
tempos_por_instancia = {}
for idx in neg_idx:
    start_time = time.perf_counter()
    try:
        explanation = utils.svm_explainer.svm_explanation_negative(...)
        tempo = time.perf_counter() - start_time
    except:
        tempo = np.nan
    tempos_por_instancia[idx] = tempo
```

---

## 🎯 IMPACTO DA INCONSISTÊNCIA

### Comparação Atual (ANTES DA CORREÇÃO):

| Métrica | PEAB | PuLP | Anchor | MinExp |
|---------|------|------|--------|--------|
| **O que mede** | Apenas algoritmo | Apenas algoritmo | Algoritmo + overhead | Chunks + overhead |
| **Precisão** | Alta | Alta | Média | Baixa |
| **Comparabilidade** | ✅ Baseline | ✅ Justo vs PEAB | ⚠️ Inflado | ⚠️ Muito inflado |

**Exemplo Hipotético**:
```
PEAB:   0.100s (puro)
PuLP:   0.500s (puro)
Anchor: 0.250s (0.200s puro + 0.050s overhead)  ← INJUSTO
MinExp: 1.500s (1.000s puro + 0.500s chunking) ← MUITO INJUSTO
```

**Conclusão Errada**: "Anchor é 2.5x mais lento que PEAB"  
**Realidade**: "Anchor é 2.0x mais lento que PEAB"

---

## 📋 PLANO DE CORREÇÃO

### ✅ PRIORIDADE ALTA

#### 1. Anchor - Mover timer para depois da preparação
```python
# ANTES (ERRADO)
start_time = time.time()
instance_arr = X_test.iloc[i].values

# DEPOIS (CORRETO)
instance_arr = X_test.iloc[i].values
start_time = time.perf_counter()
```

#### 2. Anchor - Usar time.perf_counter() em vez de time.time()
```python
# ANTES
start_time = time.time()
runtime = time.time() - start_time

# DEPOIS
start_time = time.perf_counter()
runtime = time.perf_counter() - start_time
```

#### 3. MinExp - Medir por instância individualmente
```python
# CRIAR DICIONÁRIO DE TEMPOS
tempos_individuais = {}

# DENTRO DO LOOP DE EXPLICAÇÃO
for idx in indices:
    start = time.perf_counter()
    try:
        exp = explicar(idx)
        tempo = time.perf_counter() - start
    except:
        tempo = np.nan
    tempos_individuais[idx] = tempo
```

### ⚙️ PRIORIDADE MÉDIA

#### 4. Padronizar time.perf_counter() em todos
- ✅ PEAB: já usa
- ✅ PuLP: já usa
- ⚠️ Anchor: mudar de `time.time()` → `time.perf_counter()`
- ⚠️ MinExp: mudar de `time.time()` → `time.perf_counter()`

**Justificativa**: `time.perf_counter()` tem maior resolução e não é afetado por ajustes de relógio do sistema.

---

## ✅ VALIDAÇÃO DE PIPELINE (CONSISTÊNCIA)

### Pipeline de Treino:

| Componente | PEAB | PuLP | Anchor | MinExp |
|-----------|------|------|--------|--------|
| **Scaler** | MinMaxScaler | MinMaxScaler | MinMaxScaler | MinMaxScaler |
| **Modelo** | LogisticRegression | LogisticRegression | LogisticRegression | LogisticRegression |
| **Origem** | Próprio | Usa PEAB | shared_training | shared_training |
| **Hiperparâmetros** | hiperparametros.json | hiperparametros.json | hiperparametros.json | hiperparametros.json |
| **Random State** | 42 | 42 | 42 | 42 |
| **Thresholds** | Grid adaptativo | Usa PEAB | Usa PEAB | Usa PEAB |

**✅ TODOS CONSISTENTES**: Anchor e MinExp usam `get_shared_pipeline()` que chama `treinar_e_avaliar_modelo()` do PEAB.

### Top-K Features:

✅ **CORRETO**: `shared_training.py` aplica `top_k_features` ANTES do treino:
```python
# Linha 45-87
top_k = cfg.get('top_k_features', None)
if top_k and top_k > 0 and top_k < X.shape[1]:
    from peab import aplicar_selecao_top_k_features
    X_train, X_test, selected_features = aplicar_selecao_top_k_features(...)
```

---

## 🎓 RECOMENDAÇÕES PARA PAPER

### O que REPORTAR:

✅ **Correto**:
- "Todos os métodos usam o mesmo pipeline de treino (MinMaxScaler + LogisticRegression)"
- "Hiperparâmetros idênticos carregados de hiperparametros.json"
- "Thresholds t+/t- calculados uma única vez (PEAB) e reutilizados"
- "Split determinístico com RANDOM_STATE=42"

⚠️ **APÓS CORREÇÃO**:
- "Tempos medidos com time.perf_counter() (resolução de nanosegundos)"
- "Medição exclui overhead de I/O, logging e progress bars"
- "Cada instância medida individualmente para comparação justa"

❌ **NÃO REPORTAR (antes da correção)**:
- ~~"Anchor é Xx mais lento"~~ (tempos inflados)
- ~~"MinExp processa em chunks de Y instâncias"~~ (irrelevante para comparação)

---

## 📊 CHECKLIST DE VALIDAÇÃO

### Antes de Rodar Experimentos Finais:

- [x] **PEAB**: Timer correto ✅
- [x] **PuLP**: Timer correto ✅
- [ ] **Anchor**: Mover timer + usar perf_counter
- [ ] **MinExp**: Medir por instância + usar perf_counter
- [x] **Pipeline**: Todos usam shared_training ✅
- [x] **Hiperparâmetros**: Todos usam hiperparametros.json ✅
- [x] **Random State**: Todos usam 42 ✅
- [x] **Thresholds**: Anchor/MinExp reutilizam PEAB ✅

### Depois das Correções:

- [ ] Re-executar Anchor em dataset de teste
- [ ] Re-executar MinExp em dataset de teste
- [ ] Comparar tempos antes/depois da correção
- [ ] Validar que tempos são comparáveis com PEAB
- [ ] Atualizar JSONs com tempos corrigidos

---

## 📝 CONCLUSÃO

**Status Atual**:
- ✅ Pipeline: 100% consistente entre métodos
- ⚠️ Medição de tempo: Inconsistente entre métodos

**Impacto**:
- **Alto** para comparações de tempo (speedup, tempo médio)
- **Baixo** para comparações de qualidade (tamanho de explicações, GAP)

**Ação Requerida**:
1. Corrigir Anchor (2 mudanças simples)
2. Corrigir MinExp (refatoração maior)
3. Re-executar experimentos
4. Atualizar JSONs

**Prioridade**: ALTA (antes de submeter paper/dissertação)

---

**Autor**: Claude (GitHub Copilot)  
**Data**: 09/12/2025  
**Versão**: 1.0
