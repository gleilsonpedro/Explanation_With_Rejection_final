# 🚨 PROBLEMA CRÍTICO ENCONTRADO: Instâncias Rejeitadas Incorretas

**Data:** 12 de dezembro de 2025  
**Status:** ⚠️ REQUER CORREÇÃO URGENTE

---

## 📋 Resumo Executivo

**54% (24/44) das instâncias marcadas como "rejeitadas" NO estão na zona de rejeição!**

Isso explica completamente por que a fidelidade de predições rejeitadas é apenas ~23%.

---

## 🔍 O Que Foi Descoberto

### Teste Realizado
Verificamos se todas as 44 instâncias marcadas como `"rejected": true` no JSON estão realmente dentro da zona de rejeição definida como `[-0.1096, 0.0779]`.

### Resultado
```
❌ 24/44 instâncias NÃO estão na zona de rejeição
✓ 20/44 instâncias estão corretamente na zona
```

### Exemplos de Instâncias Incorretas

| ID  | Score Salvo | Score Norm | Na Zona? | Deveria Ser       |
|-----|-------------|------------|----------|-------------------|
| 417 | 0.4390      | 0.1467     | ❌ NÃO   | **Positiva** (1)  |
| 78  | 0.4038      | 0.1429     | ❌ NÃO   | **Positiva** (1)  |
| 558 | 0.4501      | 0.1478     | ❌ NÃO   | **Positiva** (1)  |
| 351 | -0.6082     | (muito baixo) | ❌ NÃO   | **Negativa** (0)  |

**Zona de rejeição:** [-0.1096, 0.0779]  
**Instâncias com score_norm > 0.0779:** Deveriam ser classificadas como POSITIVAS  
**Instâncias com score_norm < -0.1096:** Deveriam ser classificadas como NEGATIVAS

---

## 🎯 Por Que Isso Causa Baixa Fidelidade

### O Teste de Fidelidade
A validação verifica se, ao perturbar features NÃO explicadas, a instância **continua na zona de rejeição**.

### O Problema
Se uma instância tem score normalizado de **0.1467** (fora da zona que vai até 0.0779):
- Ela NÃO está rejeitada de fato
- Foi **marcada incorretamente** como rejeitada
- Quando perturbada, naturalmente sai da zona (porque já estava fora!)
- Resultado: **fidelidade baixa**

### Exemplo Concreto
```
Instância 417:
├─ Score normalizado: 0.1467
├─ Zona de rejeição: [-0.1096, 0.0779]
├─ Está na zona? NÃO! (0.1467 > 0.0779)
├─ Mas está marcada como: rejected=true
│
├─ Ao validar fidelidade:
│   ├─ Perturba features não explicadas
│   ├─ Espera que fique na zona [-0.1096, 0.0779]
│   ├─ Mas naturalmente fica em ~0.14-0.15 (onde sempre esteve!)
│   └─ Resultado: FALHA (0% de fidelidade)
│
└─ CONCLUSÃO: Instância está INCORRETAMENTE marcada como rejeitada
```

---

## 🐛 Onde Está o Bug?

O problema está na **inconsistência entre**:

1. **Thresholds usados no TREINO** (para encontrar t+ e t-)
2. **Thresholds usados no TESTE** (para classificar instâncias)

### Código Suspeito

Provavelmente em [utils/rejection_logic.py](utils/rejection_logic.py) ou no próprio [peab.py](peab.py), há uma diferença entre:

```python
# Durante o TREINO (correto)
t_plus_norm, t_minus_norm = encontrar_thresholds(...)  # Retorna em espaço normalizado

# Durante o TESTE (incorreto?)
scores = model.decision_function(X_test)  # Scores SEM normalização?
rejeitadas = (scores >= t_minus) & (scores <= t_plus)  # Compara ERRADO!
```

### Hipótese
Os thresholds estão sendo salvos em **espaço normalizado** mas aplicados a scores **não normalizados** (ou vice-versa).

---

## ✅ Como Corrigir

### 1. Localizar o Problema

Procurar em `peab.py` onde as instâncias são classificadas como rejeitadas:

```python
# Linha ~351 em peab.py
is_rejected = t_minus <= score_norm <= t_plus  # ← Verificar se score_norm está correto
```

### 2. Garantir Consistência

Durante classificação, DEVE:
1. Calcular score do modelo: `score = model.decision_function(X)`
2. **Normalizar** usando mesmos parâmetros do treino
3. Comparar com thresholds normalizados

```python
# CORRETO
score_raw = model.decision_function(X)
score_z = (score_raw - mean_score) / std_score
score_norm = score_z / max_abs
is_rejected = (t_minus <= score_norm <= t_plus)

# INCORRETO
score_raw = model.decision_function(X)
is_rejected = (t_minus <= score_raw <= t_plus)  # ← Compara raw com norm!
```

### 3. Re-executar PEAB

Após correção:
- Rodar `python peab.py` novamente
- Selecionar `pima_indians_diabetes`
- Verificar que TODAS rejeitadas estão na zona

### 4. Validar Correção

```bash
python verificar_normalizacao.py
```

Deve mostrar: `✓ Todas as instâncias rejeitadas estão corretamente na zona!`

---

## 📊 Impacto Esperado Após Correção

### Antes (Atual)
- 44 instâncias marcadas como rejeitadas
- Apenas 20 realmente na zona (45%)
- Fidelidade: ~23%

### Depois (Esperado)
- ~20 instâncias rejeitadas (apenas as que estão na zona)
- 100% na zona (por definição)
- Fidelidade: **~70-90%** (muito mais alta!)

### Por Quê?
Com as instâncias corretas:
- Elas **realmente** estão na zona ambígua
- Explicações do PEAB são adequadas
- Perturbações mantêm instâncias na zona
- Fidelidade alta como esperado

---

## 🎯 Validação da Hipótese Original

**Sua pergunta estava 100% correta:**
> "Nas rejeitadas o que se espera é que mesmo perturbando ainda continuem na zona de rejeição?"

**SIM!** E o problema não era com o conceito, mas com a **implementação**:
- Instâncias marcadas como rejeitadas NÃO estão na zona
- Por isso não conseguem "continuar" na zona (nunca estiveram!)
- Após correção, a fidelidade deve subir significativamente

---

## 📁 Arquivos de Investigação

Criados para diagnóstico:
- `investigar_rejeitadas.py` - Análise de perturbações
- `verificar_normalizacao.py` - Verificação de consistência

Execute após correção para validar!

---

## ✨ Próximos Passos

1. ☐ Corrigir código de classificação em `peab.py`
2. ☐ Re-executar PEAB em `pima_indians_diabetes`
3. ☐ Validar com `verificar_normalizacao.py`
4. ☐ Re-executar validação: `python peab_validation.py`
5. ☐ Esperar fidelidade ~70-90% para rejeitadas
6. ☐ Repetir para outros datasets

---

**Status:** ⚠️ AGUARDANDO CORREÇÃO DO BUG

*Documento gerado em: 12 de dezembro de 2025*
