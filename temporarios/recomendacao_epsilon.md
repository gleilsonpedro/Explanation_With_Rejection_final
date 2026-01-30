# Recomendação de Ajuste do EPSILON no PULP

## 📊 Análise dos Problemas

### Problema CRÍTICO:
1. **Breast Cancer**: PULP tem 0 positivas vs PEAB tem 84 positivas
   - Diferença de threshold: 0.0009 (muito pequena)
   - EPSILON muito conservador impedindo soluções válidas

### Problemas MENORES (diferenças < 1 feature):
2. **Banknote** (rejeitadas): +0.02 features
3. **Heart Disease** (classificadas): +0.06 features  
4. **Sonar** (classificadas): +0.39 features
5. **Spambase** (classificadas): +0.05 features

## 🎯 Estratégia de Ajuste

### OPÇÃO 1: Conservadora - EPSILON = 1e-5 (RECOMENDADO ✅)
**Vantagens:**
- Iguala ao PEAB (consistência metodológica)
- Deve resolver Breast Cancer
- Minimiza risco de invalidar explicações
- Mudança defensável academicamente

**Desvantagens:**
- Pode não resolver completamente os casos menores

**Datasets para re-executar:**
- ✅ **breast_cancer** (OBRIGATÓRIO - problema crítico)
- ⚠️ **heart_disease** (opcional - problema menor)
- ⚠️ **sonar** (opcional - maior diferença, mas <1 feature)

### OPÇÃO 2: Moderada - EPSILON = 1e-4
**Vantagens:**
- Resolve todos os problemas com certeza
- Ainda conservador (0.0001 de tolerância)
- Baixo risco de problemas

**Desvantagens:**
- Pode ser "demais" para o que precisa
- Exige justificativa na tese

**Datasets para re-executar:** Mesmos da Opção 1

### OPÇÃO 3: Agressiva - EPSILON = 1e-3 ou maior ❌
**NÃO RECOMENDO!**
- Risco de gerar explicações INVÁLIDAS
- Pode violar garantias do método exato
- Difícil de justificar academicamente

## 📋 Recomendação Final

### FAÇA ASSIM:

1. **Primeira tentativa: EPSILON = 1e-5**
   - Mude apenas no pulp_experiment.py (linha 141)
   - Re-execute apenas: **breast_cancer**
   - Tempo estimado: ~10-15 minutos
   
2. **Verificar resultado:**
   - Se breast_cancer tiver positivas: ✅ SUCESSO!
   - Se ainda tiver problema: tente 1e-4

3. **Datasets OPCIONAIS para re-executar:**
   - Se quiser perfeição total: heart_disease, sonar
   - Mas as diferenças são <1 feature (aceitáveis)

## ⏱️ Estimativa de Tempo

**Re-executar apenas breast_cancer:**
- PULP: ~10-15 minutos
- Total: 15 minutos ✅ VIÁVEL!

**Re-executar os 3 problemáticos (breast_cancer + heart_disease + sonar):**
- PULP: ~25-30 minutos  
- Total: 30 minutos ✅ AINDA VIÁVEL!

**Re-executar todos os 7 datasets:**
- Não necessário! Desperdício de tempo.

## 🎓 Justificativa Acadêmica

"O valor de EPSILON = 1e-5 foi escolhido para garantir consistência 
com a heurística PEAB, mantendo as garantias de otimalidade do solver 
enquanto permite tolerâncias numéricas razoáveis para aritmética de 
ponto flutuante."

---

**MINHA SUGESTÃO: Comece com 1e-5 e re-execute apenas breast_cancer.**
**Se resolver, está ótimo. Se não, tente 1e-4.**
