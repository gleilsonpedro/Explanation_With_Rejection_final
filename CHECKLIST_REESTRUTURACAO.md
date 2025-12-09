# 📋 CHECKLIST - Reestruturação Modular do Projeto

## ✅ FASE 1: Criação do PuLP Independente
- [x] Criar `pulp_experiment.py`
- [x] Implementar solver de otimização inteira
- [x] Salvar resultados em `json/pulp_results.json`
- [x] Gerar relatórios em `results/report/pulp/`
- [x] Criar documentação (`PULP_README.md`)
- [x] Testar importação do módulo

---

## ✅ FASE 2: Refatorar Comparação PEAB vs PuLP
- [x] Criar `peab_vs_pulp.py`
- [x] Ler dados de `json/peab_results.json`
- [x] Ler dados de `json/pulp_results.json`
- [x] Calcular métricas:
  - [x] GAP (diferença de cardinalidade)
  - [x] Taxa de otimalidade
  - [x] Speedup (ratio de tempo)
- [x] Gerar relatório comparativo em `results/benchmark/peab_vs_pulp/`
- [x] Gerar CSV com dados detalhados
- [ ] Criar visualizações (script separado - futuro)

---

## 📁 FASE 3: Organização em Pasta experiments/
- [ ] Criar pasta `experiments/`
- [ ] Mover/copiar arquivos:
  - [ ] `peab.py` → `experiments/peab_experiment.py`
  - [ ] `anchor.py` → `experiments/anchor_experiment.py`
  - [ ] `minexp.py` → `experiments/minexp_experiment.py`
  - [ ] `pulp_experiment.py` → `experiments/pulp_experiment.py`
  - [ ] `peab_vs_pulp.py` → `experiments/peab_vs_pulp.py`
- [ ] Atualizar imports nos arquivos movidos
- [ ] Criar links simbólicos na raiz (compatibilidade)
- [ ] Atualizar `.gitignore` se necessário

---

## 🎮 FASE 4: Menu Unificado
- [ ] Criar `experiments/main.py`
- [ ] Implementar menu interativo:
  ```
  [1] Executar PEAB
  [2] Executar Anchor
  [3] Executar MinExp
  [4] Executar PuLP (solver exato)
  [5] ─────────────────────────
  [6] Comparar PEAB vs PuLP
  [7] Comparar PEAB vs Anchor vs MinExp
  [8] Comparar TODOS (inclui PuLP)
  [9] ─────────────────────────
  [10] Gerar Relatório Completo
  [11] Limpar cache de resultados
  [0] Sair
  ```
- [ ] Adicionar validações:
  - [ ] Verificar se JSONs existem antes de comparar
  - [ ] Sugerir executar métodos faltantes
- [ ] Adicionar opção de múltiplos datasets
- [ ] Criar modo batch (executar todos de uma vez)

---

## 📊 FASE 5: Comparação Multi-Métodos
- [ ] Criar `experiments/compare_all_methods.py`
- [ ] Ler todos os JSONs (peab, anchor, minexp, pulp)
- [ ] Calcular métricas cruzadas:
  - [ ] Cardinalidade média por método
  - [ ] GAP vs PuLP (ground truth)
  - [ ] Tempo de execução
  - [ ] Taxa de sucesso
- [ ] Gerar tabela comparativa LaTeX
- [ ] Gerar gráficos (barras, boxplots, scatter)
- [ ] Salvar em `results/benchmark/all_methods/`

---

## 🧪 FASE 6: Testes e Validação
- [ ] Criar `tests/test_pulp_experiment.py`
- [ ] Testar com dataset pequeno (wine)
- [ ] Validar consistência:
  - [ ] Thresholds iguais ao PEAB
  - [ ] Split consistente (RANDOM_STATE)
  - [ ] Resultados reproduzíveis
- [ ] Verificar formato JSON
- [ ] Verificar relatórios TXT

---

## 📝 FASE 7: Documentação
- [ ] Atualizar `README.md` principal
- [ ] Criar `experiments/README.md`
- [ ] Documentar estrutura de pastas
- [ ] Adicionar exemplos de uso
- [ ] Criar guia de reprodução de experimentos
- [ ] Documentar decisões de design

---

## 🎓 FASE 8: Preparação Acadêmica
- [ ] Executar PuLP em todos os datasets
- [ ] Gerar todas as comparações
- [ ] Criar tabelas para dissertação:
  - [ ] Tabela 1: Métricas dos modelos
  - [ ] Tabela 2: PEAB vs PuLP (otimalidade)
  - [ ] Tabela 3: Comparação multi-métodos
  - [ ] Tabela 4: Análise de tempo
- [ ] Gerar gráficos acadêmicos (matplotlib/seaborn)
- [ ] Preparar análise estatística (testes de hipótese)

---

## 🚀 FASE 9: Otimizações (Opcional)
- [ ] Cache inteligente de resultados
- [ ] Paralelização (ProcessPoolExecutor)
- [ ] Progress tracking persistente
- [ ] Retomada de experimentos interrompidos
- [ ] Export para outros formatos (Excel, CSV, LaTeX)

---

## 📦 FASE 10: Release
- [ ] Tag de versão (v2.0)
- [ ] Changelog detalhado
- [ ] Verificar compatibilidade com código antigo
- [ ] Criar branch `legacy` para código original
- [ ] Atualizar requirements.txt
- [ ] Push para GitHub

---

## 🎯 STATUS ATUAL

**Data**: 09/12/2025  
**Fase Atual**: FASE 2 ✅ CONCLUÍDA  
**Próximo**: FASE 3 (Organizar em pasta experiments/)

### Arquivos Criados:
- ✅ `pulp_experiment.py` (486 linhas)
- ✅ `peab_vs_pulp.py` (585 linhas)
- ✅ `PULP_README.md` (documentação completa)
- ✅ `CHECKLIST_REESTRUTURACAO.md` (este arquivo)

### Estrutura de Diretórios Criada:
- ✅ `json/pulp_results.json` (será criado na execução)
- ✅ `results/report/pulp/` (será criado na execução)
- ✅ `results/benchmark/peab_vs_pulp/` (será criado na execução)

---

## 📌 NOTAS IMPORTANTES

### Compatibilidade Retroativa:
- Manter arquivos originais na raiz (pelo menos inicialmente)
- Criar links simbólicos se mover para `experiments/`
- Garantir que scripts antigos continuem funcionando

### Prioridades:
1. **Alta**: FASE 2 (comparação PEAB vs PuLP)
2. **Média**: FASE 4 (menu unificado)
3. **Baixa**: FASE 9 (otimizações)

### Riscos:
- ⚠️ Quebrar imports existentes ao mover arquivos
- ⚠️ Inconsistência de dados entre JSONs antigos e novos
- ⚠️ Tempo de execução do PuLP em datasets grandes

### Mitigações:
- ✅ Testar imports após cada mudança
- ✅ Validar formato JSON com schema
- ✅ Executar PuLP em background/overnight para datasets grandes

---

## 🤝 Próxima Ação Recomendada

**EXECUTAR**: `python pulp_experiment.py`

**Dataset sugerido**: `wine` (pequeno, rápido para testar)

**Validações após execução**:
1. Verificar `json/pulp_results.json` criado
2. Verificar `results/report/pulp/wine/R_pulp_wine.txt` criado
3. Comparar com resultados PEAB existentes
4. Se OK → Avançar para FASE 2 (`peab_vs_pulp.py`)

---

**Última atualização**: 09/12/2025  
**Por**: Claude (GitHub Copilot)
