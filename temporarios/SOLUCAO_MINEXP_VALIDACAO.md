# Problema: MinExp não valida - Falta 'per_instance'

## 🔴 PROBLEMA IDENTIFICADO

O arquivo `json/minexp/vertebral_column.json` (e outros 9 datasets) **NÃO contém** a chave `'per_instance'` com as explicações individuais.

Estes arquivos foram gerados com uma **versão antiga** do código MinExp que salvava apenas estatísticas agregadas, sem as explicações por instância.

## 📊 STATUS DOS DATASETS

Dos 11 datasets MinExp:
- ✅ **1 dataset OK** (mnist) - tem `per_instance`
- ❌ **10 datasets com problema** - falta `per_instance`:
  1. banknote
  2. breast_cancer
  3. covertype
  4. creditcard
  5. heart_disease
  6. pima_indians_diabetes
  7. sonar
  8. spambase
  9. vertebral_column
  10. wine

## ✅ SOLUÇÃO

### Opção 1: Reprocessar apenas vertebral_column (RECOMENDADO para teste rápido)

```bash
# Execute:
python minexp.py

# Quando solicitar, escolha:
# Dataset: vertebral_column
```

Aguarde a conclusão (pode levar alguns minutos). O novo arquivo `json/minexp/vertebral_column.json` incluirá `per_instance`.

Depois execute a validação:
```bash
python peab_validation.py
```

### Opção 2: Reprocessar TODOS os datasets (para análise completa)

Execute para cada um dos 10 datasets:

```bash
python minexp.py
# Escolha: banknote
# Aguarde conclusão

python minexp.py
# Escolha: breast_cancer
# Aguarde conclusão

# ... e assim por diante para os outros 8
```

**Nota:** Alguns datasets (ex: covertype, creditcard, spambase) podem demorar bastante tempo.

### Opção 3: Usar script de verificação

Para verificar o status atual:
```bash
python temporarios/verificar_minexp_status.py
```

## 🔧 O QUE FOI CORRIGIDO

1. **Mensagem de erro melhorada** em `peab_validation.py`:
   - Agora mostra claramente que o arquivo foi gerado com versão antiga
   - Indica o caminho exato do arquivo com problema
   - Fornece instruções específicas de como resolver

2. **Scripts utilitários criados** em `temporarios/`:
   - `diagnosticar_minexp.py` - Diagnóstico completo
   - `test_minexp_load.py` - Teste de carregamento
   - `verificar_minexp_status.py` - Status de todos os datasets
   - `reprocessar_minexp_batch.py` - Tentativa de reprocessamento automático

## 📝 EXPLICAÇÃO TÉCNICA

O código atual do `minexp.py` **está correto** e inclui `per_instance` no salvamento:

```python
dataset_cache = {
    # ... outras chaves ...
    'per_instance': per_instance,  # ✅ ISSO ESTÁ NO CÓDIGO
    # ...
}
```

O problema é que os arquivos existentes em `json/minexp/` foram gerados **antes** dessa correção ser implementada.

A validação precisa de `per_instance` porque:
- Testa Fidelity, Necessity, Sufficiency para CADA explicação individual
- Gera perturbações específicas para cada instância
- Valida feature por feature de cada explicação
- Estatísticas agregadas não são suficientes para validação rigorosa

## 🎯 PRÓXIMOS PASSOS

1. Execute `python minexp.py` para `vertebral_column`
2. Execute `python peab_validation.py` e escolha MinExp + vertebral_column
3. A validação deve funcionar agora!

---

**Arquivos na pasta temporarios/**:
- ✅ Todos os scripts de diagnóstico e verificação foram criados
- ✅ Não bagunçaram a pasta raiz do projeto
- 🔍 Use-os para monitorar o status antes/depois do reprocessamento
