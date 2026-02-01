# Solução: Validação Anchor e MinExp

## Problema Identificado

Os JSONs do **Anchor** e **MinExp** para MNIST não contêm as explicações individuais necessárias para validação.

```
✅ PEAB e PULP: Têm "per_instance" com todas explicações
❌ Anchor e MinExp: Só têm estatísticas agregadas
```

## Por Que Isso Aconteceu?

Os scripts `anchor.py` e `minexp.py` **criam** a lista `per_instance` internamente mas **não salvam** no JSON final. Eles só salvam estatísticas agregadas para economizar espaço em disco.

## Opções de Solução

### Opção 1: NÃO FAZER NADA (Recomendado) ⭐

**Razão**: A validação de Anchor/MinExp é **opcional** e não afeta:
- ✅ Tabelas de speedup (já funcionando)
- ✅ Tabelas de tamanho de explicações (já funcionando)  
- ✅ Comparações de tempo (já funcionando)

**O que você perde**:
- ❌ Métricas de "necessidade" e "redundância" do Anchor/MinExp
- ❌ Tabelas de validação comparativas

**Vantagem**: Não precisa reprocessar nada (economiza ~2-3 horas)

---

### Opção 2: Adicionar flag para salvar explicações individuais

Modificar `anchor.py` e `minexp.py` para aceitar uma flag `--save-individual` que salva as explicações completas.

**Passos**:
1. Modificar código para incluir `per_instance` no dataset_cache
2. Reexecutar: `python anchor.py` e `python minexp.py`
3. Executar: `python peab_validation.py`

**Tempo estimado**: 
- Anchor MNIST: ~45-60 minutos
- MinExp MNIST: ~30-45 minutos  
- Validação: ~15-20 minutos cada
- **Total: ~2-3 horas**

**Vantagens**:
- ✅ Validação completa
- ✅ Todas as métricas disponíveis

**Desvantagens**:
- ❌ JSONs muito grandes (~50-100 MB cada)
- ❌ Tempo de reprocessamento

---

### Opção 3: Usar apenas PEAB vs PULP na validação

Focar nas comparações PEAB vs PULP (que já estão completas) e ignorar Anchor/MinExp na validação.

**Vantagem**: Comparação justa (ambos são métodos de otimização)

---

## Minha Recomendação

**Use Opção 1** (não fazer nada) SE:
- Você já tem as tabelas principais de speedup e explicações
- Não precisa das métricas de necessidade/redundância do Anchor/MinExp
- Quer economizar tempo

**Use Opção 2** (modificar scripts) SE:
- Precisa de validação completa para todos os métodos
- Vai usar as métricas de necessidade/redundância no artigo
- Tem tempo disponível (~3 horas)

---

## Como Modificar os Scripts (Se escolher Opção 2)

Vou criar um script que modifica automaticamente o anchor.py e minexp.py para salvarem per_instance.

Diga se quer que eu crie esse script modificador.
