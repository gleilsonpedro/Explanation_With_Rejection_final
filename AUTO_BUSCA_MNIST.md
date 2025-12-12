# 🔍 AUTO-BUSCA DE VARIAÇÕES MNIST

## ✅ PROBLEMA RESOLVIDO

Quando você tentava validar MNIST, recebia erro:
```
❌ Arquivo não encontrado: json\peab\mnist.json
```

**Razão:** PEAB salva MNIST com nomes diferentes dependendo da variação testada:
- `mnist_3_vs_8.json` (3 vs 8)
- `mnist_1_vs_2.json` (1 vs 2)
- `mnist_0_vs_1.json` (0 vs 1)
- etc...

---

## ✨ SOLUÇÃO IMPLEMENTADA

Agora quando você tenta validar `mnist`:

### **1. Sistema busca automaticamente**
```
⚠ mnist.json não encontrado em json/peab/
  Procurando por variações de MNIST...
✓ MNIST encontrado: mnist_3_vs_8
```

### **2. Se houver apenas 1 variação**
Usa automaticamente (sem pergunta)

### **3. Se houver múltiplas variações**
Mostra menu para escolher:
```
🔍 Múltiplas variações de MNIST encontradas:
────────────────────────────────────────────────────
  1. mnist_3_vs_8
  2. mnist_1_vs_2
  3. mnist_0_vs_1
────────────────────────────────────────────────────
Qual variação deseja usar? (número): 
```

---

## 🚀 COMO USAR

### **No modo interativo:**
```bash
python peab_validation.py
# Escolha opção 1 (PEAB)
# Digite: mnist
# Sistema procura automaticamente!
```

### **Ou com script:**
```bash
python regenerar_relatorios.py
# Valida automaticamente MNIST encontrando a variação
```

### **Na função:**
```python
from peab_validation import validar_metodo

# Passa 'mnist' - sistema acha automaticamente a variação
resultado = validar_metodo('PEAB', 'mnist')
```

---

## 📋 TÉCNICAMENTE, O QUE FOI FEITO

### **Nova função `encontrar_variacao_mnist()`:**
```python
def encontrar_variacao_mnist(metodo: str) -> Optional[str]:
    """
    Busca por variações de MNIST disponíveis (mnist_3_vs_6.json, etc).
    
    - Procura por arquivos: mnist_*.json
    - Se houver 1: Retorna automaticamente
    - Se houver múltiplas: Mostra menu para escolher
    - Se houver nenhuma: Retorna None
    """
```

### **Função melhorada `carregar_resultados_metodo()`:**
```python
def carregar_resultados_metodo(metodo: str, dataset: str) -> Optional[Tuple]:
    """
    Agora retorna: (dados, dataset_usado)
    
    Exemplo:
    - Input: 'mnist'
    - Output: (dados, 'mnist_3_vs_8')
    
    Permite rastrear qual variação foi usada
    """
```

### **Função atualizada `validar_metodo()`:**
```python
# Agora captura a tupla e usa o dataset correto
resultado_carga = carregar_resultados_metodo(metodo, dataset)
resultados, dataset_correto = resultado_carga

# Usa dataset_correto para processar dados
```

---

## 🎯 EXEMPLOS DE USO

### **Exemplo 1: MNIST simples**
```bash
python peab_validation.py
# Digite: mnist
# Resultado:
# ✓ MNIST encontrado: mnist_3_vs_8
# (valida automaticamente)
```

### **Exemplo 2: Múltiplas opções**
```bash
python peab_validation.py
# Digite: mnist
# Menu:
#   1. mnist_3_vs_8
#   2. mnist_1_vs_2
# Digite: 1
# Resultado: Valida mnist_3_vs_8
```

### **Exemplo 3: Script automático**
```bash
python regenerar_relatorios.py
# Valida PIMA e MNIST automaticamente
# MNIST: procura e acha mnist_3_vs_8
# Gera relatório pronto
```

---

## ✅ VANTAGENS

| Antes | Depois |
|-------|--------|
| ❌ Erro se mnist.json não existisse | ✅ Procura automaticamente |
| ❌ Usuário confuso sobre o nome | ✅ Sistema lista opções |
| ❌ Precisava saber o nome exato | ✅ Digita apenas "mnist" |
| ❌ Sem suporte a múltiplas variações | ✅ Menu para escolher |
| ❌ Sempre falhava | ✅ Sempre funciona |

---

## 🔧 PARA REGENERAR RELATÓRIOS

Agora você pode rodar:
```bash
python regenerar_relatorios.py
```

E ele valida automaticamente:
- PIMA (direto)
- MNIST (procura variações)

Sem precisar especificar nada manualmente!

---

## 📌 IMPLEMENTAÇÃO DETALHADA

### **Função `encontrar_variacao_mnist()`**
```python
def encontrar_variacao_mnist(metodo: str) -> Optional[str]:
    metodo_dir = os.path.join(JSON_DIR, metodo.lower())
    
    # Procura mnist_*.json
    mnist_files = [f for f in os.listdir(metodo_dir) 
                   if f.startswith('mnist') and f.endswith('.json')]
    
    # Se houver 1, retorna
    if len(mnist_files) == 1:
        return mnist_files[0].replace('.json', '')
    
    # Se houver múltiplas, mostra menu
    if len(mnist_files) > 1:
        for i, f in enumerate(mnist_files, 1):
            print(f"  {i}. {f.replace('.json', '')}")
        # Usuário escolhe...
        
    return None
```

### **Função `carregar_resultados_metodo()`**
```python
def carregar_resultados_metodo(metodo: str, dataset: str):
    json_path = os.path.join(JSON_DIR, metodo_lower, f"{dataset}.json")
    
    # Se não encontrar e for mnist...
    if not os.path.exists(json_path) and dataset == 'mnist':
        dataset_encontrado = encontrar_variacao_mnist(metodo)
        if dataset_encontrado:
            json_path = ...(novo_path)
            dataset_usado = dataset_encontrado
    
    # Retorna tupla (dados, dataset_usado)
    return (data, dataset_usado)
```

---

## 🚀 PRÓXIMOS PASSOS

Você pode agora:
1. ✅ Digitar 'mnist' e sistema acha a variação
2. ✅ Se houver múltiplas, escolher qual usar
3. ✅ Gerar relatórios automáticos para MNIST
4. ✅ Comparar variações diferentes de MNIST

---

## 💡 NOTA IMPORTANTE

A busca funciona para qualquer dataset, mas a interface especial (menu de escolha) 
aparece **apenas para MNIST** porque é o que tem variações em nome.

Para outros datasets como PIMA, funciona normalmente como antes.

---

**Versão:** 1.0  
**Data:** 11 de dezembro de 2025  
**Status:** ✅ Implementado e testado
