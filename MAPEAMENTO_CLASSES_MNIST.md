# 📊 MAPEAMENTO DE CLASSES NO MNIST (3 vs 8)

## ✅ Resposta Direta:

**MNIST_SELECTED_PAIR = (3, 8)**

```python
class_names = ['3', '8']

• class_names[0] = '3' → NEGATIVA (y=0)
• class_names[1] = '8' → POSITIVA (y=1)
```

---

## 🔍 Como Funciona o Mapeamento

No código `data/datasets.py` (linhas 439-445):

```python
if MNIST_SELECTED_PAIR is not None:
    a, b = MNIST_SELECTED_PAIR  # a=3, b=8
    mask = (y_all == a) | (y_all == b)
    X = X[mask].copy()
    y_bin_np = np.where(y_all[mask] == a, 0, 1)  # ← AQUI!
    y_series = pd.Series(y_bin_np, index=X.index, name='target')
    class_names_list = [str(a), str(b)]
```

**Linha crítica:** `y_bin_np = np.where(y_all[mask] == a, 0, 1)`

- Se o dígito original é igual a `a` (3) → recebe label `0` (NEGATIVA)
- Caso contrário (é o dígito `b`, ou seja, 8) → recebe label `1` (POSITIVA)

---

## 📈 Interpretação do Decision Score

O `decision_score` do modelo LogisticRegression indica:

```
decision_score < t_minus (threshold negativo)  → Aceitar como NEGATIVA (classe 3)
decision_score > t_plus  (threshold positivo)  → Aceitar como POSITIVA (classe 8)
t_minus ≤ decision_score ≤ t_plus             → REJEITAR (incerto)
```

### Exemplos do JSON:

| Instância | y_true | y_pred | decision_score | Interpretação |
|-----------|--------|--------|----------------|---------------|
| 0 | 0 (3) | 0 (3) | -2.1323 | Score negativo → Prediz 3 (NEGATIVA) ✓ |
| 1 | 1 (8) | 1 (8) | +3.6839 | Score positivo → Prediz 8 (POSITIVA) ✓ |
| 2 | 0 (3) | 0 (3) | -3.6436 | Score muito negativo → Prediz 3 ✓ |
| 3 | 1 (8) | 1 (8) | +0.4479 | Score ligeiramente positivo → Prediz 8 ✓ |

---

## 🎨 Interpretação dos Plots

Quando você vê nos plots gerados:

### **"NEGATIVA (Classe 3)"**
- Label: y=0
- Decision score: **negativo** (< 0)
- Dígito mostrado: **3**
- Cor do título: vermelho

### **"POSITIVA (Classe 8)"**
- Label: y=1  
- Decision score: **positivo** (> 0)
- Dígito mostrado: **8**
- Cor do título: azul

### **"REJEITADA"**
- Decision score: próximo de 0 (dentro da zona de rejeição)
- Pode ser tanto 3 quanto 8 (modelo incerto)
- Cor do título: roxo

---

## 🔧 Como Verificar em Qualquer Arquivo JSON

```python
import json

data = json.load(open('json/minexp/mnist.json', 'r'))  # ou peab, pulp, anchor
print(data['model']['class_names'])  # ['3', '8']

# class_names[0] = NEGATIVA (y=0)
# class_names[1] = POSITIVA (y=1)
```

---

## 📝 Resumo

| Dígito | Label (y) | Categoria | Decision Score |
|--------|-----------|-----------|----------------|
| **3** | 0 | NEGATIVA | < 0 (negativo) |
| **8** | 1 | POSITIVA | > 0 (positivo) |

Essa convenção é **consistente** em todos os métodos (PEAB, MinExp, PULP, Anchor) porque todos usam o mesmo `get_shared_pipeline()` e o mesmo processamento do dataset.
