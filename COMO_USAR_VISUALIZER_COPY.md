# Como Usar o visualizer copy.py

## 📌 Objetivo

O `visualizer copy.py` gera **3 imagens individuais** mostrando exemplos reais de explicações geradas pelo método PEAB para o experimento MNIST 3 vs 8:

1. **Exemplo Positiva**: Uma instância da classe 8 (positiva) corretamente classificada
2. **Exemplo Negativa**: Uma instância da classe 3 (negativa) corretamente classificada
3. **Exemplo Rejeitada**: Uma instância onde o PEAB detectou evidências conflitantes e rejeitou

## 🔧 O que foi corrigido

### Problemas encontrados no código original:
1. ❌ JSON inválido embutido no código (sintaxe JavaScript em Python)
2. ❌ Tentativa de processar string JSON em vez do arquivo real
3. ❌ Código muito complexo e genérico (não focado no objetivo específico)
4. ❌ Mistura de análise estatística com visualização
5. ❌ Não gerava as imagens individuais solicitadas

### Solução implementada:
✅ Script limpo e focado em gerar 3 imagens individuais  
✅ Carrega corretamente o `json/comparative_results.json`  
✅ Usa índice sequencial correto (não o ID original do MNIST)  
✅ Mostra dígito original + overlay colorido da explicação  
✅ Cores distintas por categoria (Azul=Positiva, Vermelho=Negativa, Roxo=Rejeitada)  
✅ Salva em `analysis_output/plots/individual_examples/`  

## 🚀 Como executar

### Execução básica (padrão):
```cmd
env\Scripts\python.exe "visualizer copy.py"
```

### Com opções personalizadas:
```cmd
env\Scripts\python.exe "visualizer copy.py" --experiment mnist --results json\comparative_results.json
```

### Parâmetros disponíveis:
- `--experiment`: Nome do experimento (padrão: `mnist`)
- `--results`: Caminho do JSON (padrão: `json/comparative_results.json`)
- `--show`: Mostra janelas interativas do matplotlib (padrão: apenas salva)

### Exemplo com visualização interativa:
```cmd
env\Scripts\python.exe "visualizer copy.py" --show
```

### 🎲 Seleção aleatória de exemplos (NOVO!):

Por padrão, o script agora **seleciona aleatoriamente** um exemplo de cada categoria:

```cmd
# Execução 1: Pega um 8 aleatório
env\Scripts\python.exe "visualizer copy.py"

# Execução 2: Pega OUTRO 8 diferente
env\Scripts\python.exe "visualizer copy.py"
```

**Benefícios:**
- ✅ Você pode rodar várias vezes até encontrar um dígito 8 bonito
- ✅ Útil quando o primeiro exemplo é "torto" ou mal escrito
- ✅ Explora a diversidade das instâncias

### 🔒 Fixar índices específicos (RECOMENDADO!):

Existem **2 formas** de fixar índices:

#### ✅ **Forma 1: Editando o código (MAIS FÁCIL)**

Abra o arquivo `visualizer copy.py` e edite as linhas 23-25:

```python
IDX_POSITIVA = 104    # ← Mude aqui!
IDX_NEGATIVA = 14     # ← Mude aqui!
IDX_REJEITADA = 13    # ← Mude aqui!
```

Depois rode normalmente:
```cmd
env\Scripts\python.exe "visualizer copy.py"
```

**Vantagens:**
- ✅ Não precisa digitar --idx toda vez
- ✅ Mais fácil de lembrar
- ✅ Fica permanente no código

#### ✅ **Forma 2: Passando parâmetros na linha de comando**

A melhor forma de reproduzir os mesmos exemplos é usar os **índices exatos** na execução:

```cmd
# Fixar apenas o dígito 8 (positiva)
env\Scripts\python.exe "visualizer copy.py" --idx-positiva 104

# Fixar o dígito 3 (negativa)
env\Scripts\python.exe "visualizer copy.py" --idx-negativa 14

# Fixar a rejeitada
env\Scripts\python.exe "visualizer copy.py" --idx-rejeitada 13

# Fixar TODOS os 3 índices ao mesmo tempo
env\Scripts\python.exe "visualizer copy.py" --idx-positiva 104 --idx-negativa 14 --idx-rejeitada 13
```

**Como descobrir o índice?**
1. Rode sem parâmetros: `env\Scripts\python.exe "visualizer copy.py"`
2. Olhe o dígito gerado nas imagens
3. Anote o `idx` que apareceu no console (ex: `idx=104`)
4. Use esse índice na próxima execução!

**Vantagens:**
- ✅ Não precisa editar o código
- ✅ Útil para testar índices diferentes rapidamente
- ✅ Parâmetros da linha de comando têm prioridade sobre o código

**Qual forma escolher?**
- Use **Forma 1 (código)** se você já sabe os índices e quer deixar fixo
- Use **Forma 2 (linha de comando)** se está testando índices diferentes

### 🎲 Alternativa: Reproduzir com seed aleatória:

Se você não quer escolher índices específicos, pode usar seed:

```cmd
# Usando seed 42, sempre pega os mesmos exemplos
env\Scripts\python.exe "visualizer copy.py" --seed 42

# Outras seeds geram outras combinações
env\Scripts\python.exe "visualizer copy.py" --seed 123
```

**Limitação:** A seed gera uma **combinação** aleatória, mas você não controla qual índice específico vai sair.

### 📊 Estatísticas mostradas:

O script agora mostra quantos candidatos existem:
```
📊 Candidatos disponíveis:
  • Positivas: 52 instâncias
  • Negativas: 60 instâncias
  • Rejeitadas: 14 instâncias
```

Isso significa que há **52 dígitos 8 diferentes** para escolher!

## 📂 Arquivos gerados

As imagens são salvas em:
```
analysis_output/plots/individual_examples/
├── mnist_exemplo_positiva.png    ← Instância classe 8 (correta)
├── mnist_exemplo_negativa.png    ← Instância classe 3 (correta)
└── mnist_exemplo_rejeitada.png   ← Instância rejeitada (conflito)
```

## 🎨 Estrutura de cada imagem

Cada imagem tem **2 painéis lado a lado**:

**Painel Esquerdo**: Dígito original (28×28 em escala de cinza)
- Mostra a classe verdadeira

**Painel Direito**: Overlay da explicação PEAB
- Pixels destacados = features que compõem a explicação mínima
- Cor do overlay indica a categoria:
  - 🔵 **Azul**: Classe Positiva (8)
  - 🔴 **Vermelho**: Classe Negativa (3)
  - 🟣 **Roxo**: Rejeitada
- Mostra: classe predita, score de decisão, número de pixels na explicação

## 📊 Informações exibidas

Para cada imagem:
- **Classe Verdadeira**: O rótulo correto do dígito
- **Predito**: Classe prevista pelo modelo (0=classe 3, 1=classe 8)
- **Score**: Score de decisão do modelo (distância à fronteira)
- **Pixels na explicação**: Quantos pixels compõem a explicação mínima

## 💡 Diferença entre os dois visualizers

| Recurso | `visualizer.py` | `visualizer copy.py` |
|---------|----------------|---------------------|
| Objetivo | Análise agregada (médias por classe) | Exemplos individuais |
| Imagens geradas | 1 figura com 3 painéis (médias) | 3 imagens separadas |
| Tipo | Mapas de calor agregados | Dígitos individuais + overlay |
| Uso | Entender padrão geral | Mostrar exemplos concretos |

## 🎯 Para que usar este visualizer

Use `visualizer copy.py` quando quiser:
- ✅ Mostrar exemplos visuais concretos das explicações do PEAB
- ✅ Ilustrar o que significa "explicação mínima" em um caso real
- ✅ Demonstrar visualmente por que uma instância foi rejeitada
- ✅ Preparar figuras para apresentações ou artigos
- ✅ Validar manualmente que as explicações fazem sentido

Use `visualizer.py` quando quiser:
- 📊 Ver tendências gerais por classe
- 📈 Comparar padrões médios entre positivas/negativas
- 🔬 Análise quantitativa agregada

## 🐛 Resolução de problemas

### Erro: "Arquivo não encontrado"
Verifique se `json/comparative_results.json` existe:
```cmd
dir json\comparative_results.json
```

### Erro: "Experimento não encontrado"
Liste os experimentos disponíveis e escolha um válido.

### Nenhuma instância encontrada
Certifique-se que o JSON contém o campo `per_instance` com dados.

### Imagens em branco
Verifique se as instâncias têm o campo `explanation` preenchido.

## 💡 Fluxo de trabalho recomendado

### Passo 1: Explorar opções
```cmd
# Rode várias vezes para ver diferentes exemplos
env\Scripts\python.exe "visualizer copy.py"
env\Scripts\python.exe "visualizer copy.py"
env\Scripts\python.exe "visualizer copy.py"
```

### Passo 2: Anotar os bons índices
Quando encontrar exemplos que você goste, anote os `idx` do console:
```
🎨 Gerando imagens individuais...
  • Positiva (idx=104, id=30743)    ← Anote: 104
  • Negativa (idx=14, id=16849)     ← Anote: 14
  • Rejeitada (idx=13, id=16750)    ← Anote: 13
```

### Passo 3: Fixar para sempre
```cmd
# Use os índices que você anotou
env\Scripts\python.exe "visualizer copy.py" --idx-positiva 104 --idx-negativa 14 --idx-rejeitada 13
```

**Resultado:** Agora você tem controle total sobre quais exemplos aparecem! 🎯

Você pode fixar apenas um índice (ex: só o 8 bonito) e deixar os outros aleatórios:
```cmd
# Fixa só a positiva, resto é aleatório
env\Scripts\python.exe "visualizer copy.py" --idx-positiva 104
```

## ✅ Resumo

O script está **100% funcional** e gera as 3 imagens solicitadas automaticamente. Basta executar:

```cmd
env\Scripts\python.exe "visualizer copy.py"
```

E as imagens aparecerão em `analysis_output/plots/individual_examples/`.
