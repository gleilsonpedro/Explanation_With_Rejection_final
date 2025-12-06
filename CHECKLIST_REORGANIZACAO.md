# 📋 CHECKLIST DETALHADO DE REORGANIZAÇÃO DO PROJETO
## Guia Passo a Passo para Organização Profissional

---

## 🎯 OVERVIEW DO PROCESSO

**Tempo estimado**: 2-3 horas  
**Abordagem**: Incremental e testada (fazer → testar → próximo passo)  
**Princípio**: Nunca deletar antes de confirmar que funciona

---

## 📦 FASE 0: PREPARAÇÃO E BACKUP (15 minutos)

### ✅ Passo 0.1: Fazer Backup Completo
```bash
# No diretório pai do projeto:
cd "c:\Users\gleilsonpedro\OneDrive\Área de Trabalho\PYTHON\MESTRADO\XAI"
# Criar cópia de segurança
xcopy "Explanation_With_Rejection_final" "Explanation_With_Rejection_final_BACKUP" /E /I /H
```
- [X] Backup criado e verificado
- [X] Conferir se backup tem todos os arquivos

### ✅ Passo 0.2: Criar Branch no Git (Opcional mas Recomendado)
```bash
cd Explanation_With_Rejection_final
git status
git add .
git commit -m "Estado antes da reorganização"
git checkout -b reorganization
```
- [ ] Branch criada (se usar Git)
- [ ] Commit do estado atual feito

### ✅ Passo 0.3: Documentar Estado Atual
```bash
# Listar estrutura atual
tree /F > estrutura_antes.txt
# OU no PowerShell:
Get-ChildItem -Recurse | Select-Object FullName > estrutura_antes.txt
```
- [ ] Arquivo `estrutura_antes.txt` criado
- [ ] Revisar arquivos importantes

---

## 📂 FASE 1: CRIAR ESTRUTURA DE PASTAS (10 minutos)

### ✅ Passo 1.1: Criar Pastas Principais
```bash
# Criar todas as pastas novas de uma vez
mkdir src analysis scripts config notebooks docs tests

# Dentro de results/, criar subpastas
cd results
mkdir json reports tables
cd reports
mkdir peab anchor minexp benchmark
cd ..

# Dentro de plots/, organizar
cd plots
mkdir comparative mnist_explanations
cd ..\..
```
- [ ] Pasta `src/` criada
- [ ] Pasta `analysis/` criada
- [ ] Pasta `scripts/` criada
- [ ] Pasta `config/` criada
- [ ] Pasta `notebooks/` criada
- [ ] Pasta `docs/` criada
- [ ] Pasta `tests/` criada
- [ ] Subpastas em `results/` criadas
  - [ ] `results/json/`
  - [ ] `results/reports/peab/`
  - [ ] `results/reports/anchor/`
  - [ ] `results/reports/minexp/`
  - [ ] `results/reports/benchmark/`
  - [ ] `results/plots/comparative/`
  - [ ] `results/plots/mnist_explanations/`
  - [ ] `results/tables/`

### ✅ Passo 1.2: Criar Arquivos __init__.py
```bash
# Tornar pastas em módulos Python
type nul > src\__init__.py
type nul > analysis\__init__.py
type nul > utils\__init__.py
type nul > data\__init__.py
type nul > tests\__init__.py
```
- [ ] `src/__init__.py` criado
- [ ] `analysis/__init__.py` criado
- [ ] `utils/__init__.py` criado
- [ ] `data/__init__.py` criado
- [ ] `tests/__init__.py` criado

---

## 🚚 FASE 2: MOVER ARQUIVOS PRINCIPAIS (20 minutos)

### ✅ Passo 2.1: Mover Scripts Principais para src/
```bash
copy peab.py src\peab.py
copy anchor.py src\anchor.py
copy minexp.py src\minexp.py
copy benchmark_peab.py src\benchmark_peab.py
```
- [ ] `peab.py` copiado para `src/`
- [ ] `anchor.py` copiado para `src/`
- [ ] `minexp.py` copiado para `src/`
- [ ] `benchmark_peab.py` copiado para `src/`

**⚠️ NÃO DELETAR OS ORIGINAIS AINDA!**

### ✅ Passo 2.2: Testar Scripts Movidos
```bash
# Testar se consegue importar
python -c "from src import peab"
python -c "from src import anchor"
python -c "from src import minexp"
```
- [ ] Import de `src.peab` funciona
- [ ] Import de `src.anchor` funciona
- [ ] Import de `src.minexp` funciona
- [ ] Se algum falhar, ajustar imports internos

### ✅ Passo 2.3: Ajustar Imports nos Arquivos Movidos (SE NECESSÁRIO)
**Exemplo**: Se `src/peab.py` importa `from utils import...`  
**Mudar para**: `from utils import...` (continua igual, pois utils está na raiz)

**Exemplo**: Se `src/peab.py` importa `from data.datasets import...`  
**Mudar para**: `from data.datasets import...` (continua igual)

- [ ] Verificar imports em `src/peab.py`
- [ ] Verificar imports em `src/anchor.py`
- [ ] Verificar imports em `src/minexp.py`
- [ ] Verificar imports em `src/benchmark_peab.py`

---

## 📊 FASE 3: MOVER SCRIPTS DE ANÁLISE (15 minutos)

### ✅ Passo 3.1: Mover para analysis/
```bash
copy summarize_results.py analysis\summarize_results.py
copy visualizer.py analysis\visualize_mnist_explanations.py
copy benchmark_analysis.py analysis\generate_comparative_plots.py
copy bench_resumo.py analysis\generate_comparative_tables.py
copy audit\detailed_json.py analysis\detailed_json.py
copy audit\score_overlap.py analysis\score_overlap.py
```
- [ ] `summarize_results.py` → `analysis/`
- [ ] `visualizer.py` → `analysis/visualize_mnist_explanations.py`
- [ ] `benchmark_analysis.py` → `analysis/generate_comparative_plots.py`
- [ ] `bench_resumo.py` → `analysis/generate_comparative_tables.py`
- [ ] Scripts de `audit/` movidos para `analysis/`

### ✅ Passo 3.2: Testar Scripts de Análise
```bash
python -c "from analysis import summarize_results"
python -c "from analysis import visualize_mnist_explanations"
```
- [ ] Imports funcionam
- [ ] Ajustar paths de leitura de JSONs se necessário

---

## 🔧 FASE 4: MOVER SCRIPTS AUXILIARES (10 minutos)

### ✅ Passo 4.1: Mover para scripts/
```bash
copy busca_mnist.py scripts\busca_mnist.py
copy teste_debug_mnist.py scripts\teste_debug_mnist.py
copy diagnostico_*.py scripts\

# Arquivos "copy" e temporários
copy peab_copy.py scripts\peab_copy.py
copy "benchmark_peab copy.py" "scripts\benchmark_peab_copy.py"
```
- [ ] `busca_mnist.py` → `scripts/`
- [ ] `teste_debug_mnist.py` → `scripts/`
- [ ] `diagnostico_*.py` → `scripts/`
- [ ] Arquivos `*_copy.py` → `scripts/` (para deletar depois)

---

## ⚙️ FASE 5: MOVER CONFIGURAÇÕES (5 minutos)

### ✅ Passo 5.1: Mover JSONs de Configuração
```bash
copy json\hiperparametros.json config\hiperparametros.json
```
- [ ] `hiperparametros.json` → `config/`

### ✅ Passo 5.2: Criar experiment_config.json (Opcional)
```bash
# Criar arquivo de configuração geral
notepad config\experiment_config.json
```
**Conteúdo sugerido**:
```json
{
  "random_state": 42,
  "test_size": 0.3,
  "rejection_cost": 0.24,
  "output_dirs": {
    "json": "results/json",
    "reports": "results/reports",
    "plots": "results/plots",
    "tables": "results/tables"
  }
}
```
- [ ] `config/experiment_config.json` criado

---

## 📁 FASE 6: REORGANIZAR RESULTADOS EXISTENTES (15 minutos)

### ✅ Passo 6.1: Mover JSONs de Resultados
```bash
# Mover JSONs existentes
move json\comparative_results.json results\json\comparative_results.json
# Verificar se há outros JSONs em json/
dir json
```
- [ ] `comparative_results.json` → `results/json/`
- [ ] Outros JSONs movidos (se existirem)

### ✅ Passo 6.2: Organizar Relatórios
```bash
# Os relatórios já devem estar em results/report/
# Apenas renomear a pasta 'report' para 'reports'
cd results
if exist report (
    move report reports_temp
    mkdir reports
    move reports_temp\peab reports\peab
    move reports_temp\anchor reports\anchor
    move reports_temp\minexp reports\minexp
    rmdir reports_temp
)
cd ..

# Mover benchmarks
if exist results\benchmark (
    xcopy results\benchmark results\reports\benchmark /E /I
)
```
- [ ] Pasta `results/report/` → `results/reports/`
- [ ] `results/benchmark/` → `results/reports/benchmark/`
- [ ] Estrutura conferida

### ✅ Passo 6.3: Organizar Plots
```bash
cd results\plots
# Criar subpastas se não existirem
mkdir comparative 2>nul
mkdir mnist_explanations 2>nul

# Mover plots interativos
cd ..
if exist plots_interativos (
    xcopy plots_interativos plots\interactive /E /I
)

# Mover plots de score overlap
cd ..\analysis_output
if exist plots (
    xcopy plots ..\results\plots\score_overlap /E /I
)
cd ..
```
- [ ] Plots organizados em subpastas
- [ ] `plots_interativos/` → `results/plots/interactive/`
- [ ] `analysis_output/plots/` → `results/plots/score_overlap/`

---

## 🎮 FASE 7: CRIAR MENU PRINCIPAL (30 minutos)

### ✅ Passo 7.1: Criar main.py
```bash
notepad main.py
```

**Cole este código inicial**:

```python
"""
EXPLAINABLE AI WITH REJECTION OPTION
Sistema de Experimentos - Menu Principal
Dissertação de Mestrado
"""

import os
import sys
from pathlib import Path

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def exibir_menu_principal():
    limpar_tela()
    print("═" * 70)
    print("  EXPLAINABLE AI WITH REJECTION OPTION - Sistema de Experimentos")
    print("═" * 70)
    print()
    print("[1] 🔬 EXECUTAR EXPERIMENTOS")
    print("    [1.1] PEAB (Método Proposto)")
    print("    [1.2] Anchor (Baseline)")
    print("    [1.3] MinExp (Baseline)")
    print("    [1.4] Executar Todos os Métodos")
    print("    [1.5] Benchmark PEAB vs MILP (PuLP)")
    print()
    print("[2] 📊 ANÁLISE E VISUALIZAÇÃO")
    print("    [2.1] Gerar Gráficos Comparativos")
    print("    [2.2] Gerar Tabelas para Dissertação")
    print("    [2.3] Visualizar Explicações MNIST")
    print("    [2.4] Resumo Geral dos Resultados")
    print()
    print("[3] 🔧 UTILITÁRIOS")
    print("    [3.1] Buscar Melhores Hiperparâmetros (MNIST)")
    print("    [3.2] Limpar Resultados Antigos")
    print("    [3.3] Exportar Resultados (ZIP)")
    print()
    print("[0] ❌ SAIR")
    print()
    print("═" * 70)
    
def executar_peab():
    print("\n🔬 Executando PEAB...")
    from src import peab
    # peab.main() ou chamar a função apropriada
    input("\n✅ Pressione ENTER para voltar ao menu...")

def executar_anchor():
    print("\n🔬 Executando Anchor...")
    from src import anchor
    # anchor.main()
    input("\n✅ Pressione ENTER para voltar ao menu...")

def executar_minexp():
    print("\n🔬 Executando MinExp...")
    from src import minexp
    # minexp.main()
    input("\n✅ Pressione ENTER para voltar ao menu...")

def executar_todos():
    print("\n🔬 Executando todos os métodos...")
    executar_peab()
    executar_anchor()
    executar_minexp()

def executar_benchmark():
    print("\n🔬 Executando Benchmark PEAB vs MILP...")
    from src import benchmark_peab
    # benchmark_peab.main()
    input("\n✅ Pressione ENTER para voltar ao menu...")

def gerar_graficos():
    print("\n📊 Gerando gráficos comparativos...")
    from analysis import generate_comparative_plots
    # generate_comparative_plots.main()
    input("\n✅ Pressione ENTER para voltar ao menu...")

def gerar_tabelas():
    print("\n📊 Gerando tabelas para dissertação...")
    from analysis import generate_comparative_tables
    # generate_comparative_tables.main()
    input("\n✅ Pressione ENTER para voltar ao menu...")

def visualizar_mnist():
    print("\n📊 Visualizando explicações MNIST...")
    from analysis import visualize_mnist_explanations
    # visualize_mnist_explanations.main()
    input("\n✅ Pressione ENTER para voltar ao menu...")

def resumo_resultados():
    print("\n📊 Gerando resumo dos resultados...")
    from analysis import summarize_results
    # summarize_results.main()
    input("\n✅ Pressione ENTER para voltar ao menu...")

def buscar_hiperparametros():
    print("\n🔧 Buscando melhores hiperparâmetros...")
    from scripts import busca_mnist
    # busca_mnist.main()
    input("\n✅ Pressione ENTER para voltar ao menu...")

def limpar_resultados():
    print("\n🔧 Limpeza de resultados antigos...")
    resposta = input("⚠️  Tem certeza? Isso removerá resultados antigos. (s/N): ")
    if resposta.lower() == 's':
        print("Limpando...")
        # Implementar limpeza
        print("✅ Limpeza concluída!")
    input("\n✅ Pressione ENTER para voltar ao menu...")

def exportar_resultados():
    print("\n🔧 Exportando resultados...")
    import shutil
    from datetime import datetime
    
    data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo_zip = f"resultados_{data_hora}.zip"
    
    shutil.make_archive(f"resultados_{data_hora}", 'zip', 'results')
    print(f"✅ Resultados exportados para: {arquivo_zip}")
    input("\n✅ Pressione ENTER para voltar ao menu...")

def main():
    while True:
        exibir_menu_principal()
        opcao = input("Digite sua opção: ").strip()
        
        if opcao == "0":
            print("\n👋 Até logo!")
            break
        elif opcao == "1.1":
            executar_peab()
        elif opcao == "1.2":
            executar_anchor()
        elif opcao == "1.3":
            executar_minexp()
        elif opcao == "1.4":
            executar_todos()
        elif opcao == "1.5":
            executar_benchmark()
        elif opcao == "2.1":
            gerar_graficos()
        elif opcao == "2.2":
            gerar_tabelas()
        elif opcao == "2.3":
            visualizar_mnist()
        elif opcao == "2.4":
            resumo_resultados()
        elif opcao == "3.1":
            buscar_hiperparametros()
        elif opcao == "3.2":
            limpar_resultados()
        elif opcao == "3.3":
            exportar_resultados()
        else:
            print("\n❌ Opção inválida!")
            input("Pressione ENTER para continuar...")

if __name__ == "__main__":
    main()
```

- [ ] `main.py` criado
- [ ] Código base inserido

### ✅ Passo 7.2: Testar Menu Inicial
```bash
python main.py
# Testar opção [0] para sair
# Verificar se menu exibe corretamente
```
- [ ] Menu exibe corretamente
- [ ] Opção [0] funciona

---

## 🔗 FASE 8: AJUSTAR IMPORTS E PATHS (30 minutos)

### ✅ Passo 8.1: Atualizar Paths em src/peab.py
**Verificar e ajustar**:
- [ ] Imports de `utils.*` (deve continuar funcionando)
- [ ] Imports de `data.datasets` (deve continuar funcionando)
- [ ] Paths de salvamento de resultados:
  - De: `'results/report/peab'` 
  - Para: `'results/reports/peab'`
  - De: `'json/comparative_results.json'`
  - Para: `'results/json/comparative_results.json'`
- [ ] Path do hiperparametros.json:
  - De: `'json/hiperparametros.json'`
  - Para: `'config/hiperparametros.json'`

### ✅ Passo 8.2: Atualizar Paths em src/anchor.py
- [ ] Paths de resultados: `results/reports/anchor/`
- [ ] Path do hiperparametros: `config/hiperparametros.json`
- [ ] Path do JSON: `results/json/comparative_results.json`

### ✅ Passo 8.3: Atualizar Paths em src/minexp.py
- [ ] Paths de resultados: `results/reports/minexp/`
- [ ] Path do hiperparametros: `config/hiperparametros.json`
- [ ] Path do JSON: `results/json/comparative_results.json`

### ✅ Passo 8.4: Atualizar Paths em src/benchmark_peab.py
- [ ] Path de resultados: `results/reports/benchmark/`

### ✅ Passo 8.5: Atualizar Paths em Scripts de Análise
**Em `analysis/summarize_results.py`**:
- [ ] Leitura de JSON: `results/json/comparative_results.json`

**Em `analysis/visualize_mnist_explanations.py`**:
- [ ] Leitura de JSON: `results/json/comparative_results.json`
- [ ] Salvamento de imagens: `results/plots/mnist_explanations/`

**Em `analysis/generate_comparative_plots.py`**:
- [ ] Leitura de JSON: `results/json/comparative_results.json`
- [ ] Salvamento: `results/plots/comparative/`

**Em `analysis/generate_comparative_tables.py`**:
- [ ] Leitura de JSON: `results/json/comparative_results.json`
- [ ] Salvamento: `results/tables/`

---

## ✅ FASE 9: TESTAR TUDO (45 minutos)

### ✅ Passo 9.1: Teste Individual - PEAB
```bash
python main.py
# Escolher opção [1.1]
# Ou testar direto:
python -m src.peab
```
- [ ] PEAB executa sem erros
- [ ] JSON salvo em `results/json/`
- [ ] Relatório salvo em `results/reports/peab/`
- [ ] Paths todos corretos

### ✅ Passo 9.2: Teste Individual - Anchor
```bash
python main.py
# Escolher opção [1.2]
```
- [ ] Anchor executa sem erros
- [ ] Resultados salvos corretamente

### ✅ Passo 9.3: Teste Individual - MinExp
```bash
python main.py
# Escolher opção [1.3]
```
- [ ] MinExp executa sem erros
- [ ] Resultados salvos corretamente

### ✅ Passo 9.4: Teste Individual - Benchmark
```bash
python main.py
# Escolher opção [1.5]
```
- [ ] Benchmark executa sem erros
- [ ] Relatório em `results/reports/benchmark/`

### ✅ Passo 9.5: Teste - Visualização MNIST
```bash
python main.py
# Escolher opção [2.3]
```
- [ ] Imagens geradas em `results/plots/mnist_explanations/`
- [ ] Sem erros de path

### ✅ Passo 9.6: Teste - Gráficos Comparativos
```bash
python main.py
# Escolher opção [2.1]
```
- [ ] Gráficos gerados em `results/plots/comparative/`
- [ ] Sem erros

### ✅ Passo 9.7: Teste - Tabelas
```bash
python main.py
# Escolher opção [2.2]
```
- [ ] Tabelas geradas em `results/tables/`
- [ ] Formatos CSV e/ou LaTeX

### ✅ Passo 9.8: Teste - Resumo
```bash
python main.py
# Escolher opção [2.4]
```
- [ ] Resumo gerado corretamente
- [ ] Lê JSONs sem problemas

---

## 🧹 FASE 10: LIMPEZA FINAL (20 minutos)

### ✅ Passo 10.1: Deletar Arquivos Duplicados da Raiz
**⚠️ APENAS DEPOIS DE CONFIRMAR QUE TUDO FUNCIONA!**

```bash
# Deletar scripts que foram movidos para src/
del peab.py
del anchor.py
del minexp.py
del benchmark_peab.py

# Deletar scripts que foram movidos para analysis/
del summarize_results.py
del visualizer.py
del benchmark_analysis.py
del bench_resumo.py
del bench_PLOTLY.py

# Deletar scripts que foram movidos para scripts/
del busca_mnist.py
del teste_debug_mnist.py
del diagnostico_*.py
del peab_copy.py
del "benchmark_peab copy.py"
```
- [ ] Scripts duplicados da raiz removidos
- [ ] Conferir que os arquivos em pastas organizadas funcionam

### ✅ Passo 10.2: Arquivar ou Deletar Pastas Antigas
```bash
# Arquivar pastas antigas
mkdir _archived
move test_old _archived\test_old
move analysis_old _archived\analysis_old

# Deletar pasta audit (conteúdo já movido)
rmdir /S audit

# Deletar pasta json antiga (conteúdo movido para results/json)
rmdir /S json

# Deletar analysis_output (conteúdo movido)
rmdir /S analysis_output

# Deletar plots_interativos (movido para results/plots/interactive)
rmdir /S results\plots_interativos

# Limpar __pycache__
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
```
- [ ] Pasta `test_old/` arquivada ou deletada
- [ ] Pasta `analysis_old/` arquivada ou deletada
- [ ] Pasta `audit/` deletada
- [ ] Pasta `json/` antiga deletada
- [ ] Pasta `analysis_output/` deletada
- [ ] `__pycache__/` limpo

### ✅ Passo 10.3: Limpar Arquivos Temporários
```bash
# No scripts/, deletar arquivos *_copy.py
del scripts\*_copy.py

# Deletar arquivos de diagnóstico se não precisar mais
# del scripts\diagnostico_*.py
# del scripts\teste_debug_mnist.py
```
- [ ] Arquivos `*_copy.py` removidos
- [ ] Arquivos temporários removidos (se não precisar)

---

## 📚 FASE 11: DOCUMENTAÇÃO (30 minutos)

### ✅ Passo 11.1: Atualizar README.md
```bash
notepad README.md
```

**Adicionar/Atualizar**:
- [ ] Seção "Estrutura do Projeto" com árvore de pastas
- [ ] Seção "Como Executar" com `python main.py`
- [ ] Seção "Requisitos" atualizada
- [ ] Seção "Datasets" com descrição
- [ ] Seção "Métodos" com PEAB, Anchor, MinExp

### ✅ Passo 11.2: Criar docs/methodology.md
```bash
notepad docs\methodology.md
```
- [ ] Descrição do método PEAB
- [ ] Algoritmo explicado
- [ ] Diferenças para baselines

### ✅ Passo 11.3: Criar docs/datasets.md
```bash
notepad docs\datasets.md
```
- [ ] Lista de datasets
- [ ] Características de cada um
- [ ] Fontes

### ✅ Passo 11.4: Criar docs/setup_guide.md
```bash
notepad docs\setup_guide.md
```
- [ ] Guia de instalação
- [ ] Configuração do ambiente
- [ ] Primeiros passos

---

## 🔄 FASE 12: ATUALIZAR .gitignore (5 minutos)

### ✅ Passo 12.1: Atualizar .gitignore
```bash
notepad .gitignore
```

**Adicionar/Verificar**:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Resultados (opcional - você pode querer versioná-los)
results/json/*.json
results/plots/**/*.png
results/reports/**/*.txt

# Arquivos temporários
_archived/
estrutura_antes.txt
*.bak
*.tmp

# OS
.DS_Store
Thumbs.db
```
- [ ] `.gitignore` atualizado
- [ ] Conferir se arquivos corretos estão ignorados

---

## ✨ FASE 13: COMMIT E FINALIZAÇÃO (10 minutos)

### ✅ Passo 13.1: Criar Estrutura Nova no Git
```bash
git status
git add .
git commit -m "Reorganização completa da estrutura do projeto

- Scripts principais movidos para src/
- Scripts de análise movidos para analysis/
- Scripts auxiliares movidos para scripts/
- Configurações centralizadas em config/
- Resultados organizados em results/ com subpastas
- Menu principal criado (main.py)
- Documentação atualizada
- Paths ajustados em todos os scripts
- Estrutura profissional para dissertação"
```
- [ ] Commit criado
- [ ] Mensagem descritiva

### ✅ Passo 13.2: Testar Tudo Novamente (Smoke Test)
```bash
# Teste rápido de cada funcionalidade
python main.py
```
- [ ] Menu funciona
- [ ] PEAB executa
- [ ] Visualizações funcionam
- [ ] Nenhum erro de import ou path

### ✅ Passo 13.3: Fazer Merge (Se estiver usando branch)
```bash
git checkout main
git merge reorganization
git branch -d reorganization
```
- [ ] Merge feito
- [ ] Branch temporária deletada

### ✅ Passo 13.4: Gerar Estrutura Final
```bash
tree /F > estrutura_depois.txt
# Comparar com estrutura_antes.txt
```
- [ ] `estrutura_depois.txt` criado
- [ ] Comparação feita
- [ ] Estrutura conforme proposta

---

## 🎉 FASE 14: VALIDAÇÃO FINAL (15 minutos)

### ✅ Checklist Final de Validação

**Estrutura**:
- [ ] Todos os arquivos estão nas pastas corretas
- [ ] Não há duplicatas na raiz
- [ ] Pastas organizadas logicamente

**Funcionalidade**:
- [ ] `python main.py` funciona
- [ ] Todos os métodos executam (PEAB, Anchor, MinExp)
- [ ] Benchmark funciona
- [ ] Visualizações geram imagens
- [ ] Tabelas são criadas
- [ ] Resumo funciona

**Paths**:
- [ ] JSONs salvos em `results/json/`
- [ ] Relatórios em `results/reports/{metodo}/`
- [ ] Plots em `results/plots/{tipo}/`
- [ ] Tabelas em `results/tables/`
- [ ] Config lido de `config/`

**Documentação**:
- [ ] README.md atualizado
- [ ] docs/ com arquivos .md criados
- [ ] PROPOSTA_ORGANIZACAO.md na raiz

**Git**:
- [ ] `.gitignore` correto
- [ ] Commit feito
- [ ] Histórico limpo

---

## 📊 RESULTADO ESPERADO

Ao final deste checklist, você terá:

✅ **Estrutura Profissional**: Projeto organizado em módulos lógicos  
✅ **Menu Central**: Uma entrada única (`main.py`) para todas as operações  
✅ **Resultados Organizados**: Tudo separado por tipo em `results/`  
✅ **Fácil Manutenção**: Simples adicionar novos scripts  
✅ **Pronto para Publicação**: Estrutura ideal para GitHub/dissertação  
✅ **Testado**: Tudo funcionando perfeitamente  

---

## ⏱️ TEMPO ESTIMADO POR FASE

- Fase 0: 15 min (Backup)
- Fase 1: 10 min (Criar pastas)
- Fase 2: 20 min (Mover principais)
- Fase 3: 15 min (Mover análises)
- Fase 4: 10 min (Mover auxiliares)
- Fase 5: 5 min (Mover configs)
- Fase 6: 15 min (Reorganizar resultados)
- Fase 7: 30 min (Criar menu)
- Fase 8: 30 min (Ajustar paths)
- Fase 9: 45 min (Testar tudo)
- Fase 10: 20 min (Limpeza)
- Fase 11: 30 min (Documentação)
- Fase 12: 5 min (.gitignore)
- Fase 13: 10 min (Git)
- Fase 14: 15 min (Validação)

**TOTAL**: ~2h30min - 3h

---

## 💡 DICAS IMPORTANTES

1. **Não pule etapas**: Cada fase prepara a próxima
2. **Teste frequentemente**: Melhor descobrir erro cedo
3. **Mantenha backup**: Até ter certeza que tudo funciona
4. **Use Git**: Facilita reverter se algo der errado
5. **Documente problemas**: Anote ajustes que precisou fazer
6. **Peça ajuda**: Se travar em alguma fase, pode pedir auxílio
7. **Não delete até testar**: Só remova originais após confirmar que cópias funcionam

---

## 🆘 EM CASO DE PROBLEMAS

**Problema**: Import não funciona após mover arquivo  
**Solução**: Verificar se `__init__.py` existe na pasta, ajustar imports relativos

**Problema**: Path não encontrado  
**Solução**: Usar `Path()` do pathlib ou verificar se está usando path relativo correto

**Problema**: Menu não importa módulo  
**Solução**: Adicionar pasta ao PYTHONPATH ou usar import relativo

**Problema**: Testes falham  
**Solução**: Revisar Fase 8 (ajuste de paths) para o script específico

---

**Boa sorte com a reorganização! 🚀**

Siga passo a passo e você terá um projeto profissional e bem organizado!
