"""
Script auxiliar para re-executar experimentos e atualizar tabela de runtime com desvio padrão.
Permite executar apenas os datasets rápidos ou todos os datasets.
"""

import subprocess
import sys
import time
from datetime import datetime

# Datasets organizados por velocidade
DATASETS_RAPIDOS = [
    "banknote",
    "vertebral_column",
    "pima_indians_diabetes",
    "heart_disease",
    "breast_cancer",
    "sonar",
    "spambase"
]

DATASETS_DEMORADOS = [
    "creditcard",
    "covertype",
    "mnist"
]

METODOS = {
    "peab": "PEAB (MINABRO)",
    "anchor": "Anchor",
    "minexp": "MinExp (AbLinRO)"
}


def executar_comando(cmd, descricao):
    """Executa um comando e mede o tempo."""
    print(f"\n{'='*70}")
    print(f"🚀 {descricao}")
    print(f"{'='*70}")
    print(f"Comando: {' '.join(cmd)}")
    print()
    
    inicio = time.time()
    try:
        resultado = subprocess.run(cmd, check=True, capture_output=False)
        tempo_decorrido = time.time() - inicio
        print(f"\n✅ Concluído em {tempo_decorrido:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        tempo_decorrido = time.time() - inicio
        print(f"\n❌ Erro após {tempo_decorrido:.1f}s: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Execução interrompida pelo usuário!")
        sys.exit(1)


def executar_dataset(metodo, dataset):
    """Executa um método para um dataset específico."""
    cmd = [sys.executable, f"{metodo}.py", "--dataset", dataset]
    descricao = f"{METODOS[metodo]} - {dataset}"
    return executar_comando(cmd, descricao)


def gerar_tabela():
    """Gera a tabela de runtime unificada."""
    print(f"\n{'='*70}")
    print("📊 GERANDO TABELA DE RUNTIME UNIFICADA COM DESVIO PADRÃO")
    print(f"{'='*70}\n")
    
    cmd = [sys.executable, "temporarios/gerar_tabela_runtime_unificada.py"]
    return executar_comando(cmd, "Gerando tabela LaTeX")


def main():
    print("="*70)
    print("RE-EXECUTAR EXPERIMENTOS PARA DESVIO PADRÃO")
    print("="*70)
    print()
    print("Escolha uma opção:")
    print()
    print("1. Apenas datasets RÁPIDOS (7 datasets - ~1-2 horas)")
    print("   - banknote, vertebral_column, pima_indians_diabetes,")
    print("   - heart_disease, breast_cancer, sonar, spambase")
    print()
    print("2. TODOS os datasets (10 datasets - ~10-15 horas)")
    print("   - Inclui: creditcard, covertype, mnist")
    print()
    print("3. Apenas GERAR TABELA (sem re-executar)")
    print()
    print("4. Escolher datasets MANUALMENTE")
    print()
    
    opcao = input("Digite sua escolha (1-4): ").strip()
    
    if opcao == "3":
        gerar_tabela()
        return
    
    # Determinar lista de datasets
    if opcao == "1":
        datasets = DATASETS_RAPIDOS
        todos = False
    elif opcao == "2":
        datasets = DATASETS_RAPIDOS + DATASETS_DEMORADOS
        todos = True
    elif opcao == "4":
        print("\nDatasets disponíveis:")
        todos_datasets = DATASETS_RAPIDOS + DATASETS_DEMORADOS
        for i, ds in enumerate(todos_datasets, 1):
            status = "⚡ rápido" if ds in DATASETS_RAPIDOS else "🐌 demorado"
            print(f"  {i}. {ds} ({status})")
        
        escolhas = input("\nDigite os números separados por vírgula (ex: 1,2,3): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in escolhas.split(",")]
            datasets = [todos_datasets[i] for i in indices if 0 <= i < len(todos_datasets)]
            todos = False
        except:
            print("❌ Entrada inválida!")
            return
    else:
        print("❌ Opção inválida!")
        return
    
    if not datasets:
        print("❌ Nenhum dataset selecionado!")
        return
    
    # Escolher métodos
    print("\nQuais métodos executar?")
    print("1. Apenas PEAB (mais rápido)")
    print("2. PEAB + MinExp")
    print("3. TODOS (PEAB + Anchor + MinExp)")
    print("4. Apenas Anchor")
    print("5. Apenas MinExp")
    
    metodo_opcao = input("\nDigite sua escolha (1-5): ").strip()
    
    if metodo_opcao == "1":
        metodos = ["peab"]
    elif metodo_opcao == "2":
        metodos = ["peab", "minexp"]
    elif metodo_opcao == "3":
        metodos = ["peab", "anchor", "minexp"]
    elif metodo_opcao == "4":
        metodos = ["anchor"]
    elif metodo_opcao == "5":
        metodos = ["minexp"]
    else:
        print("❌ Opção inválida!")
        return
    
    # Resumo
    print(f"\n{'='*70}")
    print("📋 RESUMO DA EXECUÇÃO")
    print(f"{'='*70}")
    print(f"Datasets: {len(datasets)}")
    for ds in datasets:
        status = "⚡" if ds in DATASETS_RAPIDOS else "🐌"
        print(f"  {status} {ds}")
    print(f"\nMétodos: {len(metodos)}")
    for m in metodos:
        print(f"  - {METODOS[m]}")
    print(f"\nTotal de execuções: {len(datasets) * len(metodos)}")
    print(f"{'='*70}\n")
    
    confirmar = input("Deseja continuar? (s/n): ").strip().lower()
    if confirmar != "s":
        print("❌ Execução cancelada.")
        return
    
    # Executar
    inicio_geral = time.time()
    total_execucoes = len(datasets) * len(metodos)
    execucao_atual = 0
    sucessos = 0
    falhas = 0
    
    print(f"\n{'='*70}")
    print(f"🎯 INICIANDO EXECUÇÃO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    for dataset in datasets:
        for metodo in metodos:
            execucao_atual += 1
            print(f"\n[{execucao_atual}/{total_execucoes}] ", end="")
            
            if executar_dataset(metodo, dataset):
                sucessos += 1
            else:
                falhas += 1
                
                # Perguntar se quer continuar após falha
                if falhas <= 2:
                    continuar = input("\n⚠️  Deseja continuar mesmo com erro? (s/n): ").strip().lower()
                    if continuar != "s":
                        print("❌ Execução interrompida.")
                        break
    
    tempo_total = time.time() - inicio_geral
    
    # Relatório final
    print(f"\n{'='*70}")
    print("📊 RELATÓRIO FINAL")
    print(f"{'='*70}")
    print(f"✅ Sucessos: {sucessos}/{total_execucoes}")
    print(f"❌ Falhas: {falhas}/{total_execucoes}")
    print(f"⏱️  Tempo total: {tempo_total/60:.1f} minutos ({tempo_total/3600:.2f} horas)")
    print(f"{'='*70}\n")
    
    if sucessos > 0:
        gerar = input("Deseja gerar a tabela agora? (s/n): ").strip().lower()
        if gerar == "s":
            gerar_tabela()
            print(f"\n✅ Tabela salva em: results/tabelas_latex/runtime_unified_with_std.tex")
    
    print("\n✨ Execução concluída!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execução interrompida pelo usuário!")
        sys.exit(0)
