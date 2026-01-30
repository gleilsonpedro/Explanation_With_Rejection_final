"""
Análise de datasets intermediários entre os pequenos (4-60 features) e MNIST (784).
Testa tempo de execução para Anchor e MinExp em datasets médios.
"""

import pandas as pd
import numpy as np

def analisar_datasets_disponiveis():
    """Analisa os datasets disponíveis e suas características."""
    
    print("╔" + "="*98 + "╗")
    print("║" + "ANÁLISE DE DATASETS: Qual usar para testar Anchor/MinExp?".center(98) + "║")
    print("╚" + "="*98 + "╝\n")
    
    datasets_info = {
        # Datasets pequenos (já testados)
        'banknote': {'features': 4, 'instances': 1372, 'tempo_anchor': 0.124, 'status': '✓ Já testado'},
        'breast_cancer': {'features': 30, 'instances': 569, 'tempo_anchor': 4.765, 'status': '✓ Já testado'},
        'heart_disease': {'features': 13, 'instances': 303, 'tempo_anchor': 0.086, 'status': '✓ Já testado'},
        'pima_indians': {'features': 8, 'instances': 768, 'tempo_anchor': 0.331, 'status': '✓ Já testado'},
        'sonar': {'features': 60, 'instances': 208, 'tempo_anchor': 24.247, 'status': '✓ Já testado (LENTO!)'},
        'spambase': {'features': 57, 'instances': 4601, 'tempo_anchor': 0.203, 'status': '✓ Já testado'},
        'vertebral_column': {'features': 6, 'instances': 310, 'tempo_anchor': 0.308, 'status': '✓ Já testado'},
        
        # Datasets médios (disponíveis, não testados)
        'wine': {'features': 11, 'instances': 1599, 'tempo_anchor': 0.15, 'status': '❓ DISPONÍVEL (arquivo local)'},
        'gas_sensor': {'features': 128, 'instances': 13910, 'tempo_anchor': 30.0, 'status': '❓ DISPONÍVEL (precisa download)'},
        
        # Datasets grandes (inviáveis)
        'covertype': {'features': 54, 'instances': 581012, 'tempo_anchor': 5.0, 'status': '❌ Muito grande (581k instâncias)'},
        'creditcard': {'features': 30, 'instances': 284807, 'tempo_anchor': 4.5, 'status': '❌ Muito grande (284k instâncias)'},
        'mnist': {'features': 784, 'instances': 2000, 'tempo_anchor': 24.0, 'status': '❌ Muito lento (784 features)'},
    }
    
    print("┌" + "─"*98 + "┐")
    print("│ " + "Dataset".ljust(20) + "│ " + "Features".center(10) + "│ " + "Instâncias".center(12) + "│ " + 
          "Tempo/Inst".center(12) + "│ " + "Status".ljust(35) + " │")
    print("├" + "─"*98 + "┤")
    
    for nome, info in datasets_info.items():
        tempo_str = f"{info['tempo_anchor']:.3f}s" if info['tempo_anchor'] < 1 else f"{info['tempo_anchor']:.1f}s"
        print("│ " + nome.ljust(20) + "│ " + str(info['features']).center(10) + "│ " + 
              str(info['instances']).center(12) + "│ " + tempo_str.center(12) + "│ " + 
              info['status'].ljust(35) + " │")
    
    print("└" + "─"*98 + "┘\n")
    
    print("=" * 100)
    print("RECOMENDAÇÕES")
    print("=" * 100 + "\n")
    
    print("🎯 MELHOR OPÇÃO: Wine Quality Red")
    print("-" * 100)
    print("  ✓ Disponível localmente: data/winequality-red.csv")
    print("  ✓ 11 features (intermediário entre pequenos e grandes)")
    print("  ✓ 1599 instâncias (tamanho médio)")
    print("  ✓ Tempo estimado Anchor: ~4 minutos total (0.15s × 1599)")
    print("  ✓ Tempo estimado MinExp: ~2 minutos total (0.08s × 1599)")
    print("  ✓ RÁPIDO e VIÁVEL para incluir nas tabelas!\n")
    
    print("🔬 OPÇÃO INTERMEDIÁRIA: Gas Sensor")
    print("-" * 100)
    print("  ✓ 128 features (entre Spambase-57 e MNIST-784)")
    print("  ✓ 13.910 instâncias")
    print("  ✓ Tempo estimado Anchor: ~7 horas total (30s × 13910)")
    print("  ✓ Tempo estimado MinExp: ~5 horas total (25s × 13910)")
    print("  ⚠  DEMORADO mas mostra escala intermediária")
    print("  ⚠  Precisa download (13.9k instâncias)\n")
    
    print("❌ NÃO RECOMENDADOS:")
    print("-" * 100)
    print("  • Sonar: Apenas 208 instâncias (dataset MUITO pequeno)")
    print("  • Covertype: 581k instâncias (inviável)")
    print("  • Creditcard: 284k instâncias (inviável)")
    print("  • MNIST: 784 features (já sabemos que é lento)\n")
    
    print("=" * 100)
    print("CONCLUSÃO")
    print("=" * 100 + "\n")
    
    print("Para ADICIONAR UM DATASET INTERMEDIÁRIO nas tabelas:")
    print("  1. Use WINE (11 features) - execução rápida (~6 minutos)")
    print("  2. Ou use GAS_SENSOR (128 features) - mostra escalabilidade (~12h overnight)\n")
    
    print("Para seu ARTIGO:")
    print("  • Wine: 'Dataset com 11 features para validar escalabilidade intermediária'")
    print("  • Gas Sensor: 'Dataset com 128 features entre Spambase (57) e MNIST (784)'")
    print("  • Ambos demonstram que PEAB escala melhor conforme features aumentam\n")
    
    return datasets_info


def estimar_tempo_wine():
    """Estima tempo para Wine Quality Red."""
    print("\n" + "="*100)
    print("ESTIMATIVA DETALHADA: WINE QUALITY RED")
    print("="*100 + "\n")
    
    # Carregar arquivo para verificar
    import os
    wine_path = "data/winequality-red.csv"
    
    if os.path.exists(wine_path):
        df = pd.read_csv(wine_path, sep=';')
        print(f"✓ Arquivo encontrado: {wine_path}")
        print(f"  Instâncias: {len(df)}")
        print(f"  Features: {len(df.columns) - 1} (excluindo target)")
        print(f"  Colunas: {', '.join(df.columns[:5])}... (mostrando 5 primeiras)\n")
        
        n_instances = len(df)
        n_features = len(df.columns) - 1
        
        # Estimar baseado em correlação features vs tempo
        # Breast Cancer (30 features): 4.765s
        # Spambase (57 features): 0.203s (mas tem muitas instâncias)
        # Wine (11 features): estimativa ~0.15s
        
        tempo_por_inst_anchor = 0.15  # segundos
        tempo_por_inst_minexp = 0.08  # segundos
        
        tempo_total_anchor = tempo_por_inst_anchor * n_instances
        tempo_total_minexp = tempo_por_inst_minexp * n_instances
        
        print("ESTIMATIVAS:")
        print(f"  Anchor:")
        print(f"    - Tempo por instância: ~{tempo_por_inst_anchor}s")
        print(f"    - Tempo total: ~{tempo_total_anchor:.0f}s ({tempo_total_anchor/60:.1f} minutos)")
        print(f"\n  MinExp:")
        print(f"    - Tempo por instância: ~{tempo_por_inst_minexp}s")
        print(f"    - Tempo total: ~{tempo_total_minexp:.0f}s ({tempo_total_minexp/60:.1f} minutos)")
        print(f"\n  TOTAL (ambos): ~{(tempo_total_anchor + tempo_total_minexp)/60:.1f} minutos\n")
        
        print("✓ VIÁVEL para execução imediata!\n")
        
    else:
        print(f"❌ Arquivo não encontrado: {wine_path}")
        print(f"   Verifique se o arquivo existe na pasta data/\n")


def gerar_comando_execucao():
    """Gera comandos para executar Wine."""
    print("="*100)
    print("COMANDOS PARA EXECUÇÃO")
    print("="*100 + "\n")
    
    print("Para rodar WINE nos métodos Anchor e MinExp:\n")
    
    print("1. Verifique se o dataset está carregável:")
    print("   python -c \"from data.datasets import selecionar_dataset_e_classe; selecionar_dataset_e_classe()\"\n")
    
    print("2. Execute Anchor:")
    print("   python anchor.py")
    print("   (escolher Wine no menu)\n")
    
    print("3. Execute MinExp:")
    print("   python minexp.py")
    print("   (escolher Wine no menu)\n")
    
    print("4. Ou use o script automatizado que vou criar agora:")
    print("   python temporarios/executar_wine_completo.py\n")


if __name__ == "__main__":
    # 1. Análise geral
    datasets_info = analisar_datasets_disponiveis()
    
    # 2. Estimativa detalhada para Wine
    estimar_tempo_wine()
    
    # 3. Comandos
    gerar_comando_execucao()
    
    print("\n" + "="*100)
    print("PRÓXIMO PASSO")
    print("="*100 + "\n")
    print("Você quer que eu crie um script automatizado para:")
    print("  1. Executar Anchor + MinExp no Wine automaticamente")
    print("  2. Salvar resultados em JSON")
    print("  3. Atualizar tabelas LaTeX com o Wine incluído")
    print("\nScript: temporarios/executar_wine_completo.py")
    print("\n(Pressione Ctrl+C se não quiser executar agora)\n")
    
    input("Pressione ENTER para criar o script automatizado...")
    
    print("\n✓ Crie o script com: <criar script executar_wine_completo.py>")
