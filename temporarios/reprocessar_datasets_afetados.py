"""
Script auxiliar para reprocessar datasets afetados pela correção do subsample_size.

Datasets que PRECISAM ser reprocessados (treino estava em 1-2% ao invés de 70%):
- creditcard (subsample_size: 0.02)
- covertype (subsample_size: 0.01)
- mnist (subsample_size: 0.12) - se houver variações processadas

Métodos afetados:
- PEAB
- MinExp
- PuLP (se existir)
- Anchor (se existir)
"""
import os
import sys
import json

JSON_DIR = "json"

def verificar_datasets_existentes():
    """Verifica quais datasets já foram processados em cada método."""
    
    metodos = ['peab', 'minexp', 'pulp', 'anchor']
    datasets_afetados = ['creditcard', 'covertype', 'mnist']
    
    print("\n" + "="*80)
    print("VERIFICAÇÃO DE DATASETS PROCESSADOS (Afetados pelo bug do subsample)")
    print("="*80 + "\n")
    
    resultados = {}
    
    for metodo in metodos:
        metodo_dir = os.path.join(JSON_DIR, metodo)
        if not os.path.exists(metodo_dir):
            continue
        
        resultados[metodo] = []
        
        for dataset in datasets_afetados:
            # Verificar arquivo principal
            json_file = os.path.join(metodo_dir, f"{dataset}.json")
            if os.path.exists(json_file):
                resultados[metodo].append(dataset)
            
            # Para MNIST, verificar variações (mnist_3_vs_8, etc)
            if dataset == 'mnist':
                files = [f for f in os.listdir(metodo_dir) if f.startswith('mnist_') and f.endswith('.json')]
                for f in files:
                    variant = f.replace('.json', '')
                    if variant not in resultados[metodo]:
                        resultados[metodo].append(variant)
    
    # Mostrar resultados
    print("STATUS DOS DATASETS AFETADOS:")
    print("-"*80)
    
    for metodo in metodos:
        if metodo in resultados and resultados[metodo]:
            print(f"\n{metodo.upper()}:")
            for dataset in resultados[metodo]:
                json_file = os.path.join(JSON_DIR, metodo, f"{dataset}.json")
                
                # Verificar tamanho do arquivo para estimar se tem dados
                if os.path.exists(json_file):
                    size_kb = os.path.getsize(json_file) / 1024
                    print(f"  ❌ {dataset:30s} - PRECISA REPROCESSAR ({size_kb:.1f} KB)")
        else:
            print(f"\n{metodo.upper()}: Nenhum dataset afetado encontrado")
    
    return resultados


def gerar_comandos_reprocessamento(resultados):
    """Gera lista de comandos para reprocessar os datasets."""
    
    print("\n" + "="*80)
    print("COMANDOS PARA REPROCESSAMENTO")
    print("="*80 + "\n")
    
    print("Execute os comandos abaixo NA ORDEM para reprocessar os datasets:")
    print("(Recomendado começar pelo mais rápido - creditcard)\n")
    
    comandos = []
    
    # Ordem: creditcard (médio) → covertype (grande) → mnist (se houver)
    datasets_ordem = ['creditcard', 'covertype', 'mnist']
    metodos_ordem = ['peab', 'minexp', 'pulp', 'anchor']
    
    for dataset in datasets_ordem:
        encontrou_dataset = False
        
        for metodo in metodos_ordem:
            if metodo in resultados:
                # Verificar se este dataset específico existe
                datasets_metodo = resultados[metodo]
                
                # Para mnist, incluir variações
                datasets_check = [d for d in datasets_metodo if d == dataset or d.startswith(f"{dataset}_")]
                
                if datasets_check:
                    encontrou_dataset = True
                    for d in datasets_check:
                        comandos.append((metodo, d))
        
        if encontrou_dataset:
            print(f"# ───────────────────────────────────────────────────────────")
            print(f"# {dataset.upper()}")
            print(f"# ───────────────────────────────────────────────────────────")
            
            for metodo, d in comandos:
                if d == dataset or d.startswith(f"{dataset}_"):
                    print(f"python {metodo}.py")
                    print(f"# Quando solicitar, escolha: {d}")
                    print()
            
            comandos = [c for c in comandos if c[1] != dataset and not c[1].startswith(f"{dataset}_")]
    
    print("\n" + "="*80)
    print("ESTIMATIVA DE TEMPO")
    print("="*80)
    print("  creditcard (284k instâncias):")
    print("    - PEAB: ~5-10 min")
    print("    - MinExp: ~30-60 min")
    print("    - PuLP: ~60-120 min (se existir)")
    print("    - Anchor: ~20-40 min (se existir)")
    print()
    print("  covertype (581k instâncias):")
    print("    - PEAB: ~10-20 min")
    print("    - MinExp: ~60-120 min")
    print("    - PuLP: ~120-240 min (se existir)")
    print("    - Anchor: ~40-80 min (se existir)")
    print()
    print("  TOTAL ESTIMADO: 3-8 horas (dependendo dos métodos)")
    print("="*80 + "\n")


def criar_script_batch():
    """Cria um script batch para executar tudo automaticamente (Windows)."""
    
    print("\n" + "="*80)
    print("OPÇÃO AVANÇADA: Script Batch Automático")
    print("="*80 + "\n")
    
    print("⚠️  NOTA: Reprocessamento automático via script pode não funcionar bem")
    print("   porque cada método requer interação (seleção de dataset).")
    print()
    print("   RECOMENDADO: Execute manualmente comando por comando.")
    print()
    
    resposta = input("Deseja criar um script .bat de referência mesmo assim? (s/N): ").strip().lower()
    
    if resposta == 's':
        batch_content = """@echo off
REM Script para reprocessar datasets afetados pelo bug do subsample_size
REM ATENÇÃO: Este script é apenas de REFERÊNCIA
REM Execute os comandos MANUALMENTE um por vez

echo ========================================================================
echo REPROCESSAMENTO - CREDITCARD
echo ========================================================================
echo.
echo Execute: python peab.py
echo Escolha: creditcard
echo.
pause

echo Execute: python minexp.py
echo Escolha: creditcard
echo.
pause

REM Adicione outros métodos conforme necessário

echo ========================================================================
echo REPROCESSAMENTO - COVERTYPE
echo ========================================================================
echo.
echo Execute: python peab.py
echo Escolha: covertype
echo.
pause

echo Execute: python minexp.py
echo Escolha: covertype
echo.
pause

echo ========================================================================
echo CONCLUÍDO
echo ========================================================================
pause
"""
        
        batch_path = "temporarios/reprocessar_datasets.bat"
        with open(batch_path, 'w', encoding='utf-8') as f:
            f.write(batch_content)
        
        print(f"\n✓ Script criado: {batch_path}")
        print("  Execute-o para ver os lembretes dos comandos")
    else:
        print("\n✓ OK, execute os comandos manualmente")


if __name__ == '__main__':
    print("\n" + "#"*80)
    print("REPROCESSAMENTO DE DATASETS - Correção do bug subsample_size")
    print("#"*80)
    print("\nContexto: O subsample estava sendo aplicado ANTES do split treino/teste,")
    print("resultando em treino com apenas 1-2% dos dados ao invés de 70%.")
    print("\nApós a correção, o modelo treina com dataset COMPLETO (70%),")
    print("então os resultados serão MUITO MELHORES (accuracy +5-10%).\n")
    
    resultados = verificar_datasets_existentes()
    
    if not any(resultados.values()):
        print("\n✓ Nenhum dataset afetado encontrado nos métodos!")
        print("  Não é necessário reprocessar nada.")
    else:
        gerar_comandos_reprocessamento(resultados)
        criar_script_batch()
        
        print("\n" + "="*80)
        print("PRÓXIMOS PASSOS")
        print("="*80)
        print("1. Execute os comandos acima NA ORDEM")
        print("2. Comece por creditcard (mais rápido)")
        print("3. Depois covertype (mais demorado)")
        print("4. Por último mnist (se aplicável)")
        print()
        print("5. Compare os resultados antigos vs novos:")
        print("   - Accuracy deve aumentar +5-10%")
        print("   - Explicações devem ser mais confiáveis")
        print("="*80 + "\n")
