"""
Script para regenerar todos os relatórios de validação com o novo formato melhorado.
"""

import json
import os
import sys
from pathlib import Path

# Adicionar o diretório de validação ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from peab_validation import (
    carregar_resultados_metodo, 
    validar_metodo,
    gerar_relatorio_txt,
    gerar_plots,
    salvar_json_validacao,
    NUM_PERTURBATIONS_DEFAULT,
    NUM_PERTURBATIONS_LARGE
)

def main():
    """Regenera relatórios para PIMA e MNIST."""
    
    print("═" * 80)
    print("REGENERAÇÃO DE RELATÓRIOS DE VALIDAÇÃO")
    print("═" * 80)
    
    # Lista de datasets para validar
    datasets = [
        ("PEAB", "pima_indians_diabetes"),
        ("PEAB", "mnist"),  # Vai procurar automaticamente por mnist_3_vs_8, etc
    ]
    
    for metodo, dataset in datasets:
        print(f"\n{'─'*80}")
        print(f"Validando {dataset} com {metodo}...")
        print(f"Método: Avaliação de Fidelidade por Perturbação")
        print(f"Perturbações: {NUM_PERTURBATIONS_DEFAULT:,} por instância")
        print(f"{'─'*80}")
        
        resultado = validar_metodo(metodo, dataset, verbose=True)
        
        if resultado:
            # Salvar JSON
            salvar_json_validacao(resultado, metodo, dataset)
            
            # Gerar relatório TXT
            gerar_relatorio_txt(resultado, metodo, dataset)
            
            # Gerar gráficos
            gerar_plots(resultado, metodo, dataset)
            
            print(f"\n✓ Relatório gerado com sucesso!")
            print(f"  Fidelidade: {resultado['global_metrics']['fidelity_overall']:.2f}%")
            print(f"  Cobertura: {resultado['global_metrics']['coverage']:.2f}%")
            print(f"  Compactação: {resultado['global_metrics']['reduction_rate']:.1f}%")
        else:
            print(f"✗ Erro ao gerar relatório para {dataset}")
    
    print(f"\n{'═'*80}")
    print("REGENERAÇÃO COMPLETA!")
    print(f"{'═'*80}")

if __name__ == "__main__":
    main()
