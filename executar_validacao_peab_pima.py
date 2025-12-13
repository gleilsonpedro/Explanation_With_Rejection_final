"""
Script para executar validação do PEAB no Pima Indians Diabetes automaticamente
"""
import sys
import os

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar funções necessárias
from peab_validation import validar_metodo, gerar_relatorio_txt, gerar_plots

# Executar validação
print("Executando validação do PEAB para Pima Indians Diabetes...")
resultados = validar_metodo('peab', 'pima_indians_diabetes', verbose=True)

if resultados:
    print("\n✅ Validação concluída com sucesso!")
    print(f"Fidelidade geral: {resultados['global_metrics']['fidelity_overall']:.2f}%")
    
    # Gerar relatório TXT
    print("\nGerando relatório TXT...")
    gerar_relatorio_txt(resultados, 'peab', 'pima_indians_diabetes')
    
    # Gerar gráficos
    print("Gerando gráficos...")
    gerar_plots(resultados, 'peab', 'pima_indians_diabetes')
    
    print("\n✅ Relatório e gráficos gerados!")
else:
    print("\n❌ Erro na validação!")
