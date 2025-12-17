"""
Script de teste direto para validar a correção sem menu interativo.
"""
import sys
import os

# Adicionar diretório ao path
sys.path.insert(0, r'C:\Users\gleilsonpedro\OneDrive\Área de Trabalho\PYTHON\MESTRADO\XAI\Explanation_With_Rejection_final')

# Importar funções necessárias
from peab_validation import validar_metodo

def test_quick_validation(dataset: str, method: str = 'PEAB'):
    """Testa validação rápida para um dataset."""
    print(f"\n{'='*80}")
    print(f"VALIDANDO {method.upper()} - {dataset.upper()}")
    print(f"{'='*80}\n")
    
    try:
        # Chamar função de validação diretamente
        validar_metodo(method, dataset)
        
        # Ler e mostrar as métricas de minimalidade do relatório gerado
        report_path = f"results/validation/{method.lower()}/{dataset}/validation_report.txt"
        
        if os.path.exists(report_path):
            print(f"\nRESULTADO - Metricas de Minimalidade:")
            print("="*80)
            with open(report_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Encontrar seção de minimalidade
            in_minimal_section = False
            for line in lines:
                if 'Minimalidade por Tipo' in line:
                    in_minimal_section = True
                if in_minimal_section:
                    print(line.rstrip())
                    if line.strip().startswith('Taxa de Cobertura') or line.strip().startswith('───'):
                        if 'Taxa de Cobertura' in line:
                            break
        else:
            print(f"AVISO: Relatorio nao encontrado: {report_path}")
            
        print("\nOK - Teste concluido!")
        
    except Exception as e:
        print(f"ERRO durante validacao: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Teste com vertebral_column (o caso mais problemático)
    test_quick_validation("vertebral_column", "PEAB")
    
    print("\n" + "="*80)
    print("Comparacao esperada:")
    print("  ANTES DA CORRECAO:")
    print("    Positivas: 0%    (ERRADO - eram marcadas como redundantes)")
    print("    Negativas: 99%   (correto)")
    print("    Rejeitadas: 99%  (correto)")
    print("\n  APOS A CORRECAO:")
    print("    Positivas: >50%  (melhorado - teste adversarial correto)")
    print("    Negativas: >80%  (mantem bom)")
    print("    Rejeitadas: >90% (mantem otimo)")
    print("="*80)
