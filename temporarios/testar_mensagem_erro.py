"""
Script para testar se a validação mostra a mensagem de erro correta para MinExp.
"""
import sys
import os

# Adicionar path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def testar_validacao_minexp():
    """Testa a validação do MinExp para ver a mensagem de erro."""
    
    print("\n" + "="*80)
    print("TESTE: Mensagem de Erro da Validação MinExp")
    print("="*80 + "\n")
    
    # Importar função de validação
    from peab_validation import validar_metodo
    
    # Tentar validar MinExp com vertebral_column
    print("Tentando validar MinExp com vertebral_column...")
    print("(Deve mostrar mensagem de erro clara sobre per_instance faltando)\n")
    
    resultado = validar_metodo(
        metodo='MinExp',
        dataset='vertebral_column',
        n_perturbacoes=100,  # Número baixo para teste rápido
        estrategia='uniform',
        modo_necessidade='auto',
        verbose=True
    )
    
    if resultado is None:
        print("\n" + "─"*80)
        print("✓ Teste concluído:")
        print("  - Função retornou None (esperado quando falta per_instance)")
        print("  - Mensagem de erro deve ter sido exibida acima")
        print("  - Verifique se a mensagem está clara e útil")
        print("─"*80 + "\n")
    else:
        print("\n⚠️  INESPERADO: Validação não retornou None")
        print("   Isso significa que per_instance foi encontrado!")


if __name__ == '__main__':
    testar_validacao_minexp()
