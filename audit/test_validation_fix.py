"""
Script de teste rápido para verificar se a correção do peab_validation
resolveu o problema de minimalidade.
"""
import subprocess
import sys

def test_validation(dataset: str):
    """Executa validação para um dataset específico."""
    print(f"\n{'='*80}")
    print(f"🧪 TESTANDO VALIDAÇÃO: {dataset.upper()}")
    print(f"{'='*80}\n")
    
    # Simular inputs para o script de validação
    # 1 = PEAB, dataset_number, 0 = sair
    cmd = f'echo 1 & echo {dataset} & echo 0 | python peab_validation.py'
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=r"C:\Users\gleilsonpedro\OneDrive\Área de Trabalho\PYTHON\MESTRADO\XAI\Explanation_With_Rejection_final"
        )
        
        # Procurar por linhas de minimalidade no output
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if 'Minimalidade' in line or 'necessity' in line.lower():
                # Mostrar contexto (3 linhas antes e depois)
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                for j in range(start, end):
                    print(lines[j])
        
        print(f"\n✅ Validação concluída para {dataset}")
        
    except Exception as e:
        print(f"❌ Erro ao executar validação: {e}")

if __name__ == "__main__":
    print("🔧 Teste de Correção do peab_validation.py")
    print("="*80)
    
    # Testar apenas vertebral_column por enquanto
    test_validation("vertebral_column")
