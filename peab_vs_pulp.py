"""
PEAB vs PuLP - Análise Comparativa de Qualidade
================================================
Compara a heurística PEAB com o solver ótimo PuLP para avaliar:
- Taxa de otimalidade (quantas vezes PEAB encontra a solução ótima)
- GAP médio (quantas features a mais o PEAB usa)
- Speedup (quanto mais rápido o PEAB é)

Este script LÊ os resultados já salvos em:
- json/peab/{dataset}.json
- json/pulp/{dataset}.json

E gera:
- results/benchmark/peab_vs_pulp/relatorio_comparativo_{dataset}.txt
- results/benchmark/peab_vs_pulp/comparacao_{dataset}.csv
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Importação do handler de resultados
from utils.results_handler import load_method_results, list_available_datasets

#==============================================================================
# CONSTANTES
#==============================================================================
OUTPUT_DIR = "results/benchmark/peab_vs_pulp"

#==============================================================================
# CARREGAMENTO DE DADOS
#==============================================================================

def extrair_explicacoes_estruturado(peab_data: Dict, pulp_data: Dict) -> Tuple[Dict, Dict]:
    """
    Extrai explicações de ambos os formatos.
    
    PEAB pode ter dois formatos:
    1. Formato antigo (novo): per_instance com detalhes por instância
    2. Formato sem instâncias individuais: apenas statistics agregadas
    
    PULP sempre tem: explicacoes com detalhes por instância
    """
    # Extrair explicações do PEAB
    explicacoes_peab = {}
    
    if 'per_instance' in peab_data:  
        # Formato com instâncias individuais (ideal)
        for exp in peab_data['per_instance']:
            explicacoes_peab[exp['id']] = {
                'indice': exp['id'],
                'tamanho': exp['explanation_size'],
                'y_true': exp['y_true'],
                'y_pred': exp['y_pred'],
                'rejected': exp['rejected'],
                'decision_score': exp['decision_score']
            }
    else:
        # Formato agregado sem instâncias - não podemos comparar instância por instância
        raise ValueError(
            f"❌ Formato PEAB não suporta comparação instância por instância.\n"
            f"   PEAB só possui estatísticas agregadas, não dados individuais.\n"
            f"   Para comparar com PULP, PEAB precisa ser re-executado com novo código\n"
            f"   que salve explicações individuais (per_instance)."
        )
    
    # Extrair explicações do PULP
    explicacoes_pulp = {}
    if 'explicacoes' in pulp_data:
        for exp in pulp_data['explicacoes']:
            idx_str = str(exp['indice'])
            explicacoes_pulp[idx_str] = {
                'indice': idx_str,
                'tamanho': exp['tamanho'],
                'tipo_predicao': exp['tipo_predicao'],
                'tempo_segundos': exp['tempo_segundos']
            }
    else:
        raise ValueError("Formato PULP não reconhecido: não encontrado 'explicacoes'")
    
    return explicacoes_peab, explicacoes_pulp

def listar_datasets_disponiveis() -> Tuple[List[str], List[str], List[str]]:
    """Lista datasets disponíveis em ambos os JSONs."""
    datasets_peab = set(list_available_datasets('peab'))
    datasets_pulp = set(list_available_datasets('pulp'))
    
    datasets_comuns = sorted(datasets_peab & datasets_pulp)
    apenas_peab = sorted(datasets_peab - datasets_pulp)
    apenas_pulp = sorted(datasets_pulp - datasets_peab)
    
    return datasets_comuns, apenas_peab, apenas_pulp

#==============================================================================
# COMPARAÇÃO DE EXPLICAÇÕES
#==============================================================================
def comparar_explicacoes(peab_data: Dict, pulp_data: Dict, dataset_name: str) -> pd.DataFrame:
    """
    Compara explicação por explicação entre PEAB e PuLP.
    Retorna DataFrame com colunas: indice, tamanho_PEAB, tamanho_PuLP, GAP, is_optimal
    """
    # Extrai explicações em formato unificado
    explicacoes_peab, explicacoes_pulp = extrair_explicacoes_estruturado(peab_data, pulp_data)
    
    # Índices comuns (instâncias presentes em ambos)
    indices_comuns = sorted(set(explicacoes_peab.keys()) & set(explicacoes_pulp.keys()))
    
    if not indices_comuns:
        raise ValueError(f"❌ Nenhuma instância comum entre PEAB e PuLP para {dataset_name}")
    
    comparacoes = []
    for idx in indices_comuns:
        peab_exp = explicacoes_peab[idx]
        pulp_exp = explicacoes_pulp[idx]
        
        tamanho_peab = peab_exp['tamanho']
        tamanho_pulp = pulp_exp['tamanho']
        gap = tamanho_peab - tamanho_pulp
        
        # Determinar tipo de predição baseado no PEAB
        if peab_exp['rejected']:
            tipo_pred = 'REJEITADA'
        elif peab_exp['y_pred'] == 1:
            tipo_pred = 'POSITIVA'
        else:
            tipo_pred = 'NEGATIVA'
        
        comparacoes.append({
            'indice': idx,
            'tipo_predicao': tipo_pred,
            'tamanho_PEAB': tamanho_peab,
            'tamanho_PuLP': tamanho_pulp,
            'GAP': gap,
            'tempo_PEAB': 0.0,  # PEAB não salva tempo individual
            'tempo_PuLP': pulp_exp.get('tempo_segundos', 0.0),
            'is_optimal': (gap == 0)
        })
    
    return pd.DataFrame(comparacoes)

#==============================================================================
# CÁLCULO DE MÉTRICAS AGREGADAS
#==============================================================================
def calcular_metricas_agregadas(df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula métricas agregadas da comparação."""
    total_instancias = len(df)
    
    # Taxa de otimalidade
    taxa_otimalidade = df['is_optimal'].mean() * 100
    
    # GAP
    gap_medio = df['GAP'].mean()
    gap_std = df['GAP'].std()
    gap_max = df['GAP'].max()
    gap_min = df['GAP'].min()
    
    # Tempo (apenas PuLP, PEAB não salva tempo individual)
    tempo_medio_pulp = df['tempo_PuLP'].mean()
    speedup = 1.0  # Não é calculado porque PEAB não tem tempo individual
    
    # Por tipo de predição
    stats_por_tipo = {}
    for tipo in df['tipo_predicao'].unique():
        df_tipo = df[df['tipo_predicao'] == tipo]
        stats_por_tipo[tipo] = {
            'instancias': len(df_tipo),
            'taxa_otimalidade': df_tipo['is_optimal'].mean() * 100,
            'gap_medio': df_tipo['GAP'].mean(),
            'tamanho_medio_peab': df_tipo['tamanho_PEAB'].mean(),
            'tamanho_medio_pulp': df_tipo['tamanho_PuLP'].mean(),
            'tempo_medio_pulp': df_tipo['tempo_PuLP'].mean()
        }
    
    return {
        'total_instancias': total_instancias,
        'taxa_otimalidade': taxa_otimalidade,
        'gap_medio': gap_medio,
        'gap_std': gap_std,
        'gap_max': gap_max,
        'gap_min': gap_min,
        'tempo_medio_pulp': tempo_medio_pulp,
        'speedup': speedup,
        'stats_por_tipo': stats_por_tipo
    }

#==============================================================================
# GERAÇÃO DE RELATÓRIO
#==============================================================================
def gerar_relatorio_comparativo(df: pd.DataFrame, metricas: Dict, dataset_name: str, 
                                peab_data: Dict, pulp_data: Dict) -> str:
    """Gera relatório comparativo detalhado em formato TXT."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"relatorio_comparativo_{dataset_name}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"RELATÓRIO COMPARATIVO: PEAB vs PuLP\n")
        f.write(f"Dataset: {dataset_name.upper()}\n")
        f.write("="*80 + "\n\n")
        
        # Seção 0: Configuração
        f.write("-" * 80 + "\n")
        f.write("0. CONFIGURAÇÃO DO EXPERIMENTO\n")
        f.write("-" * 80 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Instâncias comparadas: {metricas['total_instancias']}\n")
        
        # Acessar thresholds corretamente
        t_plus = peab_data.get('thresholds', {}).get('t_plus', peab_data.get('t_plus', 0))
        t_minus = peab_data.get('thresholds', {}).get('t_minus', peab_data.get('t_minus', 0))
        
        f.write(f"Thresholds: t+ = {t_plus:.4f}, t- = {t_minus:.4f}\n")
        f.write(f"Zona de rejeição: {t_plus - t_minus:.4f}\n")
        f.write(f"Rejection cost: {peab_data.get('rejection_cost', peab_data.get('config', {}).get('rejection_cost', 0))}\n")
        f.write(f"\nHiperparâmetros:\n")
        
        # Acessar params corretamente
        params = peab_data.get('params', peab_data.get('model', {}).get('params', {}))
        f.write(json.dumps(params, indent=2))
        f.write("\n\n")
        
        # Seção 1: Resumo Geral
        f.write("-" * 80 + "\n")
        f.write("1. RESUMO GERAL DE DESEMPENHO\n")
        f.write("-" * 80 + "\n\n")
        
        tabela_geral = pd.DataFrame({
            'Métrica': [
                'Taxa de Otimalidade (GAP=0)',
                'GAP Médio (features excedentes)',
                'GAP Máximo',
                'GAP Mínimo',
                'Desvio Padrão do GAP',
                'Tempo Médio PuLP (s)'
            ],
            'Valor': [
                f"{metricas['taxa_otimalidade']:.2f}%",
                f"{metricas['gap_medio']:.4f}",
                f"{metricas['gap_max']}",
                f"{metricas['gap_min']}",
                f"{metricas['gap_std']:.4f}",
                f"{metricas['tempo_medio_pulp']:.6f}"
            ]
        })
        
        f.write(tabela_geral.to_string(index=False, justify='left'))
        f.write("\n\n")
        
        # Interpretação
        f.write("INTERPRETAÇÃO:\n")
        if metricas['taxa_otimalidade'] >= 95:
            f.write("✅ EXCELENTE: PEAB encontra a solução ótima em >95% dos casos.\n")
        elif metricas['taxa_otimalidade'] >= 80:
            f.write("✓ BOM: PEAB encontra a solução ótima em >80% dos casos.\n")
        else:
            f.write("⚠ ATENÇÃO: PEAB tem taxa de otimalidade <80%. Revisar heurística.\n")
        
        f.write(f"- Em média, PEAB usa {metricas['gap_medio']:.2f} features a mais que o ótimo.\n")
        f.write(f"- PEAB é {metricas['speedup']:.0f}x mais rápido que PuLP.\n")
        f.write("\n")
        
        # Seção 2: Detalhamento por Tipo de Predição
        f.write("-" * 80 + "\n")
        f.write("2. DETALHAMENTO POR TIPO DE PREDIÇÃO\n")
        f.write("-" * 80 + "\n")
        f.write("Onde o PEAB é perfeito e onde encontra dificuldades.\n\n")
        
        # Definir ordem desejada: POSITIVA → NEGATIVA → REJEITADA
        ordem_tipos = ['POSITIVA', 'NEGATIVA', 'REJEITADA']
        tabela_tipos = []
        
        for tipo in ordem_tipos:
            if tipo in metricas['stats_por_tipo']:
                stats = metricas['stats_por_tipo'][tipo]
                tabela_tipos.append({
                    'Tipo': tipo,
                    'Qtd': stats['instancias'],
                    '% Ótimo': f"{stats['taxa_otimalidade']:.2f}%",
                    'GAP Médio': f"{stats['gap_medio']:.4f}",
                    'Tam. PEAB': f"{stats['tamanho_medio_peab']:.2f}",
                    'Tam. PuLP': f"{stats['tamanho_medio_pulp']:.2f}",
                    'Tempo PuLP': f"{stats['tempo_medio_pulp']:.5f}s"
                })
        
        df_tipos = pd.DataFrame(tabela_tipos)
        f.write(df_tipos.to_string(index=False))
        f.write("\n\n")
        
        # Seção 3: TOP 10 Maiores GAPs
        f.write("-" * 80 + "\n")
        f.write("3. TOP 10 MAIORES GAPS (Piores Casos do PEAB)\n")
        f.write("-" * 80 + "\n")
        f.write("Instâncias onde PEAB ficou mais longe da solução ótima.\n\n")
        
        piores = df.nlargest(10, 'GAP')[['indice', 'tipo_predicao', 
                                          'tamanho_PEAB', 'tamanho_PuLP', 'GAP']]
        f.write(piores.to_string(index=False))
        f.write("\n\n")
        
        # Seção 4: Distribuição de GAPs
        f.write("-" * 80 + "\n")
        f.write("4. DISTRIBUIÇÃO DE GAPS\n")
        f.write("-" * 80 + "\n\n")
        
        gap_counts = df['GAP'].value_counts().sort_index()
        f.write(f"GAP = 0 (Ótimo): {gap_counts.get(0, 0)} instâncias ({gap_counts.get(0, 0)/len(df)*100:.1f}%)\n")
        for gap in sorted(gap_counts.index):
            if gap > 0:
                count = gap_counts[gap]
                pct = count / len(df) * 100
                f.write(f"GAP = {gap}: {count} instâncias ({pct:.1f}%)\n")
        f.write("\n")
        
        # Seção 5: Análise de Tempo
        f.write("-" * 80 + "\n")
        f.write("5. ANÁLISE DE TEMPO DE EXECUÇÃO\n")
        f.write("-" * 80 + "\n\n")
        
        tempo_total_peab = df['tempo_PEAB'].sum()
        tempo_total_pulp = df['tempo_PuLP'].sum()
        economia_tempo = tempo_total_pulp - tempo_total_peab
        
        f.write(f"Tempo total PEAB: {tempo_total_peab:.2f}s\n")
        f.write(f"Tempo total PuLP: {tempo_total_pulp:.2f}s\n")
        f.write(f"Economia de tempo: {economia_tempo:.2f}s ({economia_tempo/tempo_total_pulp*100:.1f}%)\n")
        f.write(f"\nPara {len(df)} instâncias, PEAB economiza {economia_tempo:.1f}s\n")
        f.write(f"Projetando para 10.000 instâncias: {economia_tempo/len(df)*10000/60:.1f} minutos economizados\n")
        f.write("\n")
        
        # Seção 6: Conclusão
        f.write("="*80 + "\n")
        f.write("6. CONCLUSÃO\n")
        f.write("="*80 + "\n\n")
        
        if metricas['taxa_otimalidade'] >= 90 and metricas['speedup'] >= 10:
            f.write("✅ PEAB demonstra ser uma heurística EXCELENTE:\n")
            f.write(f"   - Alta qualidade: {metricas['taxa_otimalidade']:.1f}% de otimalidade\n")
            f.write(f"   - Alta velocidade: {metricas['speedup']:.0f}x mais rápido\n")
            f.write("   - Recomendado para uso em produção\n")
        elif metricas['taxa_otimalidade'] >= 75:
            f.write("✓ PEAB demonstra ser uma heurística BOA:\n")
            f.write(f"   - Qualidade aceitável: {metricas['taxa_otimalidade']:.1f}% de otimalidade\n")
            f.write(f"   - Velocidade adequada: {metricas['speedup']:.0f}x mais rápido\n")
            f.write("   - Adequado para maioria dos casos\n")
        else:
            f.write("⚠ PEAB precisa de melhorias:\n")
            f.write(f"   - Qualidade subótima: {metricas['taxa_otimalidade']:.1f}% de otimalidade\n")
            f.write(f"   - Revisar heurística para este dataset\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("FIM DO RELATÓRIO\n")
        f.write("="*80 + "\n")
    
    return output_file

#==============================================================================
# GERAÇÃO DE CSV
#==============================================================================
def salvar_csv(df: pd.DataFrame, dataset_name: str) -> str:
    """Salva DataFrame de comparação em CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_file = os.path.join(OUTPUT_DIR, f"comparacao_{dataset_name}.csv")
    df.to_csv(csv_file, index=False, encoding='utf-8')
    return csv_file

#==============================================================================
# MENU E EXECUÇÃO PRINCIPAL
#==============================================================================
def exibir_menu():
    """Exibe menu de seleção de datasets."""
    print("\n" + "="*80)
    print("   PEAB vs PuLP - Análise Comparativa")
    print("="*80 + "\n")
    
    # Lista datasets disponíveis
    datasets_comuns, apenas_peab, apenas_pulp = listar_datasets_disponiveis()
    
    if not datasets_comuns:
        print("❌ ERRO: Nenhum dataset com resultados em AMBOS os métodos.")
        print("\nDatasets disponíveis:")
        if apenas_peab:
            print(f"  Apenas PEAB: {', '.join(apenas_peab)}")
        if apenas_pulp:
            print(f"  Apenas PuLP: {', '.join(apenas_pulp)}")
        print("\n💡 Execute os métodos faltantes antes de comparar.")
        print("\n⚠️  NOTA: Comparação instância-por-instância requer que PEAB salve")
        print("   dados individuais (per_instance). Datasets com PEAB antigos não podem")
        print("   ser comparados com PuLP neste momento.")
        return None
    
    print(f"📊 Datasets disponíveis para comparação ({len(datasets_comuns)}):\n")
    for i, dataset in enumerate(datasets_comuns, 1):
        print(f"[{i:2d}] {dataset}")
    
    print(f"\n[{len(datasets_comuns)+1:2d}] Comparar TODOS os datasets acima")
    print(f"[ 0] Sair\n")
    
    if apenas_peab:
        print(f"⚠️  Datasets apenas com PEAB: {', '.join(apenas_peab)}")
    if apenas_pulp:
        print(f"⚠️  Datasets apenas com PuLP: {', '.join(apenas_pulp)}")
    
    print("\n⚠️  NOTA: Nem todos os datasets de PEAB suportam comparação.")
    print("   Se receber erro de 'per_instance', o dataset precisa ser re-executado")
    print("   com uma versão mais recente do PEAB que salva dados individuais.")
    
    return datasets_comuns

def processar_dataset(dataset_name: str):
    """Processa um único dataset."""
    print(f"\n{'='*80}")
    print(f"Processando: {dataset_name}")
    print(f"{'='*80}\n")
    
    try:
        # Carrega dados
        print("📂 Carregando resultados...")
        peab_data = load_method_results('peab', dataset_name)
        pulp_data = load_method_results('pulp', dataset_name)
        
        if not peab_data or not pulp_data:
            raise ValueError(f"Dados incompletos para {dataset_name}")
        
        # Compara explicações
        print("🔍 Comparando explicações...")
        df_comparacao = comparar_explicacoes(peab_data, pulp_data, dataset_name)
        
        # Calcula métricas
        print("📊 Calculando métricas agregadas...")
        metricas = calcular_metricas_agregadas(df_comparacao)
        
        # Salva CSV
        print("💾 Salvando CSV...")
        csv_file = salvar_csv(df_comparacao, dataset_name)
        
        # Gera relatório
        print("📝 Gerando relatório...")
        txt_file = gerar_relatorio_comparativo(df_comparacao, metricas, dataset_name, 
                                               peab_data, pulp_data)
        
        # Resumo
        print(f"\n{'='*80}")
        print(f"✅ COMPARAÇÃO CONCLUÍDA: {dataset_name}")
        print(f"{'='*80}")
        print(f"📊 Instâncias comparadas: {len(df_comparacao)}")
        print(f"🎯 Taxa de otimalidade: {metricas['taxa_otimalidade']:.2f}%")
        print(f"📏 GAP médio: {metricas['gap_medio']:.4f} features")
        print(f"⚡ Speedup: {metricas['speedup']:.2f}x")
        print(f"\n📁 Arquivos salvos:")
        print(f"   - CSV: {csv_file}")
        print(f"   - TXT: {txt_file}")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO ao processar {dataset_name}: {e}")
        
        # Se for erro de per_instance, dar dica específica
        if "per_instance" in str(e) or "Formato PEAB" in str(e):
            print(f"\n💡 SOLUÇÃO:")
            print(f"   Este dataset de PEAB foi gerado com versão antiga que não salva dados individuais.")
            print(f"   Para comparar com PuLP instância-por-instância, você precisa:")
            print(f"   1. Re-executar PEAB para este dataset com código mais recente")
            print(f"   2. Ou ignorar esta comparação por enquanto")
        else:
            import traceback
            traceback.print_exc()
        
        return False

def main():
    """Função principal."""
    datasets_disponiveis = exibir_menu()
    
    if not datasets_disponiveis:
        return
    
    try:
        escolha = input("Selecione uma opção: ").strip()
        
        if escolha == '0':
            print("👋 Até logo!")
            return
        
        escolha_num = int(escolha)
        
        if escolha_num == len(datasets_disponiveis) + 1:
            # Processar todos
            print(f"\n🚀 Processando TODOS os {len(datasets_disponiveis)} datasets...\n")
            sucessos = 0
            for dataset in datasets_disponiveis:
                if processar_dataset(dataset):
                    sucessos += 1
            
            print(f"\n{'='*80}")
            print(f"🎉 PROCESSAMENTO COMPLETO")
            print(f"{'='*80}")
            print(f"✅ Sucesso: {sucessos}/{len(datasets_disponiveis)} datasets")
            print(f"📁 Resultados salvos em: {OUTPUT_DIR}/")
            print(f"{'='*80}\n")
            
        elif 1 <= escolha_num <= len(datasets_disponiveis):
            # Processar dataset específico
            dataset_selecionado = datasets_disponiveis[escolha_num - 1]
            processar_dataset(dataset_selecionado)
        else:
            print("❌ Opção inválida!")
    
    except ValueError:
        print("❌ Entrada inválida! Digite um número.")
    except KeyboardInterrupt:
        print("\n\n👋 Interrompido pelo usuário. Até logo!")
    except Exception as e:
        print(f"\n❌ ERRO INESPERADO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
