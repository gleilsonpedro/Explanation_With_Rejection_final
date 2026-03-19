import os
import json
from collections import Counter

def gerar_relatorio_do_json(caminho_json: str):
    if not os.path.exists(caminho_json):
        print(f"[ERRO] Arquivo JSON não encontrado: {caminho_json}")
        return

    with open(caminho_json, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    # 1. Extração de Blocos do Novo Layout Blindado
    config = dados.get('config', {})
    dataset_name = config.get('dataset_name', 'Desconhecido')
    
    thresh_mlp = dados.get('thresholds_globais_mlp', {})
    perf_mlp = dados.get('performance_oraculo_mlp', {})
    perf_expl = dados.get('performance_explicacoes_locais', {})
    comp_time = dados.get('computation_time', {})
    instancias = dados.get('per_instance', [])

    # 2. Configuração de Saída
    nome_arquivo_txt = f"report_{dataset_name}.txt"
    pasta_saida = "results/report/minabro_mlp"
    os.makedirs(pasta_saida, exist_ok=True)
    caminho_txt = os.path.join(pasta_saida, nome_arquivo_txt)

    # 3. Processamento das Features mais frequentes
    todas_features_explicacao = []
    explicacoes_validas = 0
    
    for inst in instancias:
        if not inst.get('rejected', False) and inst.get('explanation_size', 0) > 0:
            todas_features_explicacao.extend(inst.get('explanation', []))
            explicacoes_validas += 1
            
    contagem_features = Counter(todas_features_explicacao)
    # Evita divisão por zero caso todas as decisões sejam incondicionais ou rejeitadas
    divisor_freq = explicacoes_validas if explicacoes_validas > 0 else 1 
    top_10 = contagem_features.most_common(10)

    # 4. Geração do Relatório TXT
    with open(caminho_txt, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("       RELATÓRIO DE ANÁLISE - MINABRO MLP COM REJEIÇÃO (ARQUITETURA BLINDADA)\n")
        f.write("="*80 + "\n\n")

        f.write("-" * 80 + "\n")
        f.write("1. CONFIGURAÇÃO DO EXPERIMENTO\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Dataset: {dataset_name}\n")
        f.write(f"  Instâncias de teste avaliadas: {perf_mlp.get('num_test_instances', 0)}\n")
        f.write(f"  Features Originais do Dataset: {dados.get('model', {}).get('num_features', 'N/A')}\n")
        f.write(f"  Custo de rejeição (WR): {config.get('rejection_cost', 'N/A'):.4f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("2. AVALIAÇÃO DA CAIXA-PRETA (ORÁCULO MLP)\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Limiares Globais: t+ = {thresh_mlp.get('t_plus_global', 0):.6f} | t- = {thresh_mlp.get('t_minus_global', 0):.6f}\n")
        f.write(f"  Largura da Zona de Rejeição: {thresh_mlp.get('rejection_zone_width', 0):.6f}\n")
        f.write(f"  Acurácia (Sem Rejeição): {perf_mlp.get('accuracy_without_rejection', 0):.2f}%\n")
        f.write(f"  Acurácia (Com Rejeição): {perf_mlp.get('accuracy_with_rejection', 0):.2f}%\n")
        f.write(f"  Taxa de Rejeição Global: {perf_mlp.get('rejection_rate_global', 0):.2f}%\n")
        f.write(f"  Instâncias Rejeitadas pela MLP: {perf_mlp.get('num_rejected', 0)} de {perf_mlp.get('num_test_instances', 0)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("3. DESEMPENHO DO EXPLICADOR (SURROGATE LOCAL)\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Fidelidade Abdutiva (Pior Cenário): {perf_expl.get('fidelity_rate_worst_case', 0):.2f}%\n")
        f.write("  *Nota: Garante matematicamente a invariância da classe nos limites do hiper-retângulo da explicação.\n\n")
        
        def escrever_estatisticas(nome, stats):
            f.write(f"  EXPLICAÇÕES {nome}:\n")
            if stats.get('count', 0) == 0:
                f.write("    Quantidade: 0\n\n")
                return
            f.write(f"    Quantidade: {stats.get('count', 0)}\n")
            f.write(f"    Tamanho médio: {stats.get('mean_length', 0):.2f} features\n")
            f.write(f"    Desvio padrão: {stats.get('std_length', 0):.2f}\n")
            f.write(f"    Mínimo: {stats.get('min_length', 0)} features\n")
            f.write(f"    Máximo: {stats.get('max_length', 0)} features\n\n")

        escrever_estatisticas("POSITIVAS (Classe 1)", perf_expl.get('positive', {}))
        escrever_estatisticas("NEGATIVAS (Classe 0)", perf_expl.get('negative', {}))

        f.write("-" * 80 + "\n")
        f.write("4. TEMPOS DE EXECUÇÃO\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Tempo total da extração abdutiva: {comp_time.get('total', 0):.4f}s\n")
        f.write(f"  Tempo médio por instância: {comp_time.get('mean_per_instance', 0):.4f}s\n\n")

        f.write("-" * 80 + "\n")
        f.write("5. TOP 10 FEATURES MAIS FREQUENTES NAS EXPLICAÇÕES\n")
        f.write("-" * 80 + "\n")
        if not top_10:
            f.write("  Nenhuma feature registrada (todas as decisões foram incondicionais ou rejeitadas).\n")
        else:
            for feature, count in top_10:
                freq_relativa = (count / divisor_freq) * 100
                f.write(f"  {feature}: {count} ocorrências ({freq_relativa:.1f}%)\n")
        f.write("\n" + "="*80 + "\n")

    print(f"[SUCESSO] Relatório txt blindado gerado em: {caminho_txt}")

if __name__ == '__main__':
    # Permite rodar o script isoladamente passando o caminho do JSON no terminal
    import sys
    if len(sys.argv) > 1:
        gerar_relatorio_do_json(sys.argv[1])
    else:
        print("Passe o caminho do arquivo JSON como argumento ou importe a função 'gerar_relatorio_do_json'.")