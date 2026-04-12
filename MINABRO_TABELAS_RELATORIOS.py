import os
import json
import glob

def gerar_tabela_relatorio():
    # Diretório onde os JSONs estão salvos
    json_dir = 'json/minabro_mlp/'
    arquivos_json = glob.glob(os.path.join(json_dir, '*.json'))

    if not arquivos_json:
        print(f"Nenhum arquivo JSON encontrado no diretório: {json_dir}")
        return

    # Cabeçalho formatado para facilitar a leitura no terminal e a cópia para o LaTeX
    cabecalho = (
        f"{'Dataset':<15} | {'Testes':<6} | {'Feats':<5} | "
        f"{'Acc (s/Rej)':<12} | {'Acc (c/Rej)':<12} | {'Taxa Rej':<10} | "
        f"{'Fid. Abdut.':<11} | {'Tam. Médio':<10} | {'Tempo/Inst':<10}"
    )
    print("=" * len(cabecalho))
    print(cabecalho)
    print("-" * len(cabecalho))

    for caminho_arquivo in sorted(arquivos_json):
        try:
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                dados = json.load(f)

            # 1. dataset_name
            dataset_name = dados.get('config', {}).get('dataset_name', 'N/A')

            # Extração dos blocos principais
            perf_oraculo = dados.get('performance_oraculo_mlp', {})
            perf_explicacoes = dados.get('performance_explicacoes_locais', {})
            modelo = dados.get('model', {})
            tempo = dados.get('computation_time', {})

            # 2. num_test_instances
            num_testes = perf_oraculo.get('num_test_instances', 0)
            
            # 3. num_features
            num_features = modelo.get('num_features', 0)

            # 4, 5 e 6. Acurácias e Taxa de Rejeição (formatados como porcentagem)
            acc_sem_rej = perf_oraculo.get('accuracy_without_rejection', 0.0)
            acc_com_rej = perf_oraculo.get('accuracy_with_rejection', 0.0)
            taxa_rej = perf_oraculo.get('rejection_rate_global', 0.0)

            # 7. fidelidade_abdutiva
            fidelidade = perf_explicacoes.get('fidelity_rate_worst_case', 0.0)

            # 8. tam_medio_explicacao (Média ponderada)
            pos_dados = perf_explicacoes.get('positive', {})
            neg_dados = perf_explicacoes.get('negative', {})
            
            pos_tam_medio = pos_dados.get('mean_length', 0.0)
            pos_count = pos_dados.get('count', 0)
            
            neg_tam_medio = neg_dados.get('mean_length', 0.0)
            neg_count = neg_dados.get('count', 0)

            total_instancias_explicadas = pos_count + neg_count
            
            if total_instancias_explicadas > 0:
                tam_medio = ((pos_tam_medio * pos_count) + (neg_tam_medio * neg_count)) / total_instancias_explicadas
            else:
                tam_medio = 0.0

            # 9. tempo_medio_instancia
            tempo_medio = tempo.get('mean_per_instance', 0.0)

            # Formatação da linha
            linha = (
                f"{dataset_name:<15} | {num_testes:<6} | {num_features:<5} | "
                f"{acc_sem_rej:>11.2f}% | {acc_com_rej:>11.2f}% | {taxa_rej:>9.2f}% | "
                f"{fidelidade:>10.2f}% | {tam_medio:>10.2f} | {tempo_medio:>8.4f}s"
            )
            print(linha)

        except json.JSONDecodeError:
            print(f"Erro de formatação JSON no arquivo: {caminho_arquivo}")
        except Exception as e:
            print(f"Erro inesperado ao processar {caminho_arquivo}: {str(e)}")

    print("=" * len(cabecalho))

if __name__ == "__main__":
    gerar_tabela_relatorio()