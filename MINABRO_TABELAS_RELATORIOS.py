import os
import json
import glob
import numpy as np

def gerar_tabelas_terminais():
    json_dir = 'json/minabro_mlp/'
    arquivos_json = sorted(glob.glob(os.path.join(json_dir, '*.json')))

    if not arquivos_json:
        print(f"[ERRO] Nenhum arquivo JSON encontrado no diretório: {json_dir}")
        return

    linhas_tabela1 = [] # Original (Padrão)
    linhas_tabela2 = [] # Nova (Professor)
    linhas_tabela3 = [] # Comparativa

    for caminho in arquivos_json:
        with open(caminho, 'r', encoding='utf-8') as f:
            dados = json.load(f)

        # Extração Básica do Oráculo
        dataset_name = dados.get('config', {}).get('dataset_name', 'N/A')
        num_features = dados.get('model', {}).get('num_features', 0)
        
        perf_mlp = dados.get('performance_oraculo_mlp', {})
        num_testes = perf_mlp.get('num_test_instances', 0)
        acc_sem = perf_mlp.get('accuracy_without_rejection', 0.0)
        acc_com = perf_mlp.get('accuracy_with_rejection', 0.0)
        taxa_rej = perf_mlp.get('rejection_rate_global', 0.0)
        
        comp_time = dados.get('computation_time', {})
        tempo_medio = comp_time.get('mean_per_instance', 0.0)

        # Iteração das Instâncias
        instancias = dados.get('per_instance', [])
        
        tam_padrao, tam_aprimorado, jaccard = [], [], []
        fid_padrao_hits, fid_aprimorado_hits = 0, 0
        count_padrao_acc, count_aprimorado_acc = 0, 0

        for inst in instancias:
            padrao = inst.get('padrao', {})
            aprimorado = inst.get('aprimorado', {})
            
            # Checa o método Padrão
            if padrao:
                if not padrao.get('rejected', False):
                    tam_padrao.append(padrao.get('size', 0))
                    count_padrao_acc += 1
                    if padrao.get('faithful', False): fid_padrao_hits += 1
            
            # Checa o método Aprimorado
            if aprimorado:
                if not aprimorado.get('rejected', False):
                    tam_aprimorado.append(aprimorado.get('size', 0))
                    count_aprimorado_acc += 1
                    if aprimorado.get('faithful', False): fid_aprimorado_hits += 1

            # Calcula Jaccard (Sobreposição de Features) se ambos não foram rejeitados
            if padrao and aprimorado and not padrao.get('rejected', False) and not aprimorado.get('rejected', False):
                exp_p = set(padrao.get('explanation', []))
                exp_a = set(aprimorado.get('explanation', []))
                if len(exp_p) > 0 or len(exp_a) > 0:
                    intersecao = len(exp_p.intersection(exp_a))
                    uniao = len(exp_p.union(exp_a))
                    jaccard.append(intersecao / uniao)

        # Consolidação Estatística
        mean_tam_p = np.mean(tam_padrao) if tam_padrao else 0.0
        mean_tam_a = np.mean(tam_aprimorado) if tam_aprimorado else 0.0
        
        fid_p = (fid_padrao_hits / count_padrao_acc * 100) if count_padrao_acc > 0 else 100.0
        fid_a = (fid_aprimorado_hits / count_aprimorado_acc * 100) if count_aprimorado_acc > 0 else 100.0
        
        jaccard_medio = (np.mean(jaccard) * 100) if jaccard else 100.0

        # Montagem Tabela 1
        linhas_tabela1.append(
            f"{dataset_name:<15} | {num_testes:<6} | {num_features:<5} | "
            f"{acc_sem:>10.2f}% | {acc_com:>10.2f}% | {taxa_rej:>8.2f}% | "
            f"{fid_p:>10.2f}% | {mean_tam_p:>10.2f} | {tempo_medio:>8.4f}s"
        )

        # Montagem Tabela 2
        linhas_tabela2.append(
            f"{dataset_name:<15} | {num_testes:<6} | {num_features:<5} | "
            f"{acc_sem:>10.2f}% | {acc_com:>10.2f}% | {taxa_rej:>8.2f}% | "
            f"{fid_a:>10.2f}% | {mean_tam_a:>10.2f} | {tempo_medio:>8.4f}s"
        )

        # Montagem Tabela 3
        linhas_tabela3.append(
            f"{dataset_name:<15} | {fid_p:>10.1f}% | {fid_a:>13.1f}% | "
            f"{mean_tam_p:>14.2f} | {mean_tam_a:>15.2f} | {jaccard_medio:>19.1f}%"
        )

    # =========================================================
    # IMPRESSÃO DAS TABELAS NO TERMINAL
    # =========================================================
    print("\n" + "="*112)
    print(" TABELA 1: RESULTADOS DO EXPERIMENTO ORIGINAL (TÉCNICA DO ESPELHO PADRÃO)")
    print("="*112)
    cabecalho1 = (
        f"{'Dataset':<15} | {'Testes':<6} | {'Feats':<5} | "
        f"{'Acc(s/Rej)':<11} | {'Acc(c/Rej)':<11} | {'Taxa Rej':<9} | "
        f"{'Fid. Abdut':<11} | {'Tam. Médio':<10} | {'Tempo/Inst':<10}"
    )
    print(cabecalho1)
    print("-" * len(cabecalho1))
    for l in linhas_tabela1: print(l)

    print("\n" + "="*112)
    print(" TABELA 2: RESULTADOS DO NOVO EXPERIMENTO (PIOR CASO ABDUTIVO - PROFESSOR)")
    print("="*112)
    print(cabecalho1)
    print("-" * len(cabecalho1))
    for l in linhas_tabela2: print(l)

    print("\n" + "="*104)
    print(" TABELA 3: COMPARATIVO DIRETO (PADRÃO VS. ABDUTIVO)")
    print("="*104)
    cabecalho3 = (
        f"{'Dataset':<15} | {'Fid. Padrão':<11} | {'Fid. Abdutiva':<14} | "
        f"{'Tam Médio Pad':<14} | {'Tam Médio Abdut':<15} | {'Sobreposição (Jaccard)':<22}"
    )
    print(cabecalho3)
    print("-" * len(cabecalho3))
    for l in linhas_tabela3: print(l)
    print("\n")

if __name__ == '__main__':
    gerar_tabelas_terminais()