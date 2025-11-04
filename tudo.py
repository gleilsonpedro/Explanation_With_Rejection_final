import time
import os
import traceback

# Experimentos disponíveis
try:
    from peab_2 import executar_experimento_para_dataset
except Exception as e:
    executar_experimento_para_dataset = None
    print(f"[AVISO] Não foi possível importar PEAB (peab_2.executar_experimento_para_dataset): {e}")

try:
    from minexp_comparation import run_minexp_for_dataset
except Exception as e:
    run_minexp_for_dataset = None
    print(f"[AVISO] Não foi possível importar MinExp (minexp_comparation.run_minexp_for_dataset): {e}")

try:
    from anchor_comparation import run_anchor_for_dataset
except Exception as e:
    run_anchor_for_dataset = None
    print(f"[AVISO] Não foi possível importar Anchor (anchor_comparation.run_anchor_for_dataset): {e}")


DATASETS = [
    # Ordem solicitada: nrest (breast_cancer), mnist, pima, sonar, vertebral, wine
    "breast_cancer",
    "mnist",
    "pima_indians_diabetes",
    "sonar",
    "vertebral_column",
    "wine",
]

EXPERIMENTS = [
    ("PEAB", executar_experimento_para_dataset),
    ("MinExp", run_minexp_for_dataset),
    ("Anchor", run_anchor_for_dataset),
]


def run_all():
    os.makedirs('results', exist_ok=True)
    log_path = os.path.join('results', 'run_all_log.txt')
    start_all = time.time()

    with open(log_path, 'w', encoding='utf-8') as log:
        log.write(f"==== Execução automática iniciada em {time.strftime('%Y-%m-%d %H:%M:%S')} ====" + "\n")
        for exp_name, exp_func in EXPERIMENTS:
            if exp_func is None:
                log.write(f"[SKIP] {exp_name}: função não disponível.\n")
                continue

            log.write(f"\n----- EXPERIMENTO: {exp_name} -----\n")
            print(f"\n===== Rodando experimento: {exp_name} =====")
            exp_start = time.time()

            for ds in DATASETS:
                ds_start = time.time()
                try:
                    print(f"[ {exp_name} ] Dataset: {ds} …")
                    result = exp_func(ds)
                    took = time.time() - ds_start
                    log.write(f"[OK] {exp_name} :: {ds} em {took:.2f}s\n")
                    if isinstance(result, dict):
                        # resumo opcional
                        report = result.get('report_path')
                        json_ds = result.get('json_updated_for')
                        if report:
                            log.write(f"   - Report: {report}\n")
                        if json_ds:
                            log.write(f"   - JSON atualizado para: {json_ds}\n")
                except KeyboardInterrupt:
                    log.write(f"[ABORTADO] {exp_name} :: {ds} – interrompido pelo usuário.\n")
                    print("Execução interrompida pelo usuário.")
                    return
                except Exception as e:
                    took = time.time() - ds_start
                    log.write(f"[ERRO] {exp_name} :: {ds} em {took:.2f}s → {e}\n")
                    trace = traceback.format_exc(limit=2)
                    log.write(trace + "\n")
                    print(f"[ERRO] {exp_name} :: {ds}: {e}")

            exp_took = time.time() - exp_start
            log.write(f"Tempo total {exp_name}: {exp_took:.2f}s\n")

        total_took = time.time() - start_all
        log.write(f"\n==== Finalizado em {total_took:.2f}s ====\n")

    print(f"\nTudo concluído. Log em: {log_path}")


if __name__ == "__main__":
    run_all()
