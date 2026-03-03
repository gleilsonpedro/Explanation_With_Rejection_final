"""
Script otimizado para rodar Anchor/MinExp no MNIST com feedback visual aprimorado.
Mostra progresso detalhado e tempo estimado.
"""

import sys
import os

def rodar_anchor_mnist_otimizado():
    """Roda Anchor no MNIST com configurações otimizadas e feedback visual."""
    print("\n" + "="*80)
    print("EXECUTANDO: ANCHOR + MNIST (OTIMIZADO)")
    print("="*80)
    
    # Importar após configurar MNIST
    from data.datasets import set_mnist_options, selecionar_dataset_e_classe
    from utils.shared_training import get_shared_pipeline
    from alibi.explainers import AnchorTabular
    from utils.results_handler import update_method_results
    import numpy as np
    import time
    from collections import Counter
    
    # Configurar MNIST (3 vs 8 é o padrão)
    set_mnist_options('raw', (3, 8))
    nome_dataset = 'mnist'
    
    print(f"\n📊 Carregando dataset e treinando modelo...")
    pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(nome_dataset)
    
    feature_names = meta['feature_names']
    nome_relatorio = f"mnist_3_vs_8"
    
    print(f"\n✓ Dataset carregado:")
    print(f"   - Instâncias de teste: {len(X_test)}")
    print(f"   - Features: {len(feature_names)}")
    print(f"   - Thresholds: t⁻={t_minus:.4f}, t⁺={t_plus:.4f}")
    
    # Limitar a 200 instâncias para evitar tempo excessivo
    MAX_INSTANCES = 200
    if len(X_test) > MAX_INSTANCES:
        print(f"\n⚠️  Limitando a {MAX_INSTANCES} instâncias (de {len(X_test)}) para viabilizar execução")
        X_test = X_test[:MAX_INSTANCES]
        y_test = y_test[:MAX_INSTANCES]
    
    # Configurar explainer
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['model']
    
    def predict_fn(x):
        arr = x.reshape(1, -1) if x.ndim == 1 else x
        x_scaled = scaler.transform(arr)
        return model.predict_proba(x_scaled)
    
    print(f"\n🔧 Configurando Anchor Explainer...")
    explainer = AnchorTabular(predict_fn, feature_names=feature_names)
    explainer.fit(X_train.values if hasattr(X_train, 'values') else X_train, disc_perc=(25, 50, 75))
    
    # Calcular decisões e rejeições
    scores = pipeline.decision_function(X_test)
    y_pred = []
    indices_rej = []
    for i, s in enumerate(scores):
        if s >= t_plus:
            y_pred.append(1)
        elif s <= t_minus:
            y_pred.append(0)
        else:
            y_pred.append(-1)
            indices_rej.append(i)
    
    y_pred = np.array(y_pred)
    
    print(f"\n📊 Distribuição:")
    print(f"   - Positivas: {np.sum(y_pred == 1)}")
    print(f"   - Negativas: {np.sum(y_pred == 0)}")
    print(f"   - Rejeitadas: {len(indices_rej)}")
    
    print(f"\n🚀 Gerando explicações...")
    print(f"   Parâmetros otimizados para MNIST:")
    print(f"   - threshold: 0.90 (permite 10% erro)")
    print(f"   - batch_size: 200 (amostras por iteração)")
    print(f"   - max_anchor_size: 6 (pixels máximos)")
    print(f"   - beam_size: 2 (exploração limitada)")
    
    from utils.progress_bar import ProgressBar
    
    explicacoes = {}
    tempos = []
    tamanhos = []
    
    tempo_total_inicio = time.perf_counter()
    
    with ProgressBar(total=len(X_test), description="Anchor MNIST") as pbar:
        for i in range(len(X_test)):
            instance = X_test.iloc[i].values if hasattr(X_test, 'iloc') else X_test[i]
            
            start = time.perf_counter()
            try:
                explanation = explainer.explain(
                    instance,
                    threshold=0.90,
                    delta=0.15,
                    batch_size=200,
                    max_anchor_size=6,
                    beam_size=2
                )
                tempo = time.perf_counter() - start
                tempos.append(tempo)
                explicacoes[i] = explanation.anchor
                tamanhos.append(len(explanation.anchor))
                
                # Atualizar postfix com estatísticas
                if i % 10 == 0 and i > 0:
                    tempo_medio = np.mean(tempos)
                    tempo_restante = tempo_medio * (len(X_test) - i - 1)
                    pbar.set_postfix({
                        'tempo_médio': f'{tempo_medio:.1f}s',
                        'restante': f'{tempo_restante/60:.1f}min'
                    })
                
            except Exception as e:
                print(f"\n⚠️  Erro na instância {i}: {str(e)[:50]}")
                tempos.append(0)
                explicacoes[i] = []
                tamanhos.append(0)
            
            pbar.update()
    
    tempo_total = time.perf_counter() - tempo_total_inicio
    
    print(f"\n✓ Explicações geradas com sucesso!")
    print(f"\n📈 ESTATÍSTICAS:")
    print(f"   - Tempo total: {tempo_total:.0f}s ({tempo_total/60:.1f} minutos)")
    print(f"   - Tempo médio: {np.mean(tempos):.2f}s por instância")
    print(f"   - Tamanho médio: {np.mean(tamanhos):.2f} features")
    print(f"   - Explicações geradas: {len([e for e in explicacoes.values() if len(e) > 0])}/{len(X_test)}")
    
    # Salvar resultados
    print(f"\n💾 Salvando resultados em json/anchor/mnist_3_vs_8.json...")
    
    # Construir estrutura de resultados igual aos outros métodos
    resultados = {
        'dataset_name': nome_relatorio,
        'tempo_total': float(tempo_total),
        'tempo_medio_instancia': float(np.mean(tempos)) if tempos else 0.0,
        'num_explicacoes': len(explicacoes),
        'tamanho_medio': float(np.mean(tamanhos)) if tamanhos else 0.0
    }
    
    try:
        update_method_results('anchor', 'mnist_3_vs_8', resultados)
        print(f"✓ Resultados salvos!")
    except Exception as e:
        print(f"⚠️  Erro ao salvar: {e}")
    
    print(f"\n{'='*80}")
    print(f"EXECUÇÃO CONCLUÍDA!")
    print(f"{'='*80}\n")


def rodar_minexp_mnist_otimizado():
    """Roda MinExp no MNIST com configurações otimizadas e feedback visual."""
    print("\n" + "="*80)
    print("EXECUTANDO: MINEXP + MNIST (OTIMIZADO)")
    print("="*80)
    
    from data.datasets import set_mnist_options
    from utils.shared_training import get_shared_pipeline
    import utils.svm_explainer
    from utils.results_handler import update_method_results
    import numpy as np
    import time
    
    # Configurar MNIST
    set_mnist_options('raw', (3, 8))
    nome_dataset = 'mnist'
    
    print(f"\n📊 Carregando dataset e treinando modelo...")
    pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline(nome_dataset)
    
    feature_names = meta['feature_names']
    
    # Limitar a 200 instâncias
    MAX_INSTANCES = 200
    if len(X_test) > MAX_INSTANCES:
        print(f"\n⚠️  Limitando a {MAX_INSTANCES} instâncias (de {len(X_test)}) para viabilizar execução")
        X_test = X_test[:MAX_INSTANCES]
        y_test = y_test[:MAX_INSTANCES]
    
    print(f"\n✓ Dataset carregado:")
    print(f"   - Instâncias de teste: {len(X_test)}")
    print(f"   - Features: {len(feature_names)}")
    print(f"   - Thresholds: t⁻={t_minus:.4f}, t⁺={t_plus:.4f}")
    
    # Preparar solver
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['model']
    
    dual_coef = model.coef_
    w_solver = dual_coef[0] if dual_coef.ndim > 1 else dual_coef
    intercept = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_
    
    X_test_scaled = scaler.transform(X_test)
    lower_bound = X_test_scaled.min(axis=0)
    upper_bound = X_test_scaled.max(axis=0)
    
    # Calcular decisões
    scores = pipeline.decision_function(X_test)
    pos_idx = np.where(scores >= t_plus)[0]
    neg_idx = np.where(scores <= t_minus)[0]
    rej_idx = np.where((scores > t_minus) & (scores < t_plus))[0]
    
    print(f"\n📊 Distribuição:")
    print(f"   - Positivas: {len(pos_idx)}")
    print(f"   - Negativas: {len(neg_idx)}")
    print(f"   - Rejeitadas: {len(rej_idx)}")
    
    print(f"\n🚀 Gerando explicações (timeout 30s por instância)...")
    
    from utils.progress_bar import ProgressBar
    
    explicacoes = {}
    tempos = {}
    
    tempo_total_inicio = time.perf_counter()
    total_instances = len(X_test)
    
    with ProgressBar(total=total_instances, description="MinExp MNIST") as pbar:
        # Processar negativas
        if len(neg_idx) > 0:
            for i in neg_idx:
                start = time.perf_counter()
                try:
                    exp = utils.svm_explainer.svm_explanation_negative(
                        dual_coef=dual_coef,
                        support_vectors=np.array([w_solver]),
                        intercept=intercept,
                        t_lower=t_minus,
                        t_upper=t_plus,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        data=X_test_scaled[[i]],
                        show_log=0,
                        n_threads=1,
                        time_limit=30.0
                    )
                    tempos[i] = time.perf_counter() - start
                    explicacoes[i] = exp[0] if exp else []
                except:
                    tempos[i] = 0
                    explicacoes[i] = []
                pbar.update()
        
        # Processar positivas
        if len(pos_idx) > 0:
            for i in pos_idx:
                start = time.perf_counter()
                try:
                    exp = utils.svm_explainer.svm_explanation_positive(
                        dual_coef=dual_coef,
                        support_vectors=np.array([w_solver]),
                        intercept=intercept,
                        t_lower=t_minus,
                        t_upper=t_plus,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        data=X_test_scaled[[i]],
                        show_log=0,
                        n_threads=1,
                        time_limit=30.0
                    )
                    tempos[i] = time.perf_counter() - start
                    explicacoes[i] = exp[0] if exp else []
                except:
                    tempos[i] = 0
                    explicacoes[i] = []
                pbar.update()
        
        # Processar rejeitadas
        if len(rej_idx) > 0:
            for i in rej_idx:
                start = time.perf_counter()
                try:
                    exp = utils.svm_explainer.svm_explanation_rejected(
                        dual_coef=dual_coef,
                        support_vectors=np.array([w_solver]),
                        intercept=intercept,
                        t_lower=t_minus,
                        t_upper=t_plus,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        data=X_test_scaled[[i]],
                        show_log=0,
                        n_threads=1,
                        time_limit=30.0
                    )
                    tempos[i] = time.perf_counter() - start
                    explicacoes[i] = exp[0] if exp else []
                except:
                    tempos[i] = 0
                    explicacoes[i] = []
                pbar.update()
    
    tempo_total = time.perf_counter() - tempo_total_inicio
    tamanhos = [len(e) for e in explicacoes.values()]
    
    print(f"\n✓ Explicações geradas com sucesso!")
    print(f"\n📈 ESTATÍSTICAS:")
    print(f"   - Tempo total: {tempo_total:.0f}s ({tempo_total/60:.1f} minutos)")
    print(f"   - Tempo médio: {np.mean(list(tempos.values())):.2f}s por instância")
    print(f"   - Tamanho médio: {np.mean(tamanhos):.2f} features")
    
    print(f"\n💾 Salvando resultados...")
    resultados = {
        'dataset_name': 'mnist_3_vs_8',
        'tempo_total': float(tempo_total),
        'tempo_medio_instancia': float(np.mean(list(tempos.values()))),
        'tamanho_medio': float(np.mean(tamanhos))
    }
    
    try:
        update_method_results('minexp', 'mnist_3_vs_8', resultados)
        print(f"✓ Resultados salvos!")
    except Exception as e:
        print(f"⚠️  Erro ao salvar: {e}")
    
    print(f"\n{'='*80}")
    print(f"EXECUÇÃO CONCLUÍDA!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    print("╔" + "="*78 + "╗")
    print("║" + "EXECUTOR OTIMIZADO: MNIST + ANCHOR/MINEXP".center(78) + "║")
    print("╚" + "="*78 + "╝")
    
    print("\nEste script executa Anchor e MinExp no MNIST com:")
    print("  • Limite de 200 instâncias (viabiliza execução)")
    print("  • Timeout de 30s por instância (evita travamentos)")
    print("  • Barra de progresso com tempo estimado")
    print("  • Salvamento automático em JSON")
    
    print("\nTempo estimado:")
    print("  • Anchor: ~80 minutos (24s/instância × 200)")
    print("  • MinExp: ~100 minutos (30s/instância × 200)")
    
    print("\nEscolha o método:")
    print("  1. Anchor")
    print("  2. MinExp")
    print("  3. Ambos (sequencial)")
    
    escolha = input("\nOpção (1/2/3): ").strip()
    
    if escolha == '1':
        rodar_anchor_mnist_otimizado()
    elif escolha == '2':
        rodar_minexp_mnist_otimizado()
    elif escolha == '3':
        rodar_anchor_mnist_otimizado()
        rodar_minexp_mnist_otimizado()
    else:
        print("Opção inválida!")
