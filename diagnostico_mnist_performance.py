"""
Diagnóstico de performance para MNIST nos métodos Anchor e MinExp.
Estima tempo por instância e tempo total esperado.
"""

import numpy as np
import time
from data.datasets import selecionar_dataset_e_classe, set_mnist_options
from utils.shared_training import get_shared_pipeline
from alibi.explainers import AnchorTabular
import utils.svm_explainer
import utils.utility

def estimar_anchor_mnist():
    """Estima tempo de execução do Anchor no MNIST."""
    print("\n" + "="*80)
    print("DIAGNÓSTICO: ANCHOR + MNIST")
    print("="*80)
    
    # Configurar MNIST
    set_mnist_options('raw', (3, 8))
    pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline('mnist')
    
    print(f"\n📊 Dataset: {X_test.shape[0]} instâncias, {X_test.shape[1]} features")
    print(f"   Configuração: MNIST 3 vs 8 (pixels brutos)")
    
    # Configurar Anchor com parâmetros otimizados
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['model']
    feature_names = meta['feature_names']
    
    def predict_fn(x):
        arr = x.reshape(1, -1) if x.ndim == 1 else x
        x_scaled = scaler.transform(arr)
        return model.predict_proba(x_scaled)
    
    explainer = AnchorTabular(predict_fn, feature_names=feature_names)
    explainer.fit(X_train.values if hasattr(X_train, 'values') else X_train, disc_perc=(25, 50, 75))
    
    # Testar 5 instâncias para estimar
    print(f"\n🔬 Testando 5 instâncias para estimar tempo médio...")
    tempos = []
    
    for i in range(min(5, len(X_test))):
        instance = X_test.iloc[i].values if hasattr(X_test, 'iloc') else X_test[i]
        
        start = time.perf_counter()
        try:
            explanation = explainer.explain(
                instance,
                threshold=0.90,  # Otimizado para MNIST
                delta=0.15,
                batch_size=200,
                max_anchor_size=6,
                beam_size=2
            )
            tempo = time.perf_counter() - start
            tempos.append(tempo)
            print(f"   Instância {i+1}: {tempo:.2f}s - Âncora: {len(explanation.anchor)} features")
        except Exception as e:
            print(f"   Instância {i+1}: ERRO - {str(e)[:50]}")
    
    if tempos:
        tempo_medio = np.mean(tempos)
        tempo_total_estimado = tempo_medio * len(X_test)
        
        print(f"\n📈 ESTIMATIVAS:")
        print(f"   Tempo médio por instância: {tempo_medio:.2f}s ({tempo_medio*1000:.0f}ms)")
        print(f"   Tempo total estimado: {tempo_total_estimado:.0f}s ({tempo_total_estimado/60:.1f} minutos)")
        print(f"   Total de instâncias: {len(X_test)}")
        
        if tempo_total_estimado > 3600:
            print(f"\n⚠️  ATENÇÃO: Tempo estimado > 1 hora!")
            print(f"   Sugestão: Limitar a {min(200, len(X_test))} instâncias")
            tempo_limitado = tempo_medio * min(200, len(X_test))
            print(f"   Tempo limitado: {tempo_limitado:.0f}s ({tempo_limitado/60:.1f} minutos)")
    else:
        print("\n❌ Nenhuma explicação gerada com sucesso")
    
    return tempos


def estimar_minexp_mnist():
    """Estima tempo de execução do MinExp no MNIST."""
    print("\n" + "="*80)
    print("DIAGNÓSTICO: MINEXP + MNIST")
    print("="*80)
    
    # Configurar MNIST
    set_mnist_options('raw', (3, 8))
    pipeline, X_train, X_test, y_train, y_test, t_plus, t_minus, meta = get_shared_pipeline('mnist')
    
    print(f"\n📊 Dataset: {X_test.shape[0]} instâncias, {X_test.shape[1]} features")
    print(f"   Configuração: MNIST 3 vs 8 (pixels brutos)")
    
    # Preparar solver
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['model']
    
    # Obter coeficientes do modelo
    dual_coef = model.coef_
    w_solver = dual_coef[0] if dual_coef.ndim > 1 else dual_coef
    intercept = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_
    
    # Escalar dados
    X_test_scaled = scaler.transform(X_test)
    
    # Limites
    lower_bound = X_test_scaled.min(axis=0)
    upper_bound = X_test_scaled.max(axis=0)
    
    # Testar 5 instâncias positivas
    scores = pipeline.decision_function(X_test)
    pos_idx = np.where(scores >= t_plus)[0][:5]
    
    print(f"\n🔬 Testando {len(pos_idx)} instâncias POSITIVAS para estimar tempo médio...")
    tempos = []
    
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
                time_limit=30.0  # Limite para MNIST
            )
            tempo = time.perf_counter() - start
            tempos.append(tempo)
            print(f"   Instância {i}: {tempo:.2f}s - Explicação: {len(exp[0]) if exp else 0} features")
        except Exception as e:
            print(f"   Instância {i}: ERRO - {str(e)[:50]}")
    
    if tempos:
        tempo_medio = np.mean(tempos)
        tempo_total_estimado = tempo_medio * len(X_test)
        
        print(f"\n📈 ESTIMATIVAS:")
        print(f"   Tempo médio por instância: {tempo_medio:.2f}s")
        print(f"   Tempo total estimado: {tempo_total_estimado:.0f}s ({tempo_total_estimado/60:.1f} minutos)")
        print(f"   Total de instâncias: {len(X_test)}")
        
        if tempo_total_estimado > 3600:
            print(f"\n⚠️  ATENÇÃO: Tempo estimado > 1 hora!")
    else:
        print("\n❌ Nenhuma explicação gerada com sucesso")
    
    return tempos


def verificar_progresso():
    """Verifica se a barra de progresso está configurada corretamente."""
    print("\n" + "="*80)
    print("VERIFICAÇÃO: BARRA DE PROGRESSO")
    print("="*80)
    
    try:
        from utils.progress_bar import ProgressBar
        print("\n✓ ProgressBar importada com sucesso")
        
        # Testar barra de progresso
        print("\n🔬 Testando barra de progresso (10 iterações simuladas):")
        with ProgressBar(total=10, description="Teste") as pbar:
            for i in range(10):
                time.sleep(0.2)  # Simula processamento
                pbar.update()
        
        print("\n✓ Barra de progresso funcional!")
        
    except Exception as e:
        print(f"\n❌ ERRO na barra de progresso: {e}")


def solucoes_recomendadas():
    """Mostra soluções para executar MNIST."""
    print("\n" + "="*80)
    print("SOLUÇÕES RECOMENDADAS PARA MNIST")
    print("="*80)
    
    print("""
1. LIMITAR NÚMERO DE INSTÂNCIAS:
   - Anchor: Já tem limite de 200 instâncias no código (linha 175)
   - MinExp: Precisa adicionar limite similar
   
2. USAR SUBSAMPLING:
   - Editar data/datasets.py:
     # Linha ~50, adicionar na função load_mnist():
     subsample_size = 200  # Limitar dataset
   
3. RODAR EM BACKGROUND:
   - Execute em um terminal separado e deixe rodando overnight
   - Tempo estimado: 2-4 horas para 200 instâncias no Anchor
   
4. USAR FEATURES REDUZIDAS:
   - Mudar de 'raw' (784 features) para 'pca' (50 features)
   - Em data/datasets.py, linha ~40:
     set_mnist_options('pca', (3, 8))  # Ao invés de 'raw'
   
5. AUMENTAR TIMEOUT:
   - MinExp já tem time_limit=30s para MNIST
   - Se travar, pode ser que esteja no limite e demore mesmo
   
6. VERIFICAR SE ESTÁ REALMENTE PARADO:
   - CPU deve estar em ~25% (1 thread ativa)
   - Se CPU=0%, pode estar travado
   - Use Ctrl+C e tente novamente

PARA SEU PROFESSOR:
"O MNIST tem 784 features (pixels), então cada explicação demora ~24s no Anchor
e ~30s no MinExp (com timeout). Para 200 instâncias: ~1.5h Anchor, ~2h MinExp.
Isso é normal para datasets de alta dimensionalidade. Os outros 7 datasets
rodam em minutos porque têm 4-60 features apenas."

COMANDO RECOMENDADO (com limite):
- Anchor: Já limita automaticamente em 200 instâncias
- MinExp: Adicionar limite no início do script principal
""")


if __name__ == "__main__":
    print("╔" + "="*78 + "╗")
    print("║" + "DIAGNÓSTICO DE PERFORMANCE: MNIST + ANCHOR/MINEXP".center(78) + "║")
    print("╚" + "="*78 + "╝")
    
    print("\nEste script vai:")
    print("  1. Verificar a barra de progresso")
    print("  2. Estimar tempo para Anchor + MNIST (5 instâncias de teste)")
    print("  3. Estimar tempo para MinExp + MNIST (5 instâncias de teste)")
    print("  4. Mostrar soluções para execução completa")
    
    input("\nPressione ENTER para continuar...")
    
    # 1. Verificar barra de progresso
    verificar_progresso()
    
    # 2. Estimar Anchor
    try:
        tempos_anchor = estimar_anchor_mnist()
    except Exception as e:
        print(f"\n❌ Erro ao testar Anchor: {e}")
        tempos_anchor = []
    
    # 3. Estimar MinExp
    try:
        tempos_minexp = estimar_minexp_mnist()
    except Exception as e:
        print(f"\n❌ Erro ao testar MinExp: {e}")
        tempos_minexp = []
    
    # 4. Soluções
    solucoes_recomendadas()
    
    print("\n" + "="*80)
    print("DIAGNÓSTICO COMPLETO!")
    print("="*80)
