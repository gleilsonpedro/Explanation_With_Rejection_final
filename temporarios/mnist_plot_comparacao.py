"""
Script para gerar visualizações comparativas de explicações MNIST
Suporta MNIST com ou sem pooling (detecta automaticamente o tamanho)
Gera imagens para PEAB e MinExp das mesmas instâncias para comparação válida

Funcionalidades:
1. Detecta automaticamente tamanho da imagem (28x28 original ou 14x14 com pooling)
2. Gera imagens para instâncias: positiva, negativa e rejeitada
3. Cria versões PEAB e MinExp da mesma instância
4. Salva em pastas separadas para facilitar comparação
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
RESULTS_DIR = 'json'
OUTPUT_DIR = 'results/plots/mnist/comparacao'
SAVE_PLOTS = True
SHOW_PLOTS = False

# Métodos a comparar
METODOS = ['peab', 'minexp']

# ==============================================================================
# 🎯 CONTROLE DE ÍNDICES - ESCOLHA AS INSTÂNCIAS ESPECÍFICAS
# ==============================================================================
# Deixe como None para seleção aleatória
# Ou defina o número do índice para fixar um exemplo específico

IDX_POSITIVA = None    # Ex: 104 para fixar um dígito 8 específico
IDX_NEGATIVA = None    # Ex: 14 para fixar um dígito 3 específico  
IDX_REJEITADA = None   # Ex: 6 para fixar uma instância rejeitada específica

# ==============================================================================
# FUNÇÕES
# ==============================================================================

def carregar_json(metodo: str) -> dict:
    """Carrega o arquivo JSON de resultados de um método"""
    # Peab e pulp usam mnist_3_vs_8, anchor e minexp usam mnist
    if metodo in ['peab', 'pulp']:
        filepath = f'{RESULTS_DIR}/{metodo}/mnist_3_vs_8.json'
    else:
        filepath = f'{RESULTS_DIR}/{metodo}/mnist.json'
    
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def detectar_tamanho_imagem(num_features: int) -> tuple:
    """
    Detecta o tamanho da imagem baseado no número de features
    
    Returns:
        tuple: (altura, largura) da imagem
    """
    # Tentar encontrar dimensões quadradas perfeitas
    sqrt = int(math.sqrt(num_features))
    if sqrt * sqrt == num_features:
        return (sqrt, sqrt)
    
    # Dimensões conhecidas do MNIST
    if num_features == 784:  # 28x28 original
        return (28, 28)
    elif num_features == 196:  # 14x14 com pooling 2x2
        return (14, 14)
    elif num_features == 100:  # 10x10 com pooling maior
        return (10, 10)
    else:
        # Tentar dimensões retangulares próximas
        for h in range(int(math.sqrt(num_features)), 0, -1):
            if num_features % h == 0:
                w = num_features // h
                return (h, w)
    
    raise ValueError(f"Não foi possível detectar dimensões para {num_features} features")


def recarregar_dataset(config: dict):
    """Recarrega o dataset original para obter X_test"""
    from data.datasets import carregar_dataset, set_mnist_options
    from sklearn.model_selection import train_test_split
    
    dataset_name = config.get('dataset_name', 'mnist')
    mnist_feature_mode = config.get('mnist_feature_mode', 'raw')
    mnist_digit_pair = config.get('mnist_digit_pair', [3, 8])
    test_size = config.get('test_size', 0.3)
    random_state = config.get('random_state', 42)
    
    # Configurar MNIST antes de carregar
    if dataset_name == 'mnist':
        set_mnist_options(mnist_feature_mode, tuple(mnist_digit_pair))
    
    # Carregar dataset
    X_full, y_full, class_names = carregar_dataset(dataset_name)
    if X_full is None or y_full is None:
        return None, None, None
    
    # Fazer split train/test
    _, X_test, _, _ = train_test_split(
        X_full, y_full, test_size=test_size, 
        random_state=random_state, stratify=y_full
    )
    
    return X_test, y_full, class_names


def _get_instance_vector(X_test, inst_idx: int, num_features: int) -> np.ndarray:
    """Retorna vetor de features da instância"""
    if isinstance(X_test, dict):
        pixel_keys = sorted(X_test.keys(), key=lambda x: int(x.replace('pixel', '')))
        num_instances = len(X_test[pixel_keys[0]])
        
        if inst_idx >= num_instances:
            return np.zeros(num_features)
        
        x_vals = np.zeros(num_features)
        for feat_idx, pixel_key in enumerate(pixel_keys):
            x_vals[feat_idx] = X_test[pixel_key][inst_idx]
        return x_vals
    else:
        X_arr = np.array(X_test)
        if inst_idx < X_arr.shape[0]:
            return X_arr[inst_idx]
        return np.zeros(num_features)


def criar_imagem_comparativa(inst: dict, inst_idx: int, X_test, 
                             class_names: list, metodo: str,
                             t_plus: float, t_minus: float,
                             img_shape: tuple, output_suffix: str = ""):
    """
    Cria uma imagem mostrando:
    1. Dígito original
    2. Overlay com os pixels da explicação destacados em vermelho
    
    Args:
        inst: Dicionário com dados da instância
        inst_idx: Índice sequencial da instância no X_test
        X_test: Dados de teste
        class_names: Nomes das classes
        metodo: Nome do método ('peab' ou 'minexp')
        t_plus: Threshold superior
        t_minus: Threshold inferior
        img_shape: Formato da imagem (ex: (14, 14) ou (28, 28))
        output_suffix: Sufixo para o nome do arquivo
    """
    
    num_features = img_shape[0] * img_shape[1]
    
    # Obter vetor de features
    x_vals = _get_instance_vector(X_test, inst_idx, num_features)
    
    # Normalizar para [0, 1]
    if x_vals.max() > 1.0:
        x_vals = x_vals / 255.0
    
    img_original = x_vals.reshape(img_shape)
    
    # Criar máscara da explicação
    explanation = inst.get('explanation', [])
    mask_binary = np.zeros(num_features, dtype=float)
    
    for feat in explanation:
        try:
            if isinstance(feat, str) and feat.startswith('pixel'):
                idx = int(feat.replace('pixel', '')) - 1
            elif isinstance(feat, (int, np.integer)):
                idx = int(feat) - 1
            else:
                idx = None
        except Exception:
            idx = None
        
        if idx is not None and 0 <= idx < num_features:
            mask_binary[idx] = 1.0

    mask_img = mask_binary.reshape(img_shape)
    
    # Determinar categoria e cor
    rejected = inst.get('rejected', False)
    y_pred = inst.get('y_pred', -1)
    y_true = inst.get('y_true', -1)
    decision_score = inst.get('decision_score', 0.0)
    
    if rejected:
        categoria = 'REJEITADA'
        cor_titulo = 'purple'
    elif y_pred == 1:
        categoria = f'POSITIVA (Classe {class_names[1]})'
        cor_titulo = 'blue'
    else:
        categoria = f'NEGATIVA (Classe {class_names[0]})'
        cor_titulo = 'red'
    
    # Criar figura
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Painel 1: Imagem original
    axes[0].imshow(img_original, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(
        f'Dígito Original\n'
        f'Classe Verdadeira: {class_names[y_true]} (y={y_true})',
        fontsize=11, fontweight='bold'
    )
    axes[0].axis('off')
    
    # Painel 2: Overlay com explicação em VERMELHO
    img_rgb = np.stack([img_original, img_original, img_original], axis=-1)
    
    # Aplicar overlay vermelho nos pixels da explicação
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if mask_img[i, j] > 0:
                alpha = 0.7
                img_rgb[i, j, 0] = min(1.0, img_rgb[i, j, 0] + alpha)  # Vermelho
                img_rgb[i, j, 1] = img_rgb[i, j, 1] * 0.3  # Reduzir verde
                img_rgb[i, j, 2] = img_rgb[i, j, 2] * 0.3  # Reduzir azul
    
    axes[1].imshow(img_rgb, interpolation='nearest')
    
    # Posição do score
    if decision_score >= t_plus:
        pos_score = f'> t+ ({t_plus:.3f})'
    elif decision_score <= t_minus:
        pos_score = f'< t- ({t_minus:.3f})'
    else:
        pos_score = f'entre t- ({t_minus:.3f}) e t+ ({t_plus:.3f})'
    
    metodo_nome = metodo.upper()
    axes[1].set_title(
        f'Explicação {metodo_nome}\n'
        f'Predito: {class_names[y_pred] if y_pred in [0,1] else "REJEITADA"} (y={y_pred})\n'
        f'Score: {decision_score:.3f} ({pos_score})\n'
        f'Pixels na explicação: {len(explanation)}',
        fontsize=10, fontweight='bold'
    )
    axes[1].axis('off')
    
    # Legenda
    fig.subplots_adjust(right=0.88)
    fig.text(0.91, 0.5, 
             'Legenda:\n\n'
             'Vermelho:\nPixels da\nexplicação\n(total: {})\n\n'
             'Cinza:\nDígito\noriginal'.format(len(explanation)),
             fontsize=9,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Título principal
    fig.suptitle(
        f'Exemplo de Instância {categoria} - Método: {metodo_nome}\n'
        f'MNIST 3 vs 8 | Thresholds: t- = {t_minus:.3f}, t+ = {t_plus:.3f} | Shape: {img_shape}',
        fontsize=12, fontweight='bold', color=cor_titulo, y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    
    # Salvar
    if SAVE_PLOTS:
        output_path = Path(OUTPUT_DIR) / metodo
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = output_path / f'mnist_idx{inst_idx}_{output_suffix}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  💾 Salvo: {filename}")
    
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def processar_comparacao():
    """
    Processa e gera imagens comparativas para PEAB e MinExp
    """
    print(f"\n{'='*80}")
    print(f"GERADOR DE VISUALIZAÇÕES COMPARATIVAS - MNIST")
    print(f"Comparando métodos: {', '.join([m.upper() for m in METODOS])}")
    print(f"{'='*80}\n")
    
    # Carregar dados de ambos os métodos
    dados_metodos = {}
    configs = {}
    
    for metodo in METODOS:
        try:
            print(f"📥 Carregando dados do {metodo.upper()}...")
            dados_metodos[metodo] = carregar_json(metodo)
            configs[metodo] = dados_metodos[metodo].get('config', {})
            print(f"  ✓ {len(dados_metodos[metodo].get('per_instance', []))} instâncias carregadas")
        except Exception as e:
            print(f"  ❌ Erro ao carregar {metodo}: {e}")
            return
    
    # Recarregar dataset (usar config do primeiro método)
    print("\n📥 Recarregando dataset original...")
    config = configs[METODOS[0]]
    X_test, _, class_names = recarregar_dataset(config)
    
    if X_test is None:
        print("❌ Erro ao recarregar dataset")
        return
    
    # Detectar tamanho da imagem
    num_features = X_test.shape[1] if hasattr(X_test, 'shape') else len(list(X_test.keys()))
    img_shape = detectar_tamanho_imagem(num_features)
    
    print(f"  ✓ Dataset recarregado: {X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test[list(X_test.keys())[0]])} instâncias")
    print(f"  ✓ Número de features: {num_features}")
    print(f"  ✓ Formato da imagem detectado: {img_shape}")
    
    # Processar cada método
    for metodo in METODOS:
        print(f"\n{'='*80}")
        print(f"Processando método: {metodo.upper()}")
        print(f"{'='*80}")
        
        data = dados_metodos[metodo]
        per_instance = data.get('per_instance', [])
        
        t_plus = data.get('thresholds', {}).get('t_plus', 0.0)
        t_minus = data.get('thresholds', {}).get('t_minus', 0.0)
        
        print(f"✓ Thresholds: t- = {t_minus:.4f}, t+ = {t_plus:.4f}")
        
        # Coletar candidatos
        candidatos_positiva = []
        candidatos_negativa = []
        candidatos_rejeitada = []
        
        for idx, inst in enumerate(per_instance):
            rejected = inst.get('rejected', False)
            y_pred = inst.get('y_pred', -1)
            explanation = inst.get('explanation', [])
            
            if len(explanation) == 0:
                continue
            
            if rejected:
                candidatos_rejeitada.append((idx, inst))
            elif y_pred == 1:
                candidatos_positiva.append((idx, inst))
            elif y_pred == 0:
                candidatos_negativa.append((idx, inst))
        
        print(f"✓ Candidatos: {len(candidatos_positiva)} positivas, {len(candidatos_negativa)} negativas, {len(candidatos_rejeitada)} rejeitadas")
        
        # Selecionar exemplos
        import random
        random.seed(42)
        
        exemplos = []
        
        # Positiva
        if IDX_POSITIVA is not None:
            encontrado = [c for c in candidatos_positiva if c[0] == IDX_POSITIVA]
            if encontrado:
                exemplos.append(('positiva', encontrado[0]))
                print(f"  ✓ Usando índice manual para positiva: {IDX_POSITIVA}")
        elif candidatos_positiva:
            exemplos.append(('positiva', random.choice(candidatos_positiva)))
        
        # Negativa
        if IDX_NEGATIVA is not None:
            encontrado = [c for c in candidatos_negativa if c[0] == IDX_NEGATIVA]
            if encontrado:
                exemplos.append(('negativa', encontrado[0]))
                print(f"  ✓ Usando índice manual para negativa: {IDX_NEGATIVA}")
        elif candidatos_negativa:
            exemplos.append(('negativa', random.choice(candidatos_negativa)))
        
        # Rejeitada
        if IDX_REJEITADA is not None:
            encontrado = [c for c in candidatos_rejeitada if c[0] == IDX_REJEITADA]
            if encontrado:
                exemplos.append(('rejeitada', encontrado[0]))
                print(f"  ✓ Usando índice manual para rejeitada: {IDX_REJEITADA}")
        elif candidatos_rejeitada:
            exemplos.append(('rejeitada', random.choice(candidatos_rejeitada)))
        
        # Gerar imagens
        print(f"\n📊 Gerando visualizações para {metodo.upper()}...")
        for tipo, (idx, inst) in exemplos:
            print(f"\n  📍 Processando instância {tipo.upper()} (idx={idx})...")
            criar_imagem_comparativa(
                inst, idx, X_test, class_names, metodo,
                t_plus, t_minus, img_shape, tipo
            )
    
    print(f"\n{'='*80}")
    print(f"✅ CONCLUÍDO!")
    print(f"{'='*80}")
    print(f"\nImagens salvas em: {Path(OUTPUT_DIR).resolve()}")
    print(f"Subpastas criadas para cada método: {', '.join(METODOS)}")
    print("\nPara comparar:")
    print(f"  - PEAB: {OUTPUT_DIR}/peab/")
    print(f"  - MinExp: {OUTPUT_DIR}/minexp/")


if __name__ == "__main__":
    processar_comparacao()
