"""
Script to generate comparative visualizations of MNIST explanations
Supports MNIST with or without pooling (automatically detects size)
Generates images for MINABRO and MinExp of the same instances for valid comparison

Features:
1. Automatically detects image size (28x28 original or 14x14 with pooling)
2. Generates images for instances: positive, negative and rejected
3. Creates MINABRO and MinExp versions of the same instance
4. Saves in separate folders to facilitate comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# ==============================================================================
# SETTINGS
# ==============================================================================
RESULTS_DIR = 'json'
OUTPUT_DIR = 'results/plots/mnist/comparacao'
SAVE_PLOTS = True
SHOW_PLOTS = False

# Methods to compare
METODOS = ['peab', 'minexp']

# ==============================================================================
# 🎯 INDEX CONTROL - CHOOSE SPECIFIC INSTANCES
# ==============================================================================
# Leave as None for random selection
# Or set the index number to fix a specific example

IDX_POSITIVA = 9999   # Ex: 104 to fix a specific digit 8
IDX_NEGATIVA =  9999    # Ex: 14 to fix a specific digit 3  
IDX_REJEITADA = 33   # Ex: 6 to fix a specific rejected instance

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def carregar_json(metodo: str) -> dict:
    """Loads the JSON results file for a method"""
    # Peab e pulp usam mnist_3_vs_8, anchor e minexp usam mnist
    if metodo in ['peab', 'pulp']:
        filepath = f'{RESULTS_DIR}/{metodo}/mnist_3_vs_8.json'
    else:
        filepath = f'{RESULTS_DIR}/{metodo}/mnist.json'
    
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def detectar_tamanho_imagem(num_features: int) -> tuple:
    """
    Detects image size based on number of features
    
    Returns:
        tuple: (height, width) of the image
    """
    # Try to find perfect square dimensions
    sqrt = int(math.sqrt(num_features))
    if sqrt * sqrt == num_features:
        return (sqrt, sqrt)
    
    # Known MNIST dimensions
    if num_features == 784:  # 28x28 original
        return (28, 28)
    elif num_features == 196:  # 14x14 com pooling 2x2
        return (14, 14)
    elif num_features == 100:  # 10x10 com pooling maior
        return (10, 10)
    else:
        # Try nearest rectangular dimensions
        for h in range(int(math.sqrt(num_features)), 0, -1):
            if num_features % h == 0:
                w = num_features // h
                return (h, w)
    
    raise ValueError(f"Could not detect dimensions for {num_features} features")


def recarregar_dataset(config: dict):
    """Reloads the original dataset to obtain X_test"""
    from data.datasets import carregar_dataset, set_mnist_options
    from sklearn.model_selection import train_test_split
    
    dataset_name = config.get('dataset_name', 'mnist')
    mnist_feature_mode = config.get('mnist_feature_mode', 'raw')
    mnist_digit_pair = config.get('mnist_digit_pair', [3, 8])
    test_size = config.get('test_size', 0.3)
    random_state = config.get('random_state', 42)
    
    # Configure MNIST before loading
    if dataset_name == 'mnist':
        set_mnist_options(mnist_feature_mode, tuple(mnist_digit_pair))
    
    # Load dataset
    X_full, y_full, class_names = carregar_dataset(dataset_name)
    if X_full is None or y_full is None:
        return None, None, None
    
    # Perform train/test split
    _, X_test, _, _ = train_test_split(
        X_full, y_full, test_size=test_size, 
        random_state=random_state, stratify=y_full
    )
    
    return X_test, y_full, class_names


def _get_instance_vector(X_test, inst_idx: int, num_features: int) -> np.ndarray:
    """Returns the feature vector of the instance"""
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
    Creates an image showing:
    1. Original digit
    2. Overlay with explanation pixels highlighted in red
    
    Args:
        inst: Dictionary with instance data
        inst_idx: Sequential index of the instance in X_test
        X_test: Test data
        class_names: Class names
        metodo: Method name ('peab' or 'minexp')
        t_plus: Upper threshold
        t_minus: Lower threshold
        img_shape: Image shape (ex: (14, 14) or (28, 28))
        output_suffix: Suffix for the file name
    """
    
    num_features = img_shape[0] * img_shape[1]
    
    # Get feature vector
    x_vals = _get_instance_vector(X_test, inst_idx, num_features)
    
    # Normalize to [0, 1]
    if x_vals.max() > 1.0:
        x_vals = x_vals / 255.0
    
    img_original = x_vals.reshape(img_shape)
    
    # Create explanation mask
    explanation = inst.get('explanation', [])
    mask_binary = np.zeros(num_features, dtype=float)
    
    for feat in explanation:
        try:
            if isinstance(feat, str) and feat.startswith('pixel'):
                # Format: "pixel123" -> index 122
                idx = int(feat.replace('pixel', '')) - 1
            elif isinstance(feat, str) and feat.startswith('bin_'):
                # Format: "bin_10_5" -> row 10, column 5
                parts = feat.replace('bin_', '').split('_')
                row = int(parts[0])
                col = int(parts[1])
                # Convert coordinates (row, col) to linear index
                idx = row * img_shape[1] + col
            elif isinstance(feat, (int, np.integer)):
                idx = int(feat) - 1
            else:
                idx = None
        except Exception:
            idx = None
        
        if idx is not None and 0 <= idx < num_features:
            mask_binary[idx] = 1.0

    mask_img = mask_binary.reshape(img_shape)
    
    # Determine category and color
    rejected = inst.get('rejected', False)
    y_pred = inst.get('y_pred', -1)
    y_true = inst.get('y_true', -1)
    decision_score = inst.get('decision_score', 0.0)
    
    if rejected:
        categoria = 'REJECTED'
        cor_titulo = 'purple'
    elif y_pred == 1:
        categoria = f'POSITIVE (Class {class_names[1]})'
        cor_titulo = 'blue'
    else:
        categoria = f'NEGATIVE (Class {class_names[0]})'
        cor_titulo = 'red'
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Panel 1: Original image
    axes[0].imshow(img_original, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(
        f'Original Digit\n'
        f'True Class: {class_names[y_true]} (y={y_true})',
        fontsize=11, fontweight='bold'
    )
    axes[0].axis('off')
    
    # Panel 2: Overlay with explanation in RED
    img_rgb = np.stack([img_original, img_original, img_original], axis=-1)
    
    # Apply red overlay on explanation pixels
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if mask_img[i, j] > 0:
                alpha = 0.7
                img_rgb[i, j, 0] = min(1.0, img_rgb[i, j, 0] + alpha)  # Red
                img_rgb[i, j, 1] = img_rgb[i, j, 1] * 0.3  # Reduce green
                img_rgb[i, j, 2] = img_rgb[i, j, 2] * 0.3  # Reduce blue
    
    axes[1].imshow(img_rgb, interpolation='nearest')
    
    # Score position
    if decision_score >= t_plus:
        pos_score = f'> t+ ({t_plus:.3f})'
    elif decision_score <= t_minus:
        pos_score = f'< t- ({t_minus:.3f})'
    else:
        pos_score = f'entre t- ({t_minus:.3f}) e t+ ({t_plus:.3f})'
    
    # Calculate total available pixels
    total_pixels = img_shape[0] * img_shape[1]
    
    metodo_nome = 'MINABRO' if metodo.lower() == 'peab' else metodo.upper()
    axes[1].set_title(
        f'{metodo_nome} Explanation\n'
        f'Predicted: {class_names[y_pred] if y_pred in [0,1] else "REJECTED"} (y={y_pred})\n'
        f'Score: {decision_score:.3f} ({pos_score})\n'
        f'Pixels in explanation: {len(explanation)}/{total_pixels} ({len(explanation)/total_pixels*100:.1f}%)',
        fontsize=10, fontweight='bold'
    )
    axes[1].axis('off')
    
    # Legend
    fig.subplots_adjust(right=0.88)
    fig.text(0.91, 0.5, 
             'Legend:\n\n'
             'Red:\nExplanation\npixels\n({}/{})\n\n'
             'Gray:\nOriginal\ndigit'.format(len(explanation), total_pixels),
             fontsize=9,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Main title
    fig.suptitle(
        f'{categoria} Instance Example - Method: {metodo_nome}\n'
        f'MNIST 3 vs 8 | Thresholds: t- = {t_minus:.3f}, t+ = {t_plus:.3f} | Shape: {img_shape}',
        fontsize=12, fontweight='bold', color=cor_titulo, y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    
    # Save
    if SAVE_PLOTS:
        output_path = Path(OUTPUT_DIR) / metodo
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = output_path / f'mnist_idx{inst_idx}_{output_suffix}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {filename}")
    
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def processar_comparacao():
    """
    Processes and generates comparative images for PEAB and MinExp
    """
    print(f"\n{'='*80}")
    print(f"COMPARATIVE VISUALIZATION GENERATOR - MNIST")
    print(f"Comparing methods: {', '.join([m.upper() for m in METODOS])}")
    print(f"{'='*80}\n")
    
    # Load data from both methods
    dados_metodos = {}
    configs = {}
    
    for metodo in METODOS:
        try:
            print(f"[+] Loading data from {metodo.upper()}...")
            dados_metodos[metodo] = carregar_json(metodo)
            configs[metodo] = dados_metodos[metodo].get('config', {})
            print(f"  OK {len(dados_metodos[metodo].get('per_instance', []))} instances loaded")
        except Exception as e:
            print(f"  ERROR loading {metodo}: {e}")
            return
    
    # Reload dataset (use config from first method that has MNIST info)
    print("\nReloading original dataset...")
    config = None
    for metodo in METODOS:
        cfg = configs[metodo]
        # Check if it has MNIST configurations
        if 'mnist_feature_mode' in cfg and 'mnist_digit_pair' in cfg:
            config = cfg
            print(f"  Using MNIST configurations from: {metodo.upper()}")
            break
    
    # If no method has it, use the first and try to infer from model
    if config is None:
        config = configs[METODOS[0]]
        # Try to infer from model's number of features
        num_model_features = dados_metodos[METODOS[0]].get('model', {}).get('num_features', 784)
        if num_model_features == 196:
            config['mnist_feature_mode'] = 'pool2x2'
            config['mnist_digit_pair'] = [3, 8]
            print(f"  Inferred from num_features: pool2x2 (196 features)")
        else:
            config['mnist_feature_mode'] = 'raw'
            config['mnist_digit_pair'] = [3, 8]
            print(f"  Using default: raw (784 features)")
    
    X_test, _, class_names = recarregar_dataset(config)
    
    if X_test is None:
        print("❌ Error reloading dataset")
        return
    
    # Detect image size
    num_features = X_test.shape[1] if hasattr(X_test, 'shape') else len(list(X_test.keys()))
    img_shape = detectar_tamanho_imagem(num_features)
    
    print(f"  ✓ Dataset reloaded: {X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test[list(X_test.keys())[0]])} instances")
    print(f"  ✓ Number of features: {num_features}")
    print(f"  ✓ Detected image shape: {img_shape}")
    
    # Process each method
    for metodo in METODOS:
        print(f"\n{'='*80}")
        print(f"Processing method: {metodo.upper()}")
        print(f"{'='*80}")
        
        data = dados_metodos[metodo]
        per_instance = data.get('per_instance', [])
        
        t_plus = data.get('thresholds', {}).get('t_plus', 0.0)
        t_minus = data.get('thresholds', {}).get('t_minus', 0.0)
        
        print(f"✓ Thresholds: t- = {t_minus:.4f}, t+ = {t_plus:.4f}")
        
        # Collect candidates
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
        
        print(f"✓ Candidates: {len(candidatos_positiva)} positive, {len(candidatos_negativa)} negative, {len(candidatos_rejeitada)} rejected")
        
        # Select examples
        import random
        # DO NOT use fixed seed - we want different random selection each run
        # to explore different examples when IDX is None
        
        exemplos = []
        indices_selecionados = {}
        
        # Positive
        if IDX_POSITIVA is not None:
            encontrado = [c for c in candidatos_positiva if c[0] == IDX_POSITIVA]
            if encontrado:
                exemplos.append(('positiva', encontrado[0]))
                indices_selecionados['positiva'] = IDX_POSITIVA
                print(f"  ✓ Using MANUAL index for positive: {IDX_POSITIVA}")
        elif candidatos_positiva:
            escolhido = random.choice(candidatos_positiva)
            exemplos.append(('positiva', escolhido))
            indices_selecionados['positiva'] = escolhido[0]
            print(f"  🎲 RANDOM index selected for positive: {escolhido[0]}")
        
        # Negative
        if IDX_NEGATIVA is not None:
            encontrado = [c for c in candidatos_negativa if c[0] == IDX_NEGATIVA]
            if encontrado:
                exemplos.append(('negativa', encontrado[0]))
                indices_selecionados['negativa'] = IDX_NEGATIVA
                print(f"  ✓ Using MANUAL index for negative: {IDX_NEGATIVA}")
        elif candidatos_negativa:
            escolhido = random.choice(candidatos_negativa)
            exemplos.append(('negativa', escolhido))
            indices_selecionados['negativa'] = escolhido[0]
            print(f"  🎲 RANDOM index selected for negative: {escolhido[0]}")
        
        # Rejected
        if IDX_REJEITADA is not None:
            encontrado = [c for c in candidatos_rejeitada if c[0] == IDX_REJEITADA]
            if encontrado:
                exemplos.append(('rejeitada', encontrado[0]))
                indices_selecionados['rejeitada'] = IDX_REJEITADA
                print(f"  ✓ Using MANUAL index for rejected: {IDX_REJEITADA}")
        elif candidatos_rejeitada:
            escolhido = random.choice(candidatos_rejeitada)
            exemplos.append(('rejeitada', escolhido))
            indices_selecionados['rejeitada'] = escolhido[0]
            print(f"  🎲 RANDOM index selected for rejected: {escolhido[0]}")
        
        # Show code to fix these indices
        if any(v is None for v in [IDX_POSITIVA, IDX_NEGATIVA, IDX_REJEITADA]):
            print(f"\n  {'='*60}")
            print(f"  💡 To fix these indices, update the variables at the top of the script:")
            print(f"  {'='*60}")
            if IDX_POSITIVA is None and 'positiva' in indices_selecionados:
                print(f"  IDX_POSITIVA = {indices_selecionados['positiva']}")
            if IDX_NEGATIVA is None and 'negativa' in indices_selecionados:
                print(f"  IDX_NEGATIVA = {indices_selecionados['negativa']}")
            if IDX_REJEITADA is None and 'rejeitada' in indices_selecionados:
                print(f"  IDX_REJEITADA = {indices_selecionados['rejeitada']}")
            print(f"  {'='*60}\n")
        
        # Generate images
        print(f"\n[PLOT] Generating visualizations for {metodo.upper()}...")
        
        # Translation dictionary for instance types
        tipo_translation = {
            'positiva': 'POSITIVE',
            'negativa': 'NEGATIVE',
            'rejeitada': 'REJECTED'
        }
        
        for tipo, (idx, inst) in exemplos:
            tipo_en = tipo_translation.get(tipo, tipo.upper())
            print(f"\n  [>] Processing {tipo_en} instance (idx={idx})...")
            criar_imagem_comparativa(
                inst, idx, X_test, class_names, metodo,
                t_plus, t_minus, img_shape, tipo
            )
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED!")
    print(f"{'='*80}")
    print(f"\nImages saved in: {Path(OUTPUT_DIR).resolve()}")
    print(f"Subfolders created for each method: {', '.join(METODOS)}")
    print("\nTo compare:")
    print(f"  - PEAB: {OUTPUT_DIR}/peab/")
    print(f"  - MinExp: {OUTPUT_DIR}/minexp/")


if __name__ == "__main__":
    processar_comparacao()
