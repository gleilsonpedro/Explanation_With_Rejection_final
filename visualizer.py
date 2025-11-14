import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# ==============================================================================
# CONSTANTES
# ==============================================================================
RESULTS_FILE = 'json/comparative_results.json'
OUTPUT_DIR = 'analysis_output/plots'
SAVE_PLOTS = True
SHOW_PLOTS = False  # Evita travar execução em lote (use True para visualizar interativamente)
# Limite de instâncias MNIST a plotar (por experimento). None = todas
MAX_PER_INSTANCE = 5
PER_INSTANCE_PLOTS = False  # Desativado por padrão para execução rápida
USE_ARCHETYPES_DELTAS = True  # Gera figura 3x1 com arquétipos de deltas (médio 0, médio 1 e 1 rejeitada)

# ==============================================================================
# FUNÇÕES
# ==============================================================================

def carregar_json(filepath: str) -> dict:
    """Carrega o arquivo JSON de resultados"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def encontrar_experimentos_mnist(data: dict) -> list:
    """Encontra todos os experimentos MNIST no JSON"""
    if 'peab' not in data:
        return []
    
    mnist_exp = [k for k in data['peab'].keys() if 'mnist' in k.lower()]
    return sorted(mnist_exp)


def calcular_deltas_da_explicacao(per_instance: list, model_coefs: list, 
                                   X_test, num_features: int) -> tuple:
    """
    Calcula os deltas (contribuições) a partir das explicações e do modelo.
    
    X_test pode vir em dois formatos:
    1. Lista de listas: [[feat1, feat2, ...], [feat1, feat2, ...], ...]
    2. Dict de listas: {'pixel1': [inst1, inst2, ...], 'pixel2': [...], ...}
    
    Returns:
        (heatmap_pos, heatmap_neg, heatmap_rej, count_pos, count_neg, count_rej)
    """
    heatmap_pos = np.zeros(num_features, dtype=float)
    heatmap_neg = np.zeros(num_features, dtype=float)
    heatmap_rej = np.zeros(num_features, dtype=float)
    
    count_pos = 0
    count_neg = 0
    count_rej = 0
    
    # Converter coefs para numpy array
    coefs_arr = np.array(model_coefs)
    
    # Converter X_test para formato [instâncias x features]
    if isinstance(X_test, dict):
        # X_test é dict: {'pixel1': [inst1, inst2, ...], 'pixel2': [...], ...}
        # Precisamos transpor para [[inst1_feat1, inst1_feat2, ...], [inst2_feat1, ...], ...]
        
        # Ordenar chaves por número do pixel
        pixel_keys = sorted(X_test.keys(), key=lambda x: int(x.replace('pixel', '')))
        
        # Criar matriz transposta
        num_instances = len(X_test[pixel_keys[0]])
        X_test_arr = np.zeros((num_instances, num_features))
        
        for feat_idx, pixel_key in enumerate(pixel_keys):
            X_test_arr[:, feat_idx] = X_test[pixel_key]
    else:
        # X_test já é lista de listas
        X_test_arr = np.array(X_test)
    
    print(f"  📐 Dimensões: X_test={X_test_arr.shape}, coefs={coefs_arr.shape}")
    
    for idx, inst in enumerate(per_instance):
        # Usar índice sequencial em vez do ID original
        if idx < len(X_test_arr):
            x_vals = X_test_arr[idx]
        else:
            x_vals = np.zeros(num_features)
        
        # Calcular deltas: delta_i = x_i * w_i
        deltas = x_vals * coefs_arr
        
        # Agregar por classe
        if inst['rejected']:
            heatmap_rej += deltas
            count_rej += 1
        elif inst['y_pred'] == 0:
            heatmap_neg += deltas
            count_neg += 1
        elif inst['y_pred'] == 1:
            heatmap_pos += deltas
            count_pos += 1
    
    # Normalizar (calcular médias)
    if count_pos > 0:
        heatmap_pos /= count_pos
    if count_neg > 0:
        heatmap_neg /= count_neg
    if count_rej > 0:
        heatmap_rej /= count_rej
    
    return heatmap_pos, heatmap_neg, heatmap_rej, count_pos, count_neg, count_rej


def agregar_minimas_explicacoes(per_instance: list, feature_names: list, num_features: int) -> tuple:
    """
    Agrega máscaras de explicações mínimas por categoria (positiva/negativa/rejeição).

    Em vez de contribuições (coeficientes), aqui contamos com que frequência cada
    pixel/feature aparece na explicação mínima das instâncias daquela categoria.

    Retorna três heatmaps de frequência normalizada por categoria (valores em [0,1])
    e as contagens de instâncias por categoria.
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    freq_pos = np.zeros(num_features, dtype=float)
    freq_neg = np.zeros(num_features, dtype=float)
    freq_rej = np.zeros(num_features, dtype=float)

    c_pos = c_neg = c_rej = 0

    for inst in per_instance:
        exp_feats = inst.get('explanation', []) or []
        mask = np.zeros(num_features, dtype=float)

        for feat in exp_feats:
            idx = None
            # Preferência: mapear pelo nome exato (está em feature_names)
            if isinstance(feat, str) and feat in name_to_idx:
                idx = name_to_idx[feat]
            else:
                # fallback para formatos como 'pixel123'
                try:
                    if isinstance(feat, str) and feat.startswith('pixel'):
                        idx = int(feat.replace('pixel', '')) - 1
                    elif isinstance(feat, (int, np.integer)):
                        # assumir 1-based
                        idx = int(feat) - 1
                except Exception:
                    idx = None

            if idx is not None and 0 <= idx < num_features:
                mask[idx] = 1.0

        if inst.get('rejected', False):
            freq_rej += mask
            c_rej += 1
        elif inst.get('y_pred', None) == 0:
            freq_neg += mask
            c_neg += 1
        elif inst.get('y_pred', None) == 1:
            freq_pos += mask
            c_pos += 1

    # Normalizar por quantidade de instâncias em cada categoria (frequência relativa)
    if c_pos > 0:
        freq_pos /= c_pos
    if c_neg > 0:
        freq_neg /= c_neg
    if c_rej > 0:
        freq_rej /= c_rej

    return freq_pos, freq_neg, freq_rej, c_pos, c_neg, c_rej


def agregar_deltas_por_classe(per_instance: list, num_features: int = 784) -> tuple:
    """
    Agrega os deltas (contribuições) por categoria de predição.
    
    Returns:
        (heatmap_pos, heatmap_neg, heatmap_rej, count_pos, count_neg, count_rej)
    """
    heatmap_pos = np.zeros(num_features, dtype=float)
    heatmap_neg = np.zeros(num_features, dtype=float)
    heatmap_rej = np.zeros(num_features, dtype=float)
    
    count_pos = 0
    count_neg = 0
    count_rej = 0
    
    for inst in per_instance:
        deltas = np.array(inst['deltas'], dtype=float)
        
        if inst['rejected']:
            heatmap_rej += deltas
            count_rej += 1
        elif inst['y_pred'] == 0:
            heatmap_neg += deltas
            count_neg += 1
        elif inst['y_pred'] == 1:
            heatmap_pos += deltas
            count_pos += 1
    
    # Normalizar (calcular médias)
    if count_pos > 0:
        heatmap_pos /= count_pos
    if count_neg > 0:
        heatmap_neg /= count_neg
    if count_rej > 0:
        heatmap_rej /= count_rej
    
    return heatmap_pos, heatmap_neg, heatmap_rej, count_pos, count_neg, count_rej


def criar_visualizacao(heatmap_pos, heatmap_neg, heatmap_rej,
                       class_names: list, counts: tuple,
                       experiment_name: str, img_shape=(28, 28)):
    """
    Cria visualização com 3 imagens lado a lado mostrando as explicações.
    
    INTERPRETAÇÃO:
    - AZUL (+): Evidência PARA a classe positiva (ex: "É um 7")
    - VERMELHO (-): Evidência CONTRA a classe positiva (ex: "NÃO é um 7")
    - BRANCO (0): Neutro, não contribui
    """
    count_pos, count_neg, count_rej = counts
    
    # Criar figura 1x3
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    # Calcular escala global para cores uniformes
    all_values = []
    if count_neg > 0:
        all_values.extend(heatmap_neg.flatten())
    if count_pos > 0:
        all_values.extend(heatmap_pos.flatten())
    if count_rej > 0:
        all_values.extend(heatmap_rej.flatten())
    
    if len(all_values) > 0:
        vmax = np.max(np.abs(all_values))
    else:
        vmax = 1.0
    
    if vmax == 0:
        vmax = 1.0
    
    # Colormap: Vermelho (negativo) → Branco (zero) → Azul (positivo)
    cmap = 'RdBu_r'
    
    # ========================================
    # PLOT 1: CLASSE NEGATIVA (Classe 0)
    # ========================================
    if count_neg > 0:
        img_neg = heatmap_neg.reshape(img_shape)
    else:
        img_neg = np.zeros(img_shape)
    
    # Usar apenas valores negativos (favoráveis à classe 0) e colormap azul
    img_neg_only = np.where(img_neg < 0, -img_neg, 0)  # Inverte valores negativos para positivos
    vmax_neg = np.max(img_neg_only) if np.max(img_neg_only) > 0 else 1.0
    
    im0 = axes[0].imshow(img_neg_only, cmap='Blues', vmin=0, vmax=vmax_neg)
    axes[0].set_title(
        f'CLASSE NEGATIVA: {class_names[0]}\n'
        f'({count_neg} instâncias)\n\n'
        f'Azul = Evidência para {class_names[0]}',
        fontsize=11, fontweight='bold', pad=10
    )
    axes[0].axis('off')
    
    # ========================================
    # PLOT 2: CLASSE POSITIVA (Classe 1)
    # ========================================
    if count_pos > 0:
        img_pos = heatmap_pos.reshape(img_shape)
    else:
        img_pos = np.zeros(img_shape)
    
    # Usar apenas valores positivos (favoráveis à classe 1) e colormap vermelho
    img_pos_only = np.where(img_pos > 0, img_pos, 0)
    vmax_pos = np.max(img_pos_only) if np.max(img_pos_only) > 0 else 1.0
    
    im1 = axes[1].imshow(img_pos_only, cmap='Reds', vmin=0, vmax=vmax_pos)
    axes[1].set_title(
        f'CLASSE POSITIVA: {class_names[1]}\n'
        f'({count_pos} instâncias)\n\n'
        f'Vermelho = Evidência para {class_names[1]}',
        fontsize=11, fontweight='bold', pad=10
    )
    axes[1].axis('off')
    
    # ========================================
    # PLOT 3: REJEITADAS
    # ========================================
    if count_rej > 0:
        img_rej = heatmap_rej.reshape(img_shape)
    else:
        img_rej = np.zeros(img_shape)
    
    im2 = axes[2].imshow(img_rej, cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[2].set_title(
        f'REJEITADAS\n'
        f'({count_rej} instâncias)\n\n'
        f'Cores mistas = Evidências conflitantes',
        fontsize=11, fontweight='bold', pad=10
    )
    axes[2].axis('off')
    
    # ========================================
    # COLORBAR (Legenda)
    # ========================================
    fig.subplots_adjust(right=0.87, wspace=0.3)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label(
        'Explicação:\n\n'
        f'VERMELHO (+):\nEvidência para\n{class_names[1]}\n\n'
        f'AZUL (-):\nEvidência para\n{class_names[0]}\n\n'
        'BRANCO (0):\nNeutro',
        fontsize=10,
        rotation=0,
        labelpad=15,
        ha='left'
    )
    
    # Título principal
    fig.suptitle(
        f'Explicações Abdutivas do PEAB - {experiment_name}\n'
        f'Mapa de Calor das Explicações Médias por Classe\n',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 0.87, 0.95])
    
    # Salvar se configurado
    if SAVE_PLOTS:
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / f'explicacao_peab_{experiment_name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  💾 Imagem salva: {filename}")
    
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def criar_visualizacao_minimas(freq_pos, freq_neg, freq_rej,
                               class_names: list, counts: tuple,
                               experiment_name: str, img_shape=(28, 28)):
    """
    Cria visualização 3x1 das FREQUÊNCIAS com que cada pixel aparece na
    explicação mínima por categoria (0..1). Ideal para MNIST.
    """
    c_pos, c_neg, c_rej = counts

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    vmax = 1.0  # frequência relativa
    cmap = 'magma'  # contrastante para densidade

    img_neg = freq_neg.reshape(img_shape) if c_neg > 0 else np.zeros(img_shape)
    img_pos = freq_pos.reshape(img_shape) if c_pos > 0 else np.zeros(img_shape)
    img_rej = freq_rej.reshape(img_shape) if c_rej > 0 else np.zeros(img_shape)

    im0 = axes[0].imshow(img_neg, cmap=cmap, vmin=0.0, vmax=vmax)
    axes[0].set_title(
        f"CLASSE NEGATIVA: {class_names[0]}\n"
        f"({c_neg} instâncias)\n\n"
        f"Intensidade = frequência em explicações mínimas",
        fontsize=11, fontweight='bold', pad=10
    )
    axes[0].axis('off')

    im1 = axes[1].imshow(img_pos, cmap=cmap, vmin=0.0, vmax=vmax)
    axes[1].set_title(
        f"CLASSE POSITIVA: {class_names[1]}\n"
        f"({c_pos} instâncias)\n\n"
        f"Intensidade = frequência em explicações mínimas",
        fontsize=11, fontweight='bold', pad=10
    )
    axes[1].axis('off')

    im2 = axes[2].imshow(img_rej, cmap=cmap, vmin=0.0, vmax=vmax)
    axes[2].set_title(
        f"REJEITADAS\n"
        f"({c_rej} instâncias)\n\n"
        f"Intensidade = frequência em explicações mínimas",
        fontsize=11, fontweight='bold', pad=10
    )
    axes[2].axis('off')

    fig.subplots_adjust(right=0.87, wspace=0.3)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label(
        'Frequência relativa (0..1)\n\n'
        'Mais claro = pixel mais recorrente\n'
        'nas explicações mínimas',
        fontsize=10,
        rotation=0,
        labelpad=15,
        ha='left'
    )

    fig.suptitle(
        f'Explicações Mínimas (Frequência por pixel) - {experiment_name}',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 0.87, 0.95])

    if SAVE_PLOTS:
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / f'explicacao_minimas_{experiment_name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  💾 Imagem (frequências) salva: {filename}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _get_instance_deltas(inst: dict, X_test, coefs: np.ndarray, num_features: int) -> np.ndarray:
    """Obtém o vetor de deltas (784) de uma instância a partir do JSON.

    Se 'deltas' não estiver presente em per_instance, calcula como x * w.
    DEPRECATED: Use cálculo direto com índice sequencial em vez do ID original.
    """
    deltas = inst.get('deltas', None)
    if deltas is not None and isinstance(deltas, (list, tuple)) and len(deltas) == num_features:
        return np.array(deltas, dtype=float)
    # fallback: calcular
    inst_id = int(inst.get('id', 0))
    x_vals = _get_instance_vector(X_test, inst_id, num_features)
    result = x_vals * coefs
    return result


def criar_visualizacao_arquetipos_deltas(mean_neg: np.ndarray,
                                         mean_pos: np.ndarray,
                                         deltas_rej: np.ndarray,
                                         class_names: list,
                                         experiment_name: str,
                                         img_shape=(28, 28),
                                         cmap: str = 'seismic') -> None:
    """Cria figura 1x3 com escalas iguais e uma única colorbar.

    - Plot 1: média dos deltas das negativas (y_pred==0, sem rejeição)
    - Plot 2: média dos deltas das positivas (y_pred==1, sem rejeição)
    - Plot 3: deltas de UMA instância rejeitada (sem média)
    """
    # Preparar dados e escala comum
    a0 = mean_neg.reshape(img_shape)
    a1 = mean_pos.reshape(img_shape)
    a2 = deltas_rej.reshape(img_shape)

    vmax = np.max(np.abs([mean_neg, mean_pos, deltas_rej]))
    if vmax == 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    im0 = axes[0].imshow(a0, cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"Evidência Média (Classe {class_names[0]})", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    im1 = axes[1].imshow(a1, cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"Evidência Média (Classe {class_names[1]})", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    im2 = axes[2].imshow(a2, cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[2].set_title("Instância Rejeitada (Conflito)", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Colorbar única
    fig.subplots_adjust(right=0.88, wspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label(
        f"Intensidade da evidência (δ)\n\n"
        f"Vermelho (+): favorece {class_names[1]}\n"
        f"Azul (-): favorece {class_names[0]}\n"
        f"Branco (0): neutro",
        fontsize=10,
        rotation=0,
        labelpad=15,
        ha='left'
    )

    fig.suptitle(f"Mapa de Calor de Evidência (Deltas) — {experiment_name}", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    if SAVE_PLOTS:
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / f'arquetipos_deltas_{experiment_name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  💾 Imagem (arquétipos de deltas) salva: {filename}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _get_instance_vector(X_test, inst_id, num_features):
    """Retorna vetor de features da instância inst_id a partir de X_test.

    X_test pode ser dict de pixels {'pixel1': [...]} ou lista/array.
    """
    if isinstance(X_test, dict):
        # ordenar chaves por número do pixel (pixel1 -> 1)
        pixel_keys = sorted(X_test.keys(), key=lambda x: int(x.replace('pixel', '')))
        num_instances = len(X_test[pixel_keys[0]])
        if inst_id >= num_instances:
            return np.zeros(num_features)

        x_vals = np.zeros(num_features)
        for feat_idx, pixel_key in enumerate(pixel_keys):
            x_vals[feat_idx] = X_test[pixel_key][inst_id]
        return x_vals
    else:
        X_arr = np.array(X_test)
        if inst_id < X_arr.shape[0]:
            return X_arr[inst_id]
        return np.zeros(num_features)


def criar_visualizacao_por_instancia(inst: dict, X_test, num_features: int,
                                     class_names: list, experiment_name: str,
                                     img_shape=(28, 28)):
    """Gera visualização por instância: imagem original + overlay da
    explicação mínima (heatmap) colorido conforme a categoria da predição.

    - Positiva (y_pred==1): overlay em azul (Blues)
    - Negativa (y_pred==0): overlay em vermelho (Reds)
    - Rejeitada (rejected=True): overlay em púrpura (Purples)
    """
    inst_id = int(inst.get('id', 0))
    explanation = inst.get('explanation', [])
    rejected = bool(inst.get('rejected', False))
    y_pred = inst.get('y_pred', None)

    # Reconstruir imagem
    x_vals = _get_instance_vector(X_test, inst_id, num_features)
    img = x_vals.reshape(img_shape)

    # Criar máscara de explicação (valores originais dos pixels quando selecionados)
    mask = np.zeros(num_features, dtype=float)

    # Se feature names forem conhecidos no X_test (dict), tentar resolver índices
    # Aceitar nomes como 'pixel123' ou nomes presentes em feature_names
    for feat in explanation:
        try:
            if isinstance(feat, str) and feat.startswith('pixel'):
                idx = int(feat.replace('pixel', '')) - 1
            else:
                idx = int(feat) - 1 if isinstance(feat, (str, int)) else None
        except Exception:
            idx = None

        if idx is None:
            # tentar encontrar pelo nome nas chaves de X_test (se dict)
            if isinstance(X_test, dict) and feat in X_test:
                # extrair número do pixel
                try:
                    idx = int(feat.replace('pixel', '')) - 1
                except Exception:
                    idx = None

        if idx is not None and 0 <= idx < num_features:
            mask[idx] = x_vals[idx]

    mask_img = mask.reshape(img_shape)

    # Escolher layout e colormap por categoria
    if rejected:
        cols = 2
        cm = 'Purples'
        title_overlay = 'Rejeitada'
    elif y_pred == 0:
        cols = 2
        cm = 'Reds'
        title_overlay = f'Negativa: {class_names[0]}'
    else:
        cols = 2
        cm = 'Blues'
        title_overlay = f'Positiva: {class_names[1]}'

    fig, axes = plt.subplots(1, cols, figsize=(10, 4))
    if cols == 1:
        axes = [axes]

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Original (id={inst_id})\nClasse verdadeira: {inst.get("y_true", "-")}', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(img, cmap='gray')
    axes[1].imshow(mask_img, cmap=cm, alpha=0.65)
    axes[1].set_title(f'{title_overlay}  (y_pred={y_pred})', fontsize=10)
    axes[1].axis('off')

    fig.suptitle(f'Instância {inst_id} - Explicação mínima ({experiment_name})', fontsize=12)

    plt.tight_layout()

    if SAVE_PLOTS:
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        safe_exp = experiment_name.replace('/', '_')
        filename = output_path / f'explicacao_peab_{safe_exp}_inst_{inst_id}.png'
        plt.savefig(filename, dpi=140, bbox_inches='tight')
        print(f"  💾 Imagem por instância salva: {filename}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def processar_experimento(data: dict, exp_key: str):
    """Processa um experimento e gera a visualização"""
    print(f"\n{'='*80}")
    print(f"Processando: {exp_key}")
    print(f"{'='*80}")
    
    try:
        exp_data = data['peab'][exp_key]
        
        # Verificar se tem estrutura necessária
        if 'per_instance' not in exp_data:
            print("\n❌ ERRO: Chave 'per_instance' não encontrada!")
            return
        
        if 'data' not in exp_data or 'model' not in exp_data:
            print("\n❌ ERRO: Estrutura incompleta no JSON!")
            return
        
        # Obter informações
        class_names = exp_data['data']['class_names']
        per_instance = exp_data['per_instance']
        X_test = exp_data['data']['X_test']
        model_coefs = exp_data['model']['coefs']
        feature_names = exp_data['data'].get('feature_names', [])
        
        if len(per_instance) == 0:
            print("⚠ Nenhuma instância encontrada. Pulando...")
            return
        
        # Determinar número de features
        num_features = len(model_coefs)
        
        print(f"\n✓ Classes: {class_names[0]} vs {class_names[1]}")
        print(f"✓ Total de instâncias: {len(per_instance)}")
        print(f"✓ Features: {num_features}")
        
        # Determinar shape da imagem
        if num_features == 784:
            img_shape = (28, 28)
            print("✓ Formato: 28x28 (MNIST raw)")
        elif num_features == 196:
            img_shape = (14, 14)
            print("✓ Formato: 14x14 (MNIST pooling)")
        else:
            lado = int(np.sqrt(num_features))
            if lado * lado == num_features:
                img_shape = (lado, lado)
            else:
                img_shape = (28, 28)  # fallback
            print(f"✓ Formato: {img_shape}")
        
        # Para MNIST, criar mapas com base nas EXPLICAÇÕES MÍNIMAS (frequências)
        is_mnist = (exp_key.lower().startswith('mnist') or exp_data['config'].get('dataset_name', '').lower() == 'mnist')

        # Se solicitado, gera painel de arquétipos baseado em deltas
        if is_mnist and USE_ARCHETYPES_DELTAS:
            print("\n🧬 Construindo arquétipos de evidência (deltas)...")
            coefs_arr = np.array(model_coefs, dtype=float)
            neg_deltas_sum = np.zeros(num_features, dtype=float)
            pos_deltas_sum = np.zeros(num_features, dtype=float)
            count_neg = count_pos = 0
            rejected_sample = None

            for idx, inst in enumerate(per_instance):
                y_pred = inst.get('y_pred', None)
                rejected_flag = inst.get('rejected', False)
                # Usar índice sequencial (idx) em vez do ID original
                x_vals = _get_instance_vector(X_test, idx, num_features)
                deltas_vec = x_vals * coefs_arr

                if rejected_flag and rejected_sample is None:
                    rejected_sample = deltas_vec
                if not rejected_flag:
                    if y_pred == 0:
                        neg_deltas_sum += deltas_vec
                        count_neg += 1
                    elif y_pred == 1:
                        pos_deltas_sum += deltas_vec
                        count_pos += 1

            if count_neg > 0:
                mean_neg = neg_deltas_sum / count_neg
            else:
                mean_neg = np.zeros(num_features)
            if count_pos > 0:
                mean_pos = pos_deltas_sum / count_pos
            else:
                mean_pos = np.zeros(num_features)
            if rejected_sample is None:
                rejected_sample = np.zeros(num_features)

            print(f"  • Arquétipo negativo média de {count_neg} instâncias")
            print(f"  • Arquétipo positivo média de {count_pos} instâncias")
            print("  • Instância rejeitada usada para conflito")

            criar_visualizacao_arquetipos_deltas(mean_neg, mean_pos, rejected_sample,
                                                 class_names, exp_key, img_shape, cmap='seismic')

        if is_mnist and feature_names and not USE_ARCHETYPES_DELTAS:
            print("\n📊 Agregando explicações MÍNIMAS por classe (frequência de pixels)...")
            freq_pos, freq_neg, freq_rej, c_pos, c_neg, c_rej = agregar_minimas_explicacoes(
                per_instance, feature_names, num_features
            )
            print(f"  • Negativas ({class_names[0]}): {c_neg} instâncias")
            print(f"  • Positivas ({class_names[1]}): {c_pos} instâncias")
            print(f"  • Rejeitadas: {c_rej} instâncias")

            print("\n🎨 Gerando visualização (frequências de pixels em explicações mínimas)...")
            criar_visualizacao_minimas(
                freq_pos, freq_neg, freq_rej,
                class_names,
                (c_pos, c_neg, c_rej),
                exp_key,
                img_shape
            )
        else:
            # Caso geral (não-MNIST): usar deltas/contribuições
            print("\n📊 Calculando e agregando evidências por classe (contribuições)...")
            heatmap_pos, heatmap_neg, heatmap_rej, count_pos, count_neg, count_rej = \
                calcular_deltas_da_explicacao(per_instance, model_coefs, X_test, num_features)

            print(f"  • Negativas ({class_names[0]}): {count_neg} instâncias")
            print(f"  • Positivas ({class_names[1]}): {count_pos} instâncias")
            print(f"  • Rejeitadas: {count_rej} instâncias")

            print("\n🎨 Gerando visualização (contribuições)...")
            criar_visualizacao(
                heatmap_pos, heatmap_neg, heatmap_rej,
                class_names,
                (count_pos, count_neg, count_rej),
                exp_key,
                img_shape
            )

        # Visualizações por instância (MNIST): imagem do dígito + heatmap dos pixels mínimos
        if is_mnist and PER_INSTANCE_PLOTS:
            print("\n🖼️  Gerando visualizações por instância (limitado)...")
            n_total = len(per_instance)
            n_plot = n_total if (MAX_PER_INSTANCE is None) else min(MAX_PER_INSTANCE, n_total)
            for i, inst in enumerate(per_instance[:n_plot]):
                try:
                    criar_visualizacao_por_instancia(inst, X_test, num_features, class_names, exp_key, img_shape)
                except Exception as e:
                    print(f"    ⚠️  Falha ao plotar instância {i} (id={inst.get('id')}): {e}")
        
        print(f"\n✅ Visualização concluída para {exp_key}!")
        
    except KeyError as e:
        print(f"\n❌ ERRO: Chave não encontrada no JSON: {e}")
        print("\n💡 Dica: Verifique se o experimento foi executado corretamente")
        print("         e se o JSON foi gerado com todas as informações necessárias.")
    except Exception as e:
        print(f"\n❌ ERRO ao processar {exp_key}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("="*80)
    print("VISUALIZADOR DE EXPLICAÇÕES ABDUTIVAS DO PEAB")
    print("="*80)
    parser = argparse.ArgumentParser(description='Visualizador de evidências (deltas) e explicações mínimas para MNIST.')
    parser.add_argument('--results', type=str, default=RESULTS_FILE, help='Caminho para o JSON de resultados')
    parser.add_argument('--experiment', type=str, default=None, help='Chave do experimento (ex: mnist ou mnist_3_vs_8)')
    parser.add_argument('--show', action='store_true', help='Mostrar janelas interativas do Matplotlib')
    args = parser.parse_args()

    global SHOW_PLOTS
    if args.show:
        SHOW_PLOTS = True

    print(f"\n📂 Arquivo: {args.results}")
    
    try:
        # Carregar JSON
        data = carregar_json(args.results)
        print("✓ JSON carregado com sucesso")
        
        # Encontrar experimentos MNIST
        if args.experiment:
            mnist_exp = [args.experiment]
        else:
            mnist_exp = encontrar_experimentos_mnist(data)
        
        if len(mnist_exp) == 0:
            print("\n❌ Nenhum experimento MNIST encontrado no JSON!")
            print(f"Experimentos disponíveis: {list(data.get('peab', {}).keys())}")
            return
        
        print(f"\n✓ Encontrado(s) {len(mnist_exp)} experimento(s) MNIST:")
        for exp in mnist_exp:
            print(f"  • {exp}")
        
        # Processar cada experimento automaticamente
        for exp_key in mnist_exp:
            if exp_key not in data.get('peab', {}):
                print(f"⚠ Experimento '{exp_key}' não encontrado no JSON. Pulando.")
                continue
            processar_experimento(data, exp_key)
        
        print(f"\n{'='*80}")
        print(f"✅ TODOS OS EXPERIMENTOS PROCESSADOS COM SUCESSO!")
        print(f"{'='*80}")
        
        if SAVE_PLOTS:
            print(f"\n💾 Imagens salvas em: {OUTPUT_DIR}/")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERRO: {e}")
    except KeyError as e:
        print(f"\n❌ ERRO: Chave não encontrada no JSON: {e}")
    except Exception as e:
        print(f"\n❌ ERRO INESPERADO: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
