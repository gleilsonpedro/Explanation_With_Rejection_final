# falta configurar novo json pean_results e nova estrutura do json para obter os resultados


"""
Visualizador de Instâncias Individuais do PEAB - Pares MNIST

Este script gera 3 imagens individuais mostrando exemplos de explicações para
diferentes pares de classes MNIST (ex: 9 vs 4, 5 vs 6, etc).

Gera os seguintes exemplos:
1. Uma instância POSITIVA (classe positiva) corretamente classificada
2. Uma instância NEGATIVA (classe negativa) corretamente classificada  
3. Uma instância REJEITADA (com evidências conflitantes)

Cada imagem mostra:
- Dígito original (28x28 em escala de cinza)
- Overlay colorido dos pixels que compõem a explicação mínima do PEAB
- Informações sobre a classe verdadeira, predita e score de decisão

O script permite escolher interativamente qual par MNIST deseja visualizar.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
RESULTS_FILE = 'json/peab/mnist_3_vs_8.json'
OUTPUT_DIR = 'results/plots/mnist/numbers'
SAVE_PLOTS = True
SHOW_PLOTS = False

# ==============================================================================
# 🎯 CONTROLE DE ÍNDICES - EDITE AQUI PARA FIXAR EXEMPLOS ESPECÍFICOS
# ==============================================================================
# Deixe como None para seleção aleatória
# Ou defina o número do índice para fixar um exemplo específico
#
# Como descobrir os índices:
# 1. Rode o script com valores None (aleatório)
# 2. Olhe os índices (idx) que aparecem no console
# 3. Anote os que você gostou e coloque aqui embaixo
# 4. Execute novamente - sempre usará os mesmos exemplos!

IDX_POSITIVA = 165    # Ex: 104 para fixar um dígito 8 específico
IDX_NEGATIVA = 552    # Ex: 14 para fixar um dígito 3 específico  
IDX_REJEITADA = 526   # Ex: 13 para fixar uma instância rejeitada específica

# Exemplos de uso:
# IDX_POSITIVA = 104   # ← Descomente e use o índice que você gostou
# IDX_NEGATIVA = 14
# IDX_REJEITADA = 13

# Variável global para índices específicos (será preenchida no main)
INDICES_ESPECIFICOS = None

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


def _get_instance_vector(X_test, inst_idx: int, num_features: int) -> np.ndarray:
    """
    Retorna vetor de features da instância usando índice sequencial.
    
    IMPORTANTE: Usa inst_idx (posição no array) e NÃO inst['id'] porque:
    - inst['id'] = ID original do MNIST completo (ex: 45336, 67200)
    - X_test contém apenas subset de teste (ex: 126 instâncias)
    - A ordem em per_instance corresponde exatamente à ordem em X_test
    - Logo: enumerate(per_instance) dá o índice correto para X_test
    
    X_test pode ser:
    - Dict: {'pixel1': [val1, val2, ...], 'pixel2': [...], ...}
    - Array/lista: [[feat1, feat2, ...], [feat1, feat2, ...], ...]
    """
    if isinstance(X_test, dict):
        # Ordenar chaves por número do pixel
        pixel_keys = sorted(X_test.keys(), key=lambda x: int(x.replace('pixel', '')))
        num_instances = len(X_test[pixel_keys[0]])
        
        if inst_idx >= num_instances:
            return np.zeros(num_features)
        
        x_vals = np.zeros(num_features)
        for feat_idx, pixel_key in enumerate(pixel_keys):
            x_vals[feat_idx] = X_test[pixel_key][inst_idx]
        return x_vals
    else:
        # X_test é array/lista
        X_arr = np.array(X_test)
        if inst_idx < X_arr.shape[0]:
            return X_arr[inst_idx]
        return np.zeros(num_features)


def criar_imagem_individual(inst: dict, inst_idx: int, X_test, 
                            class_names: list, experiment_name: str,
                            t_plus: float, t_minus: float,
                            img_shape=(28, 28), output_suffix: str = ""):
    """
    Cria uma imagem mostrando:
    1. Dígito original
    2. Overlay com os pixels da explicação mínima destacados
    
    Args:
        inst: Dicionário com dados da instância (do per_instance)
        inst_idx: Índice sequencial da instância no X_test
        X_test: Dados de teste
        class_names: Nomes das classes
        experiment_name: Nome do experimento
        t_plus: Threshold superior (aceitar como positiva)
        t_minus: Threshold inferior (aceitar como negativa)
        img_shape: Formato da imagem (28, 28)
        output_suffix: Sufixo para o nome do arquivo (ex: "positiva", "negativa", "rejeitada")
    """
    
    num_features = img_shape[0] * img_shape[1]
    
    # Obter vetor de features usando índice sequencial
    x_vals = _get_instance_vector(X_test, inst_idx, num_features)
    
    # IMPORTANTE: Normalizar para [0, 1] se ainda não estiver
    # Muitos datasets MNIST vêm em [0, 255] ou outros ranges
    if x_vals.max() > 1.0:
        x_vals = x_vals / 255.0
    
    img_original = x_vals.reshape(img_shape)
    
    # Criar máscara binária da explicação (1 = faz parte da explicação, 0 = não faz parte)
    explanation = inst.get('explanation', [])
    mask_binary = np.zeros(num_features, dtype=float)
    
    # Mapear features da explicação para índices
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
            mask_binary[idx] = 1.0  # Marcar pixel como parte da explicação

    mask_img = mask_binary.reshape(img_shape)
    
    # Determinar categoria e cor
    rejected = inst.get('rejected', False)
    y_pred = inst.get('y_pred', -1)
    y_true = inst.get('y_true', -1)
    decision_score = inst.get('decision_score', 0.0)
    
    # Usar SEMPRE vermelho para destacar pixels da explicação (como no experimento original)
    # A cor do título muda conforme a categoria
    cmap_overlay = 'Reds'  # Sempre vermelho para overlay
    
    if rejected:
        categoria = 'REJEITADA'
        cor_titulo = 'purple'
    elif y_pred == 1:  # POSITIVA - normalmente classe 8 no MNIST 3vs8
        categoria = f'POSITIVA (Classe {class_names[1]})'
        cor_titulo = 'blue'
    else:  # y_pred == 0 - NEGATIVA - normalmente classe 3 no MNIST 3vs8
        categoria = f'NEGATIVA (Classe {class_names[0]})'
        cor_titulo = 'red'
    
    # Criar figura com 2 painéis lado a lado + espaço para colorbar
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Painel 1: Imagem original
    axes[0].imshow(img_original, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(
        f'Dígito Original\n'
        f'Classe Verdadeira: {class_names[y_true]} (y={y_true})',
        fontsize=11, fontweight='bold'
    )
    axes[0].axis('off')
    
    # Painel 2: Overlay com explicação
    # Estratégia: criar uma imagem RGB que mostre vermelho BRILHANTE mesmo sobre fundo preto
    # Isso resolve o problema de overlay vermelho invisível sobre pixels pretos
    
    # Converter imagem cinza para RGB (3 canais)
    img_rgb = np.stack([img_original, img_original, img_original], axis=-1)
    
    # Para pixels da explicação, aplicar overlay vermelho VISÍVEL
    # IMPORTANTE: O vermelho precisa ser adicionado, não multiplicado, para ser visível em preto
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if mask_img[i, j] > 0:  # Se pixel faz parte da explicação
                # Alpha blending com vermelho PURO: mesmo pixel preto (0,0,0) fica vermelho
                alpha = 0.7  # Intensidade do overlay
                
                # Canal R (vermelho): ADICIONAR vermelho independente do valor original
                img_rgb[i, j, 0] = min(1.0, img_rgb[i, j, 0] + alpha)
                
                # Canais G e B: manter ou escurecer para realçar o vermelho
                img_rgb[i, j, 1] = img_rgb[i, j, 1] * 0.3  # Reduzir verde
                img_rgb[i, j, 2] = img_rgb[i, j, 2] * 0.3  # Reduzir azul
    
    # Mostrar imagem RGB resultante
    axes[1].imshow(img_rgb, interpolation='nearest')
    
    # Determinar posição do score em relação aos thresholds
    if decision_score >= t_plus:
        pos_score = f'> t+ ({t_plus:.3f})'
    elif decision_score <= t_minus:
        pos_score = f'< t- ({t_minus:.3f})'
    else:
        pos_score = f'entre t- ({t_minus:.3f}) e t+ ({t_plus:.3f})'
    
    axes[1].set_title(
        f'Explicação PEAB\n'
        f'Predito: {class_names[y_pred] if y_pred in [0,1] else "REJEITADA"} (y={y_pred})\n'
        f'Score: {decision_score:.3f} ({pos_score})\n'
        f'Pixels na explicação: {len(explanation)}',
        fontsize=10, fontweight='bold'
    )
    axes[1].axis('off')
    
    # Adicionar legenda explicativa (sem colorbar, pois é binário)
    fig.subplots_adjust(right=0.88)
    fig.text(0.91, 0.5, 
             'Legenda:\n\n'
             'Vermelho:\nPixels da\nexplicação\nmínima\n(total: {})\n\n'
             'Cinza:\nDígito\noriginal\n(escala\nde cinza)'.format(len(explanation)),
             fontsize=9,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Título principal
    fig.suptitle(
        f'Exemplo de Instância {categoria}\n'
        f'{experiment_name} | Thresholds: t- = {t_minus:.3f}, t+ = {t_plus:.3f}',
        fontsize=13, fontweight='bold', color=cor_titulo, y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    # ... no final da função criar_imagem_individual, antes do plt.close() etc...

  
    # Salvar
    if SAVE_PLOTS:
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        
        safe_exp = experiment_name.replace('/', '_').replace(' ', '_')
        filename = output_path / f'{safe_exp}_{output_suffix}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  💾 Salvo: {filename}")
    
    
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def processar_experimento(data: dict, exp_key: str):
    """
    Processa um experimento e gera as 3 imagens individuais:
    - 1 exemplo de instância positiva (y_pred=1, não rejeitada)
    - 1 exemplo de instância negativa (y_pred=0, não rejeitada)
    - 1 exemplo de instância rejeitada
    """
    print(f"\n{'='*80}")
    print(f"Processando: {exp_key}")
    print(f"{'='*80}")
    
    try:
        # Novo formato JSON: dados diretos na raiz (sem wrapper 'peab')
        exp_data = data
        
        # Validar estrutura do experimento
        if 'per_instance' not in exp_data:
            print("❌ ERRO: 'per_instance' não encontrado!")
            return
        
        # Verificar se experimento usa rejeição (opcional - apenas informativo)
        config = exp_data.get('config', {})
        has_rejection_cost = 'rejection_cost' in config
        
        if not has_rejection_cost:
            print("⚠️  AVISO: Experimento sem custo de rejeição configurado")
        
        # Verificar se há instâncias rejeitadas (sample das primeiras 20)
        per_instance = exp_data.get('per_instance', [])
        if not per_instance:
            print("❌ ERRO: 'per_instance' não encontrado ou vazio!")
            return
            
        sample_size = min(20, len(per_instance))
        has_rejected_instances = any(inst.get('rejected', False) for inst in per_instance[:sample_size])
        
        if not has_rejected_instances and len(per_instance) > 0:
            print("ℹ️  INFO: Nenhuma instância rejeitada encontrada (nas primeiras 20)")
        
        # NOVO FORMATO: Recarregar dados do dataset original
        # Pois X_test e class_names não estão mais salvos no JSON
        print("📥 Recarregando dataset original para obter X_test...")
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
            print(f"❌ ERRO: Falha ao carregar dataset '{dataset_name}'")
            return
        
        # Fazer split train/test (mesmo split usado no experimento original)
        _, X_test, _, _ = train_test_split(
            X_full, y_full, test_size=test_size, 
            random_state=random_state, stratify=y_full
        )
        
        print(f"✓ Dataset recarregado: {X_test.shape[0]} instâncias de teste")
        
        # Obter thresholds
        t_plus = exp_data.get('thresholds', {}).get('t_plus', 0.0)
        t_minus = exp_data.get('thresholds', {}).get('t_minus', 0.0)
        
        if len(per_instance) == 0:
            print("⚠ Nenhuma instância encontrada.")
            return
        
        print(f"✓ Classes: {class_names[0]} vs {class_names[1]}")
        print(f"✓ Total de instâncias: {len(per_instance)}")
        print(f"✓ Thresholds: t- = {t_minus:.4f}, t+ = {t_plus:.4f}")
        
        # Coletar TODOS os exemplos de cada categoria
        candidatos_positiva = []
        candidatos_negativa = []
        candidatos_rejeitada = []
        
        for idx, inst in enumerate(per_instance):
            rejected = inst.get('rejected', False)
            y_pred = inst.get('y_pred', -1)
            explanation = inst.get('explanation', [])
            
            # Garantir que a instância tem explicação
            if len(explanation) == 0:
                continue
            
            if rejected:
                candidatos_rejeitada.append((idx, inst))
            elif y_pred == 1:
                candidatos_positiva.append((idx, inst))
            elif y_pred == 0:
                candidatos_negativa.append((idx, inst))
        
        # Selecionar exemplos (manualmente se especificado, ou aleatoriamente)
        exemplo_positiva = None
        exemplo_negativa = None
        exemplo_rejeitada = None
        
        idx_positiva = None
        idx_negativa = None
        idx_rejeitada = None
        
        # Verificar se há índices específicos solicitados
        indices_manual = INDICES_ESPECIFICOS if INDICES_ESPECIFICOS else {}
        
        # POSITIVA
        if indices_manual.get('positiva') is not None:
            idx_manual = indices_manual['positiva']
            # Buscar nos candidatos pelo índice
            encontrado = [cand for cand in candidatos_positiva if cand[0] == idx_manual]
            if encontrado:
                idx_positiva, exemplo_positiva = encontrado[0]
                print(f"  ✓ Usando índice manual para positiva: {idx_manual}")
            else:
                print(f"  ⚠️  Índice {idx_manual} não encontrado em positivas, escolhendo aleatório")
                if candidatos_positiva:
                    idx_positiva, exemplo_positiva = random.choice(candidatos_positiva)
        elif candidatos_positiva:
            idx_positiva, exemplo_positiva = random.choice(candidatos_positiva)
        
        # NEGATIVA
        if indices_manual.get('negativa') is not None:
            idx_manual = indices_manual['negativa']
            encontrado = [cand for cand in candidatos_negativa if cand[0] == idx_manual]
            if encontrado:
                idx_negativa, exemplo_negativa = encontrado[0]
                print(f"  ✓ Usando índice manual para negativa: {idx_manual}")
            else:
                print(f"  ⚠️  Índice {idx_manual} não encontrado em negativas, escolhendo aleatório")
                if candidatos_negativa:
                    idx_negativa, exemplo_negativa = random.choice(candidatos_negativa)
        elif candidatos_negativa:
            idx_negativa, exemplo_negativa = random.choice(candidatos_negativa)
        
        # REJEITADA
        if indices_manual.get('rejeitada') is not None:
            idx_manual = indices_manual['rejeitada']
            encontrado = [cand for cand in candidatos_rejeitada if cand[0] == idx_manual]
            if encontrado:
                idx_rejeitada, exemplo_rejeitada = encontrado[0]
                print(f"  ✓ Usando índice manual para rejeitada: {idx_manual}")
            else:
                print(f"  ⚠️  Índice {idx_manual} não encontrado em rejeitadas, escolhendo aleatório")
                if candidatos_rejeitada:
                    idx_rejeitada, exemplo_rejeitada = random.choice(candidatos_rejeitada)
        elif candidatos_rejeitada:
            idx_rejeitada, exemplo_rejeitada = random.choice(candidatos_rejeitada)
        
        # Mostrar estatísticas
        print(f"\n📊 Candidatos disponíveis:")
        print(f"  • Positivas: {len(candidatos_positiva)} instâncias")
        print(f"  • Negativas: {len(candidatos_negativa)} instâncias")
        print(f"  • Rejeitadas: {len(candidatos_rejeitada)} instâncias")
        
        # Gerar imagens
        print("\n🎨 Gerando imagens individuais...")
        
        if exemplo_positiva:
            print(f"  • Positiva (idx={idx_positiva}, id={exemplo_positiva.get('id')})")
            criar_imagem_individual(
                exemplo_positiva, idx_positiva, X_test, 
                class_names, exp_key, t_plus, t_minus,
                img_shape=(28, 28), 
                output_suffix="exemplo_positiva"
            )
        else:
            print("  ⚠ Nenhum exemplo de instância positiva encontrado")
        
        if exemplo_negativa:
            print(f"  • Negativa (idx={idx_negativa}, id={exemplo_negativa.get('id')})")
            criar_imagem_individual(
                exemplo_negativa, idx_negativa, X_test, 
                class_names, exp_key, t_plus, t_minus,
                img_shape=(28, 28), 
                output_suffix="exemplo_negativa"
            )
        else:
            print("  ⚠ Nenhum exemplo de instância negativa encontrado")
        
        if exemplo_rejeitada:
            print(f"  • Rejeitada (idx={idx_rejeitada}, id={exemplo_rejeitada.get('id')})")
            criar_imagem_individual(
                exemplo_rejeitada, idx_rejeitada, X_test, 
                class_names, exp_key, t_plus, t_minus,
                img_shape=(28, 28), 
                output_suffix="exemplo_rejeitada"
            )
        else:
            print("  ⚠ Nenhum exemplo de instância rejeitada encontrado")
        
        print(f"\n✅ Visualizações concluídas para {exp_key}!")
        
    except KeyError as e:
        print(f"❌ ERRO: Chave não encontrada: {e}")
    except Exception as e:
        print(f"❌ ERRO: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Função principal"""
    print("="*80)
    print("VISUALIZADOR DE INSTÂNCIAS INDIVIDUAIS DO PEAB - MNIST")
    print("="*80)
    
    parser = argparse.ArgumentParser(
        description='Gera imagens individuais de exemplos positivos, negativos e rejeitados do MNIST'
    )
    parser.add_argument('--results', type=str, default=RESULTS_FILE, 
                       help='Caminho para o JSON de resultados (ex: json/peab/mnist_3_vs_8.json)')
    parser.add_argument('--show', action='store_true', 
                       help='Mostrar janelas do Matplotlib')
    parser.add_argument('--seed', type=int, default=None,
                       help='Seed aleatória para reproduzir os mesmos exemplos (ex: --seed 42)')
    parser.add_argument('--idx-positiva', type=int, default=None,
                       help='Índice específico para instância positiva (ex: --idx-positiva 104)')
    parser.add_argument('--idx-negativa', type=int, default=None,
                       help='Índice específico para instância negativa (ex: --idx-negativa 14)')
    parser.add_argument('--idx-rejeitada', type=int, default=None,
                       help='Índice específico para instância rejeitada (ex: --idx-rejeitada 13)')
    args = parser.parse_args()
    
    # Configurar seed aleatória se fornecida
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Seed aleatória definida: {args.seed}")
    
    # Armazenar índices específicos (prioridade: linha de comando > constantes no código)
    global INDICES_ESPECIFICOS
    INDICES_ESPECIFICOS = {
        'positiva': args.idx_positiva if args.idx_positiva is not None else IDX_POSITIVA,
        'negativa': args.idx_negativa if args.idx_negativa is not None else IDX_NEGATIVA,
        'rejeitada': args.idx_rejeitada if args.idx_rejeitada is not None else IDX_REJEITADA
    }
    
    # Informar se está usando índices fixos do código
    if IDX_POSITIVA is not None or IDX_NEGATIVA is not None or IDX_REJEITADA is not None:
        print("📌 Usando índices fixos definidos no código:")
        if IDX_POSITIVA is not None:
            print(f"   • Positiva: {IDX_POSITIVA}")
        if IDX_NEGATIVA is not None:
            print(f"   • Negativa: {IDX_NEGATIVA}")
        if IDX_REJEITADA is not None:
            print(f"   • Rejeitada: {IDX_REJEITADA}")
    
    global SHOW_PLOTS
    if args.show:
        SHOW_PLOTS = True
    
    print(f"\n📂 Arquivo: {args.results}")
    
    try:
        # Carregar JSON
        data = carregar_json(args.results)
        print("✓ JSON carregado com sucesso")
        
        # Novo formato: JSON único por dataset (não há mais múltiplos experimentos)
        # Extrair nome do experimento do config
        config = data.get('config', {})
        dataset_name = config.get('dataset_name', 'mnist')
        digit_pair = config.get('mnist_digit_pair', [])
        
        if len(digit_pair) == 2:
            exp_key = f"mnist_{digit_pair[0]}_vs_{digit_pair[1]}"
            print(f"\n✓ Experimento: MNIST {digit_pair[0]} vs {digit_pair[1]}")
        else:
            exp_key = dataset_name
            print(f"\n✓ Experimento: {dataset_name}")
        
        # Processar experimento
        processar_experimento(data, exp_key)
        
        print(f"\n{'='*80}")
        print("✅ PROCESSAMENTO CONCLUÍDO!")
        print(f"{'='*80}")
        
        if SAVE_PLOTS:
            print(f"\n💾 Imagens salvas em: {OUTPUT_DIR}/")
        
    except FileNotFoundError as e:
        print(f"❌ ERRO: {e}")
    except Exception as e:
        print(f"❌ ERRO: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()