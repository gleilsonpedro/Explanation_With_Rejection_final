"""
================================================================================
SCRIPT: Score Overlap Analyzer
================================================================================

OBJETIVO:
    Diagnosticar a separabilidade real entre classes em pares MNIST, analisando
    a distribuição dos decision scores produzidos pelo método PEAB.
    
    O script responde à questão: "Por que a zona de rejeição colapsou para
    largura zero em pares intuitivamente ambíguos (como 5 vs 6)?"

FUNCIONAMENTO:
    1. Carrega comparative_results.json contendo resultados PEAB para MNIST
    2. Identifica todos os pares MNIST disponíveis (mnist_X_vs_Y)
    3. Apresenta menu interativo para escolha do par a analisar
    4. Extrai decision_scores das instâncias positivas e negativas
    5. Calcula métricas de separação:
       - Cohen's d (tamanho de efeito)
       - KS statistic (Kolmogorov-Smirnov)
       - AUC (Area Under ROC Curve)
       - Percentis dos scores por classe
       - Sobreposição percentual estimada
    6. Identifica os 20 casos mais "borderline" (próximos ao threshold)
    7. Gera visualizações:
       - Histogramas sobrepostos dos scores por classe
       - Distribuições KDE (Kernel Density Estimation)
       - Boxplots comparativos
       - Scatter dos casos borderline
    8. Salva relatório textual e plots em:
       analysis_output/plots/scoreOverlap/mnist_X_vs_Y/

MÉTRICAS EXPLICADAS:
    - Cohen's d: |média₁ - média₂| / desvio_pooled
      * < 0.2: trivial, 0.2-0.5: pequeno, 0.5-0.8: médio, > 0.8: grande
    - KS statistic: distância máxima entre CDFs empíricas (0=idênticas, 1=totalmente separadas)
    - AUC: Probabilidade de score(positivo) > score(negativo)
      * 0.5: sem discriminação, 1.0: separação perfeita
    - Overlap %: Proporção de scores que caem na região de interseção das distribuições

SAÍDA:
    - analysis_output/plots/scoreOverlap/
        └── mnist_X_vs_Y/
            ├── report.txt                  # Relatório textual detalhado
            ├── histograms.png              # Histogramas sobrepostos
            ├── kde_distributions.png       # Densidades suavizadas
            ├── boxplots.png                # Comparação visual
            └── borderline_cases.png        # Scatter dos casos ambíguos

DEPENDÊNCIAS:
    - numpy, scipy (stats), matplotlib
    - sklearn.metrics (roc_auc_score)
    - json, pathlib

USO:
    python audit/score_overlap.py
    
    (Menu interativo guiará a escolha do par MNIST)

AUTOR: Análise de Ambiguidade MNIST - XAI com Rejeição
DATA: 23/11/2025
================================================================================
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Optional


# ==================== CONFIGURAÇÃO ====================

JSON_PATH = Path("json/comparative_results.json")
OUTPUT_BASE = Path("analysis_output/plots/scoreOverlap")


# ==================== CARREGAMENTO ====================

def load_json(path: Path) -> Dict:
    """Carrega JSON de resultados."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_mnist_pairs(data: Dict) -> List[str]:
    """
    Identifica todos os pares MNIST no JSON (formato mnist_X_vs_Y).
    
    Returns:
        Lista de dataset keys ordenada (ex: ['mnist_0_vs_1', 'mnist_5_vs_6', ...])
    """
    peab_data = data.get('peab', {})
    mnist_keys = [k for k in peab_data.keys() if k.startswith('mnist_')]
    return sorted(mnist_keys)


# ==================== EXTRAÇÃO DE SCORES ====================

def extract_scores(data: Dict, dataset_key: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Extrai decision_scores das instâncias positivas e negativas.
    
    Args:
        data: JSON completo
        dataset_key: chave do dataset (ex: 'mnist_5_vs_6')
    
    Returns:
        (scores_positive, scores_negative, t_minus, t_plus)
    """
    peab_data = data['peab'][dataset_key]
    
    # Thresholds
    t_minus = peab_data['thresholds']['t_minus']
    t_plus = peab_data['thresholds']['t_plus']
    
    # Instâncias
    instances = peab_data.get('per_instance', [])
    
    scores_pos = []
    scores_neg = []
    
    for inst in instances:
        score = inst['decision_score']
        true_label = inst['y_true']
        
        if true_label == 1:
            scores_pos.append(score)
        else:
            scores_neg.append(score)
    
    return (np.array(scores_pos), np.array(scores_neg), t_minus, t_plus)


# ==================== MÉTRICAS DE SEPARAÇÃO ====================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Cohen's d: tamanho de efeito padronizado.
    
    d = |μ₁ - μ₂| / σ_pooled
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return float('inf') if np.mean(group1) != np.mean(group2) else 0.0
    
    return abs(np.mean(group1) - np.mean(group2)) / pooled_std


def ks_statistic(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov: distância máxima entre CDFs.
    
    Returns:
        (statistic, p-value)
    """
    return stats.ks_2samp(group1, group2)


def calculate_auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """
    AUC: probabilidade de score(pos) > score(neg).
    """
    y_true = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    y_score = np.concatenate([scores_pos, scores_neg])
    return roc_auc_score(y_true, y_score)


def estimate_overlap(scores_pos: np.ndarray, scores_neg: np.ndarray, bins: int = 50) -> float:
    """
    Estima percentual de sobreposição entre distribuições usando histogramas.
    
    Returns:
        Percentual de sobreposição (0-100)
    """
    all_scores = np.concatenate([scores_pos, scores_neg])
    range_min, range_max = all_scores.min(), all_scores.max()
    
    hist_pos, edges = np.histogram(scores_pos, bins=bins, range=(range_min, range_max), density=True)
    hist_neg, _ = np.histogram(scores_neg, bins=bins, range=(range_min, range_max), density=True)
    
    # Normalizar para somar 1
    hist_pos = hist_pos / hist_pos.sum() if hist_pos.sum() > 0 else hist_pos
    hist_neg = hist_neg / hist_neg.sum() if hist_neg.sum() > 0 else hist_neg
    
    # Overlap = integral do mínimo das duas densidades
    overlap = np.minimum(hist_pos, hist_neg).sum()
    
    return overlap * 100


def percentiles_by_class(scores_pos: np.ndarray, scores_neg: np.ndarray) -> Dict:
    """
    Calcula percentis (0, 25, 50, 75, 100) para cada classe.
    """
    return {
        'positive': {
            'p0': np.min(scores_pos),
            'p25': np.percentile(scores_pos, 25),
            'p50': np.percentile(scores_pos, 50),
            'p75': np.percentile(scores_pos, 75),
            'p100': np.max(scores_pos)
        },
        'negative': {
            'p0': np.min(scores_neg),
            'p25': np.percentile(scores_neg, 25),
            'p50': np.percentile(scores_neg, 50),
            'p75': np.percentile(scores_neg, 75),
            'p100': np.max(scores_neg)
        }
    }


# ==================== CASOS BORDERLINE ====================

def find_borderline_cases(data: Dict, dataset_key: str, 
                          t_minus: float, t_plus: float, 
                          top_k: int = 20) -> List[Dict]:
    """
    Identifica os top_k casos mais próximos da zona de rejeição.
    
    Distância = min(|score - t_minus|, |score - t_plus|)
    """
    instances = data['peab'][dataset_key].get('per_instance', [])
    
    # Calcular distância para cada instância
    borderline = []
    for inst in instances:
        score = inst['decision_score']
        dist_minus = abs(score - t_minus)
        dist_plus = abs(score - t_plus)
        dist = min(dist_minus, dist_plus)
        
        borderline.append({
            'index': inst['id'],
            'score': score,
            'true_label': inst['y_true'],
            'predicted': inst['y_pred'],
            'distance_to_rejection': dist
        })
    
    # Ordenar por distância crescente
    borderline.sort(key=lambda x: x['distance_to_rejection'])
    
    return borderline[:top_k]


# ==================== VISUALIZAÇÕES ====================

def plot_histograms(scores_pos: np.ndarray, scores_neg: np.ndarray, 
                    t_minus: float, t_plus: float,
                    output_path: Path, pair_name: str):
    """
    Gera histogramas sobrepostos com marcação dos thresholds.
    """
    plt.figure(figsize=(12, 6))
    
    bins = 40
    alpha = 0.6
    
    plt.hist(scores_neg, bins=bins, alpha=alpha, label='Classe Negativa (0)', 
             color='steelblue', edgecolor='black')
    plt.hist(scores_pos, bins=bins, alpha=alpha, label='Classe Positiva (1)', 
             color='coral', edgecolor='black')
    
    # Thresholds
    plt.axvline(t_minus, color='red', linestyle='--', linewidth=2, 
                label=f't_minus = {t_minus:.4f}')
    plt.axvline(t_plus, color='darkred', linestyle='--', linewidth=2, 
                label=f't_plus = {t_plus:.4f}')
    
    # Zona de rejeição
    if t_plus > t_minus:
        plt.axvspan(t_minus, t_plus, alpha=0.2, color='yellow', 
                    label='Zona de Rejeição')
    
    plt.xlabel('Decision Score', fontsize=12, fontweight='bold')
    plt.ylabel('Frequência', fontsize=12, fontweight='bold')
    plt.title(f'Distribuição de Scores - {pair_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'histograms.png', dpi=300)
    plt.close()


def plot_kde(scores_pos: np.ndarray, scores_neg: np.ndarray,
             t_minus: float, t_plus: float,
             output_path: Path, pair_name: str):
    """
    Gera KDE (Kernel Density Estimation) suavizado.
    """
    plt.figure(figsize=(12, 6))
    
    # KDE
    from scipy.stats import gaussian_kde
    
    kde_pos = gaussian_kde(scores_pos)
    kde_neg = gaussian_kde(scores_neg)
    
    x_min = min(scores_pos.min(), scores_neg.min())
    x_max = max(scores_pos.max(), scores_neg.max())
    x_range = np.linspace(x_min, x_max, 500)
    
    plt.plot(x_range, kde_neg(x_range), label='Classe Negativa (0)', 
             color='steelblue', linewidth=2)
    plt.plot(x_range, kde_pos(x_range), label='Classe Positiva (1)', 
             color='coral', linewidth=2)
    
    # Fill under curves
    plt.fill_between(x_range, kde_neg(x_range), alpha=0.3, color='steelblue')
    plt.fill_between(x_range, kde_pos(x_range), alpha=0.3, color='coral')
    
    # Thresholds
    plt.axvline(t_minus, color='red', linestyle='--', linewidth=2, 
                label=f't_minus = {t_minus:.4f}')
    plt.axvline(t_plus, color='darkred', linestyle='--', linewidth=2, 
                label=f't_plus = {t_plus:.4f}')
    
    if t_plus > t_minus:
        plt.axvspan(t_minus, t_plus, alpha=0.2, color='yellow', 
                    label='Zona de Rejeição')
    
    plt.xlabel('Decision Score', fontsize=12, fontweight='bold')
    plt.ylabel('Densidade', fontsize=12, fontweight='bold')
    plt.title(f'Densidade de Scores (KDE) - {pair_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'kde_distributions.png', dpi=300)
    plt.close()


def plot_boxplots(scores_pos: np.ndarray, scores_neg: np.ndarray,
                  t_minus: float, t_plus: float,
                  output_path: Path, pair_name: str):
    """
    Gera boxplots comparativos.
    """
    plt.figure(figsize=(10, 6))
    
    data = [scores_neg, scores_pos]
    labels = ['Classe 0', 'Classe 1']
    
    bp = plt.boxplot(data, labels=labels, patch_artist=True, 
                     notch=True, showmeans=True)
    
    colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Linha dos thresholds
    plt.axhline(t_minus, color='red', linestyle='--', linewidth=1.5, 
                label=f't_minus = {t_minus:.4f}')
    plt.axhline(t_plus, color='darkred', linestyle='--', linewidth=1.5, 
                label=f't_plus = {t_plus:.4f}')
    
    plt.ylabel('Decision Score', fontsize=12, fontweight='bold')
    plt.title(f'Boxplot Comparativo - {pair_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'boxplots.png', dpi=300)
    plt.close()


def plot_borderline(borderline_cases: List[Dict], 
                    t_minus: float, t_plus: float,
                    output_path: Path, pair_name: str):
    """
    Scatter dos casos borderline coloridos por classe.
    """
    plt.figure(figsize=(12, 6))
    
    # Separar por classe
    pos_cases = [c for c in borderline_cases if c['true_label'] == 1]
    neg_cases = [c for c in borderline_cases if c['true_label'] == 0]
    
    pos_indices = [c['index'] for c in pos_cases]
    pos_scores = [c['score'] for c in pos_cases]
    
    neg_indices = [c['index'] for c in neg_cases]
    neg_scores = [c['score'] for c in neg_cases]
    
    plt.scatter(neg_indices, neg_scores, c='steelblue', s=80, 
                alpha=0.7, edgecolors='black', label='Classe 0')
    plt.scatter(pos_indices, pos_scores, c='coral', s=80, 
                alpha=0.7, edgecolors='black', label='Classe 1')
    
    # Thresholds
    plt.axhline(t_minus, color='red', linestyle='--', linewidth=2, 
                label=f't_minus = {t_minus:.4f}')
    plt.axhline(t_plus, color='darkred', linestyle='--', linewidth=2, 
                label=f't_plus = {t_plus:.4f}')
    
    if t_plus > t_minus:
        plt.axhspan(t_minus, t_plus, alpha=0.2, color='yellow', 
                    label='Zona de Rejeição')
    
    plt.xlabel('Índice da Instância', fontsize=12, fontweight='bold')
    plt.ylabel('Decision Score', fontsize=12, fontweight='bold')
    plt.title(f'Top-{len(borderline_cases)} Casos Borderline - {pair_name}', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'borderline_cases.png', dpi=300)
    plt.close()


# ==================== RELATÓRIO ====================

def generate_report(dataset_key: str, 
                   scores_pos: np.ndarray, scores_neg: np.ndarray,
                   t_minus: float, t_plus: float,
                   borderline_cases: List[Dict],
                   output_path: Path):
    """
    Gera relatório textual detalhado.
    """
    # Métricas
    cd = cohens_d(scores_pos, scores_neg)
    ks_stat, ks_pval = ks_statistic(scores_pos, scores_neg)
    auc = calculate_auc(scores_pos, scores_neg)
    overlap = estimate_overlap(scores_pos, scores_neg)
    percentiles = percentiles_by_class(scores_pos, scores_neg)
    
    rejection_width = t_plus - t_minus
    
    # Escrever relatório
    with open(output_path / 'report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"RELATÓRIO DE ANÁLISE DE SOBREPOSIÇÃO - {dataset_key}\n")
        f.write("="*80 + "\n\n")
        
        # Contagens
        f.write("[ESTATÍSTICAS DESCRITIVAS]\n")
        f.write(f"  Classe Positiva (1): {len(scores_pos)} instâncias\n")
        f.write(f"  Classe Negativa (0): {len(scores_neg)} instâncias\n")
        f.write(f"  Total: {len(scores_pos) + len(scores_neg)} instâncias\n\n")
        
        # Médias e desvios
        f.write("[MÉDIAS E DESVIOS]\n")
        f.write(f"  Classe Positiva: μ = {np.mean(scores_pos):.6f}, σ = {np.std(scores_pos):.6f}\n")
        f.write(f"  Classe Negativa: μ = {np.mean(scores_neg):.6f}, σ = {np.std(scores_neg):.6f}\n\n")
        
        # Thresholds
        f.write("[THRESHOLDS PEAB]\n")
        f.write(f"  t_minus: {t_minus:.6f}\n")
        f.write(f"  t_plus:  {t_plus:.6f}\n")
        f.write(f"  Largura da Zona de Rejeição: {rejection_width:.6f}\n")
        
        if rejection_width == 0:
            f.write("  ⚠️  ZONA DE REJEIÇÃO COLAPSADA (largura zero)!\n")
        f.write("\n")
        
        # Métricas de separação
        f.write("[MÉTRICAS DE SEPARAÇÃO]\n")
        f.write(f"  Cohen's d: {cd:.4f}\n")
        f.write("    Interpretação: ")
        if cd < 0.2:
            f.write("Trivial (quase nenhuma diferença)\n")
        elif cd < 0.5:
            f.write("Pequeno efeito\n")
        elif cd < 0.8:
            f.write("Efeito médio\n")
        else:
            f.write("Grande efeito (alta separabilidade)\n")
        
        f.write(f"\n  KS Statistic: {ks_stat:.4f} (p-value = {ks_pval:.4e})\n")
        f.write(f"    Interpretação: Distância máxima entre CDFs = {ks_stat*100:.2f}%\n")
        
        f.write(f"\n  AUC (ROC): {auc:.4f}\n")
        f.write("    Interpretação: ")
        if auc < 0.6:
            f.write("Sem discriminação\n")
        elif auc < 0.7:
            f.write("Discriminação fraca\n")
        elif auc < 0.8:
            f.write("Discriminação aceitável\n")
        elif auc < 0.9:
            f.write("Boa discriminação\n")
        else:
            f.write("Excelente discriminação\n")
        
        f.write(f"\n  Sobreposição Estimada: {overlap:.2f}%\n")
        f.write("    Interpretação: Percentual de scores na região de interseção\n\n")
        
        # Percentis
        f.write("[PERCENTIS DOS SCORES]\n")
        f.write("  Classe Positiva:\n")
        for k, v in percentiles['positive'].items():
            f.write(f"    {k}: {v:.6f}\n")
        
        f.write("\n  Classe Negativa:\n")
        for k, v in percentiles['negative'].items():
            f.write(f"    {k}: {v:.6f}\n")
        f.write("\n")
        
        # Casos borderline
        f.write(f"[TOP-{len(borderline_cases)} CASOS BORDERLINE]\n")
        f.write("  (Instâncias mais próximas da zona de rejeição)\n\n")
        f.write(f"  {'Index':<8} {'Score':<12} {'True':<6} {'Pred':<6} {'Dist':<10}\n")
        f.write("  " + "-"*50 + "\n")
        
        for case in borderline_cases:
            f.write(f"  {case['index']:<8} {case['score']:<12.6f} "
                   f"{case['true_label']:<6} {case['predicted']:<6} "
                   f"{case['distance_to_rejection']:<10.6f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETAÇÃO GERAL:\n")
        f.write("="*80 + "\n")
        
        if rejection_width == 0 and cd > 0.8 and auc > 0.9:
            f.write("✓ Separação forte detectada (Cohen's d alto, AUC elevada).\n")
            f.write("✓ Zona de rejeição zero é coerente: não há ambiguidade no subsample.\n")
            f.write("⚠️  Possíveis causas: subsample limpo, features discriminativas, regularização adequada.\n")
        elif rejection_width == 0 and overlap > 30:
            f.write("⚠️  Zona de rejeição zero APESAR de sobreposição moderada detectada!\n")
            f.write("   Possíveis causas: peso de rejeição (wr) alto, MIN_REJECTION_WIDTH=0.\n")
            f.write("   Recomendação: revisar custo de rejeição ou impor largura mínima.\n")
        elif rejection_width > 0 and overlap < 10:
            f.write("✓ Separação forte com zona de rejeição ativa.\n")
            f.write("✓ Configuração adequada para capturar casos borderline.\n")
        else:
            f.write("ℹ️  Configuração intermediária: alguma sobreposição presente.\n")
            f.write("   A zona de rejeição reflete o trade-off erro vs. rejeição otimizado.\n")
        
        f.write("\n")


# ==================== MENU ====================

def display_menu(pairs: List[str]) -> Optional[str]:
    """
    Exibe menu interativo para escolha do par MNIST.
    
    Returns:
        Dataset key escolhido ou None se cancelar
    """
    print("\n" + "="*80)
    print("ANÁLISE DE SOBREPOSIÇÃO DE SCORES - PARES MNIST")
    print("="*80)
    print("\nPares disponíveis:\n")
    
    for idx, pair in enumerate(pairs, 1):
        # Extrair dígitos do nome (ex: mnist_5_vs_6 → "5 vs 6")
        parts = pair.replace('mnist_', '').replace('_', ' ')
        print(f"  [{idx}] {parts}")
    
    print(f"  [0] Sair")
    print("\n" + "-"*80)
    
    while True:
        try:
            choice = input("\nEscolha o par (número): ").strip()
            choice_int = int(choice)
            
            if choice_int == 0:
                return None
            
            if 1 <= choice_int <= len(pairs):
                return pairs[choice_int - 1]
            else:
                print(f"❌ Opção inválida. Escolha entre 0 e {len(pairs)}.")
        except ValueError:
            print("❌ Entrada inválida. Digite um número.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Operação cancelada pelo usuário.")
            return None


# ==================== MAIN ====================

def main():
    """
    Fluxo principal: carregar, escolher, analisar, visualizar.
    """
    print("\n🔍 Carregando comparative_results.json...")
    
    if not JSON_PATH.exists():
        print(f"❌ Arquivo não encontrado: {JSON_PATH}")
        return
    
    data = load_json(JSON_PATH)
    pairs = find_mnist_pairs(data)
    
    if not pairs:
        print("❌ Nenhum par MNIST encontrado no JSON.")
        return
    
    print(f"✓ {len(pairs)} pares MNIST identificados.")
    
    # Menu
    chosen_pair = display_menu(pairs)
    
    if chosen_pair is None:
        print("\n👋 Encerrando.\n")
        return
    
    print(f"\n📊 Analisando par: {chosen_pair}...")
    
    # Extrair scores
    scores_pos, scores_neg, t_minus, t_plus = extract_scores(data, chosen_pair)
    
    print(f"  ✓ {len(scores_pos)} instâncias positivas")
    print(f"  ✓ {len(scores_neg)} instâncias negativas")
    print(f"  ✓ Thresholds: t_minus={t_minus:.4f}, t_plus={t_plus:.4f}")
    
    # Casos borderline
    borderline_cases = find_borderline_cases(data, chosen_pair, t_minus, t_plus, top_k=20)
    
    # Criar diretório de saída
    output_dir = OUTPUT_BASE / chosen_pair
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Salvando resultados em: {output_dir}")
    
    # Gerar relatório
    print("  🔹 Gerando relatório textual...")
    generate_report(chosen_pair, scores_pos, scores_neg, t_minus, t_plus, 
                   borderline_cases, output_dir)
    
    # Gerar plots
    print("  🔹 Gerando histogramas...")
    plot_histograms(scores_pos, scores_neg, t_minus, t_plus, output_dir, chosen_pair)
    
    print("  🔹 Gerando KDE...")
    plot_kde(scores_pos, scores_neg, t_minus, t_plus, output_dir, chosen_pair)
    
    print("  🔹 Gerando boxplots...")
    plot_boxplots(scores_pos, scores_neg, t_minus, t_plus, output_dir, chosen_pair)
    
    print("  🔹 Gerando scatter de casos borderline...")
    plot_borderline(borderline_cases, t_minus, t_plus, output_dir, chosen_pair)
    
    print("\n✅ Análise concluída!")
    print(f"   📄 Relatório: {output_dir / 'report.txt'}")
    print(f"   📈 Plots: {output_dir / '*.png'}")
    print()


if __name__ == '__main__':
    main()
