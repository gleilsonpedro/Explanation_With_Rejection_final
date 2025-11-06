import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt


def load_results(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_methods(results: dict):
    present = {}
    # normalizar chaves comuns
    for key in results.keys():
        k = key.lower()
        if k in ("peab", "anchor", "minexp"):
            present[key] = results[key]
    # lidar com variação de maiúsculas
    if "MinExp" in results:
        present["MinExp"] = results["MinExp"]
    if "peab" in results and "PEAB" not in present:
        present["PEAB"] = results["peab"]
    if "anchor" in results and "Anchor" not in present:
        present["Anchor"] = results["anchor"]
    return present


def get_mnist_entry(method_blob: dict):
    # Estrutura esperada: method_blob["mnist"]
    if not isinstance(method_blob, dict):
        return None
    return method_blob.get("mnist")


def pixel_name_to_index(name) -> int | None:
    """Converte nomes de features ou índices para índice 0..783.
    Aceita: 'pixel359', 'pixel_359', 359 (int), '359' (str).
    """
    # int direto
    if isinstance(name, int):
        idx = name
    else:
        s = str(name)
        # tentar extrair número após "pixel" ou pegar número cru
        m = re.search(r"pixel[_]?(\d+)", s, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1))
        else:
            if s.isdigit():
                idx = int(s)
            else:
                return None
    # normalizações: alguns runs usam 1-based
    if 1 <= idx <= 784:
        idx = idx - 1
    # limitar ao range válido 0..783
    if idx < 0:
        return None
    if idx >= 28 * 28:
        # normalizar pelo módulo se extrapolar
        idx = idx % (28 * 28)
    return idx


def map_idx_to_bin(idx: int, target_size: int) -> tuple[int, int]:
    """Mapeia índice 0..783 (28x28) para célula na grade target_size x target_size via binning proporcional."""
    r, c = divmod(idx, 28)
    br = int(r * target_size / 28)
    bc = int(c * target_size / 28)
    # resguardar limites
    br = min(max(br, 0), target_size - 1)
    bc = min(max(bc, 0), target_size - 1)
    return br, bc


def explanation_features_to_mask(feature_names, selected_feats, target_size: int = 28) -> np.ndarray:
    """Converte uma lista de features explicadas em uma máscara target_size x target_size (contagens)."""
    mask = np.zeros((target_size, target_size), dtype=float)
    if not selected_feats:
        return mask
    # Evitar duplicidade por instância
    for feat in set(selected_feats):
        idx = pixel_name_to_index(feat)
        if idx is None:
            continue
        br, bc = map_idx_to_bin(idx, target_size)
        mask[br, bc] += 1.0
    return mask


def to_int_or_none(v):
    try:
        return int(v)
    except Exception:
        return None


def aggregate_masks(feature_names, per_instance, digits: tuple[int, int], target_size: int):
    """Soma máscaras por grupo A (classe digits[0]), grupo B (classe digits[1]) e rejeitadas (entre essas classes).
    Retorna máscaras e contagens, e os índices de X_test por grupo para calcular imagens médias.
    """
    dA, dB = digits
    A_mask = np.zeros((target_size, target_size), dtype=float)
    B_mask = np.zeros((target_size, target_size), dtype=float)
    rej_mask = np.zeros((target_size, target_size), dtype=float)

    idxs_A = []
    idxs_B = []
    idxs_R = []
    n_A = n_B = n_R = 0

    for item in per_instance:
        y_true = to_int_or_none(item.get('y_true'))
        if y_true is None or y_true not in (dA, dB):
            continue
        feats = item.get('explanation') or []
        m = explanation_features_to_mask(feature_names, feats, target_size)
        idx = int(item.get('id', -1))
        if bool(item.get('rejected', False)):
            rej_mask += m
            n_R += 1
            idxs_R.append(idx)
        else:
            if y_true == dA:
                A_mask += m
                n_A += 1
                idxs_A.append(idx)
            elif y_true == dB:
                B_mask += m
                n_B += 1
                idxs_B.append(idx)

    return (A_mask, B_mask, rej_mask), (n_A, n_B, n_R), (idxs_A, idxs_B, idxs_R)


def normalize_masks(masks, counts):
    """Normaliza cada máscara pelo número de instâncias do grupo (frequência média por instância)."""
    pos, neg, rej = masks
    n_pos, n_neg, n_rej = counts
    def safe_div(m, n):
        return m / max(n, 1)
    return safe_div(pos, n_pos), safe_div(neg, n_neg), safe_div(rej, n_rej)


def threshold_mask(mask: np.ndarray, percentile: float | None = 95, top_k: int | None = None) -> np.ndarray:
    """Retorna máscara binária pelos pixels mais significativos.
    - Se percentile for definido: mantém valores >= percentil.
    - Caso contrário, usa top_k (número de pixels com maior valor).
    """
    flat = mask.flatten()
    if top_k is not None and (percentile is None):
        if top_k <= 0:
            return np.zeros_like(mask)
        # índices dos top_k
        idxs = np.argpartition(-flat, min(top_k - 1, flat.size - 1))[:top_k]
        out = np.zeros_like(flat)
        out[idxs] = 1.0
        return out.reshape(mask.shape)
    # usa percentil por padrão
    thr = np.percentile(flat, percentile)
    return (mask >= thr).astype(float)


def downscale_28_to_target(arr28: np.ndarray, target_size: int, agg: str = 'mean') -> np.ndarray:
    out = np.zeros((target_size, target_size), dtype=float)
    cnt = np.zeros((target_size, target_size), dtype=float)
    H, W = 28, 28
    for r in range(H):
        br = int(r * target_size / H)
        for c in range(W):
            bc = int(c * target_size / W)
            out[br, bc] += arr28[r, c]
            cnt[br, bc] += 1.0
    if agg == 'mean':
        out = np.divide(out, np.maximum(cnt, 1.0))
    return out


def compute_mean_image(X_group: np.ndarray, target_size: int) -> np.ndarray:
    if X_group.size == 0:
        return np.zeros((target_size, target_size), dtype=float)
    img = np.mean(X_group, axis=0)
    img28 = img.reshape(28, 28)
    imgT = downscale_28_to_target(img28, target_size, agg='mean')
    imgT = (imgT - imgT.min()) / (imgT.ptp() + 1e-8)
    return imgT


def save_mask(mask: np.ndarray, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(4, 4))
    # Normalizar para [0,1] se possível
    m = mask.copy()
    vmax = np.max(m) if np.max(m) > 0 else 1.0
    plt.imshow(m / vmax, cmap='Reds', vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_overlay_binary_on_mean(mean_img: np.ndarray, bin_mask: np.ndarray, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(mean_img, cmap='gray', vmin=0.0, vmax=1.0)
    # desenhar pixels significativos
    overlay = np.ma.masked_where(bin_mask < 0.5, bin_mask)
    plt.imshow(overlay, cmap='autumn', alpha=0.7)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def pick_representative_instance(per_instance, cond):
    """Escolhe a instância com maior explicação (mais pixels) dentro de um subconjunto."""
    best = None
    best_len = -1
    for it in per_instance:
        if not cond(it):
            continue
        L = len(it.get('explanation') or [])
        if L > best_len:
            best = it
            best_len = L
    return best


def save_instance_overlay(X_test: np.ndarray, inst: dict, title: str, method_name: str, out_path: str, target_size: int):
    if inst is None:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    idx = int(inst.get('id', 0))
    # validar X_test
    if not isinstance(X_test, np.ndarray) or X_test.ndim < 2 or idx < 0 or idx >= X_test.shape[0]:
        return
    pixels = np.array(X_test[idx])
    img28 = pixels.reshape(28, 28)
    imgT = downscale_28_to_target(img28, target_size, agg='mean')
    # normalizar imagem para melhor contraste
    imgT = (imgT - imgT.min()) / (imgT.ptp() + 1e-8)
    mask = np.zeros((target_size * target_size,), dtype=float)
    for feat in set(inst.get('explanation') or []):
        pidx = pixel_name_to_index(feat)
        if pidx is not None:
            br, bc = map_idx_to_bin(pidx, target_size)
            mask[br * target_size + bc] = 1
    mask_img = mask.reshape(target_size, target_size)
    plt.figure(figsize=(4, 4))
    plt.imshow(imgT, cmap='gray', vmin=0.0, vmax=1.0)
    overlay = np.ma.masked_where(mask_img < 0.5, mask_img)
    plt.imshow(overlay, cmap='autumn', alpha=0.7)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_mnist_images_for_method(method_name: str, method_blob: dict, output_dir: str, digits: tuple[int, int], target_size: int):
    entry = get_mnist_entry(method_blob)
    if not entry:
        print(f"[WARN] Método {method_name}: não há entrada para 'mnist' no JSON.")
        return []

    data = entry.get('data') or {}
    feature_names = data.get('feature_names') or [f"pixel{i+1}" for i in range(28*28)]
    per_instance = entry.get('per_instance') or []
    X_test_raw = data.get('X_test')
    try:
        X_test = np.array(X_test_raw if X_test_raw is not None else [])
    except Exception:
        X_test = np.array([])

    dA, dB = digits
    (A_mask, B_mask, rej_mask), counts, (idxs_A, idxs_B, idxs_R) = aggregate_masks(
        feature_names, per_instance, digits, target_size
    )
    n_A, n_B, n_R = counts
    A_norm, B_norm, rej_norm = normalize_masks((A_mask, B_mask, rej_mask), counts)

    label_suffix = f"_{dA}-vs-{dB}"
    out_files = []
    # 1) Heatmaps normalizados (frequência média por instância)
    out_pos = os.path.join(output_dir, f"mnist_{method_name}_classe{dA}_heatmap{label_suffix}.png")
    save_mask(A_norm, f"MNIST {method_name} - Classe {dA}{label_suffix} (n={n_A})", out_pos)
    out_files.append(out_pos)

    out_neg = os.path.join(output_dir, f"mnist_{method_name}_classe{dB}_heatmap{label_suffix}.png")
    save_mask(B_norm, f"MNIST {method_name} - Classe {dB}{label_suffix} (n={n_B})", out_neg)
    out_files.append(out_neg)

    out_rej = os.path.join(output_dir, f"mnist_{method_name}_rejeitadas_heatmap{label_suffix}.png")
    save_mask(rej_norm, f"MNIST {method_name} - Rejeitadas{label_suffix} (n={n_R})", out_rej)
    out_files.append(out_rej)

    # 2) Máscaras binárias por percentil/top-K sobre imagem média do grupo
    # usar percentil 95 por padrão; se poucos pixels destacados, cair para top_k
    def bin_mask_for(m):
        b = threshold_mask(m, percentile=95, top_k=None)
        if b.sum() < 20:
            b = threshold_mask(m, percentile=None, top_k=100)
        return b

    # criar imagens médias
    n_samples = X_test.shape[0] if isinstance(X_test, np.ndarray) and X_test.ndim >= 2 else 0
    mean_pos = compute_mean_image(X_test[idxs_A]) if len(idxs_A) > 0 and n_samples > 0 else np.zeros((target_size, target_size))
    mean_neg = compute_mean_image(X_test[idxs_B]) if len(idxs_B) > 0 and n_samples > 0 else np.zeros((target_size, target_size))
    mean_rej = compute_mean_image(X_test[idxs_R]) if len(idxs_R) > 0 and n_samples > 0 else np.zeros((target_size, target_size))

    pos_bin = bin_mask_for(A_norm)
    neg_bin = bin_mask_for(B_norm)
    rej_bin = bin_mask_for(rej_norm)

    out_pos_overlay = os.path.join(output_dir, f"mnist_{method_name}_classe{dA}_overlay{label_suffix}.png")
    save_overlay_binary_on_mean(mean_pos, pos_bin, f"MNIST {method_name} - Classe {dA} (pixels significativos){label_suffix}", out_pos_overlay)
    out_files.append(out_pos_overlay)

    out_neg_overlay = os.path.join(output_dir, f"mnist_{method_name}_classe{dB}_overlay{label_suffix}.png")
    save_overlay_binary_on_mean(mean_neg, neg_bin, f"MNIST {method_name} - Classe {dB} (pixels significativos){label_suffix}", out_neg_overlay)
    out_files.append(out_neg_overlay)

    out_rej_overlay = os.path.join(output_dir, f"mnist_{method_name}_rejeitadas_overlay{label_suffix}.png")
    save_overlay_binary_on_mean(mean_rej, rej_bin, f"MNIST {method_name} - Rejeitadas (pixels significativos){label_suffix}", out_rej_overlay)
    out_files.append(out_rej_overlay)

    # 3) Exemplos representativos (maior explicação) por grupo
    pos_rep = pick_representative_instance(per_instance, lambda it: (to_int_or_none(it.get('y_true')) == dA) and not it.get('rejected', False))
    neg_rep = pick_representative_instance(per_instance, lambda it: (to_int_or_none(it.get('y_true')) == dB) and not it.get('rejected', False))
    rej_rep = pick_representative_instance(per_instance, lambda it: (to_int_or_none(it.get('y_true')) in (dA, dB)) and bool(it.get('rejected', False)))

    out_pos_rep = os.path.join(output_dir, f"mnist_{method_name}_classe{dA}_exemplo{label_suffix}.png")
    save_instance_overlay(X_test, pos_rep, f"Exemplo Classe {dA} ({method_name})", method_name, out_pos_rep, target_size)
    out_files.append(out_pos_rep)

    out_neg_rep = os.path.join(output_dir, f"mnist_{method_name}_classe{dB}_exemplo{label_suffix}.png")
    save_instance_overlay(X_test, neg_rep, f"Exemplo Classe {dB} ({method_name})", method_name, out_neg_rep, target_size)
    out_files.append(out_neg_rep)

    out_rej_rep = os.path.join(output_dir, f"mnist_{method_name}_rejeitada_exemplo{label_suffix}.png")
    save_instance_overlay(X_test, rej_rep, f"Exemplo Rejeitado ({method_name})", method_name, out_rej_rep, target_size)
    out_files.append(out_rej_rep)

    print(f"[OK] Imagens salvas para {method_name} ({dA} vs {dB}):\n  - {out_pos}\n  - {out_neg}\n  - {out_rej}")
    return out_files


def main():
    # Gera imagens focadas em classe A vs classe B (ao invés de one-vs-rest) e permite downscale 28->10.
    # Detecta o caminho do JSON de forma robusta (na raiz do repo ou em json/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, os.pardir))
    candidates = [
        os.path.join(repo_root, 'comparative_results.json'),
        os.path.join(repo_root, 'json', 'comparative_results.json'),
    ]
    json_path = next((p for p in candidates if os.path.exists(p)), None)
    if not json_path:
        raise FileNotFoundError(
            "Arquivo 'comparative_results.json' não encontrado. Procurei em: "
            + ", ".join(candidates)
        )

    results = load_results(json_path)
    methods = find_methods(results)
    if not methods:
        print("[ERRO] Nenhum método encontrado no JSON (esperado: PEAB/MinExp/Anchor).")
        return

    # Perguntar classes A vs B
    print("Digite duas classes do MNIST para comparar (ex: 8 5): ", end="")
    raw = input().strip()
    parts = re.findall(r"\d+", raw)
    if len(parts) < 2:
        print("Entrada inválida. Informe dois dígitos entre 0 e 9.")
        return
    dA, dB = int(parts[0]), int(parts[1])
    if not (0 <= dA <= 9 and 0 <= dB <= 9 and dA != dB):
        print("Dígitos inválidos. Use valores distintos entre 0 e 9.")
        return

    # Tamanho alvo (10x10) para acelerar visualização
    target_size = 10

    output_dir = os.path.join('results', 'mnist_images')
    os.makedirs(output_dir, exist_ok=True)

    # Se quiser limitar a um método, defina aqui. Por padrão, gera para todos presentes.
    generated = []
    for method_name, blob in methods.items():
        files = generate_mnist_images_for_method(method_name, blob, output_dir, (dA, dB), target_size)
        generated.extend(files)

    if generated:
        print("\nArquivos gerados:")
        for p in generated:
            print(" -", p)
    else:
        print("Nada gerado. Verifique se 'mnist' existe no JSON e se 'per_instance' possui explicações.")


if __name__ == '__main__':
    main()
