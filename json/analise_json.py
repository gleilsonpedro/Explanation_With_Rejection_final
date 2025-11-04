import json
import numpy as np
import matplotlib.pyplot as plt
import random

# 1. Carregar o JSON
with open("comparative_results.json") as f:
    results = json.load(f)

# 2. Mostrar menu de métodos
methods = list(results.keys())
print("Métodos disponíveis:", methods)
method = input("Escolha o método: ")

# 3. Mostrar datasets disponíveis
datasets = list(results[method].keys())
print("Datasets disponíveis:", datasets)
dataset = input("Escolha o dataset: ")

# 4. Carregar dados do dataset escolhido
data = results[method][dataset]
instances = data["per_instance"]

# 5. Mostrar rótulos únicos
labels = sorted(set(i["y_true"] for i in instances))
print("Classes disponíveis:", labels)
digit = int(input("Escolha o número (classe): "))

# 6. Selecionar instâncias representativas
def get_instance(instances, cond):
    subset = [i for i in instances if cond(i)]
    return random.choice(subset) if subset else None

positive = get_instance(instances, lambda i: i["y_pred"] == 1 and not i["rejected"] and i["y_true"] == digit)
negative = get_instance(instances, lambda i: i["y_pred"] == 0 and not i["rejected"] and i["y_true"] == digit)
rejected = get_instance(instances, lambda i: i["rejected"] and i["y_true"] == digit)

# 7. Carregar X_test
X_test = np.array(data["data"]["X_test"])  # 784 pixels
img_size = int(np.sqrt(X_test.shape[1]))

# 8. Função de plotagem
def plot_instance(inst, title, method, color='red'):
    if not inst:
        return
    pixels = np.array(X_test[int(inst["id"])])
    img = pixels.reshape(img_size, img_size)

    mask = np.zeros_like(pixels)
    for idx in inst["explanation"]:  # assumindo que explanation = índices das features
        mask[int(idx)] = 1
    mask_img = mask.reshape(img_size, img_size)

    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.imshow(mask_img, cmap='autumn', alpha=0.5)
    plt.axis('off')

    expl_size = len(inst["explanation"])
    time = round(data["computation_time"]["mean_per_instance"], 4)
    legend = (f"Método: {method.upper()}\n"
              f"Classe: {title}\n"
              f"Tamanho explicação: {expl_size} pixels\n"
              f"Tempo médio: {time}s\n"
              f"Dataset: {dataset}\n"
              f"Instância: {inst['id']}")
    plt.title(legend, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{method}_{title.lower()}_{digit}.png", transparent=True)
    plt.close()

# 9. Gerar imagens
plot_instance(positive, "Positiva", method)
plot_instance(negative, "Negativa", method)
plot_instance(rejected, "Rejeitada", method)
