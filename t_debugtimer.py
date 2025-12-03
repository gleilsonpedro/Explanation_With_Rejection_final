import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Imports do seu projeto
from peab import (
    configurar_experimento, treinar_e_avaliar_modelo,
    gerar_explicacao_instancia
)
from src.milp_solver import calcular_minimo_exato_pulp
from config.datasets import DEFAULT_LOGREG_PARAMS, RANDOM_STATE

def cronometro(nome, inicio):
    fim = time.perf_counter()
    print(f"   -> [{nome}] Concluído em: {fim - inicio:.4f} segundos")
    return fim

def executar_teste_isolado():
    print("\n=== TESTE DE VELOCIDADE ISOLADO (MNIST - 5 AMOSTRAS) ===")
    
    # 1. Configuração (Rápida)
    print("\n1. Carregando Dataset...")
    # Forçamos MNIST aqui manualmente para o teste
    dataset_name = "mnist" 
    
    # Carrega dataset (pode demorar um pouco se não estiver em cache)
    X, y, nomes, rc, ts = configurar_experimento(dataset_name)
    
    # 2. Treino
    print("2. Treinando Modelo...")
    t0 = time.perf_counter()
    modelo, t_plus, t_minus, _ = treinar_e_avaliar_modelo(X, y, ts, rc, DEFAULT_LOGREG_PARAMS)
    cronometro("Treino", t0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=RANDOM_STATE, stratify=y)
    
    print(f"\n3. Iniciando Loop de Teste (5 Instâncias Aleatórias)...")
    print(f"   Vamos cronometrar PEAB vs PuLP separadamente.\n")
    
    # Pega 5 índices aleatórios
    indices_teste = np.random.choice(len(X_test), 5, replace=False)
    
    for i, idx in enumerate(indices_teste):
        instancia = X_test.iloc[[idx]]
        score = modelo.decision_function(instancia)[0]
        
        # Identifica classe
        if score >= t_plus: tipo = "POSITIVA"
        elif score <= t_minus: tipo = "NEGATIVA"
        else: tipo = "REJEITADA"
        
        print(f"--- AMOSTRA #{i+1} (ID: {idx}) | Tipo: {tipo} | Score: {score:.4f} ---")
        
        # TESTE 1: PEAB
        print("   > Iniciando PEAB...")
        t_start = time.perf_counter()
        expl, _, _, _ = gerar_explicacao_instancia(instancia, modelo, X_train, t_plus, t_minus)
        t_peab = time.perf_counter() - t_start
        print(f"     [PEAB] Tempo: {t_peab:.4f}s | Tamanho: {len(expl)}")
        
        # TESTE 2: PuLP (Otimização)
        print("   > Iniciando PuLP (Solver)...")
        t_start = time.perf_counter()
        tam_opt = calcular_minimo_exato_pulp(modelo, instancia, X_train, t_plus, t_minus)
        t_opt = time.perf_counter() - t_start
        print(f"     [PuLP] Tempo: {t_opt:.4f}s | Tamanho: {tam_opt}")
        
        # Comparação
        vencedor = "PEAB" if t_peab < t_opt else "PuLP"
        fator = t_peab / t_opt if t_peab > t_opt else t_opt / t_peab
        print(f"   => VENCEDOR TEMPO: {vencedor} ({fator:.1f}x mais rápido)")
        print("")

if __name__ == "__main__":
    executar_teste_isolado()