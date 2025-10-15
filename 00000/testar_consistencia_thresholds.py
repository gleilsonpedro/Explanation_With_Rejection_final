def testar_consistencia_thresholds():
    """
    Verifica se diferentes implementações calculam thresholds similares
    para o mesmo dataset e classificador
    """
    # Carregar dataset (exemplo: iris)
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Treinar modelo
    modelo = LogisticRegression().fit(X_train, y_train)
    
    # Método 1: Seu calcular_thresholds
    t_plus_peab, t_minus_peab = calcular_thresholds(
        modelo, X_train, y_train
    )
    
    # Método 2: rejection_logic.py
    t_plus_anchor, t_minus_anchor, _, _, _ = executar_logica_rejeicao(
        modelo, X_train, y_train, X_test, rejection_cost=0.25
    )
    
    # Método 3: utility.py (Mateus)
    t_plus_mateus, t_minus_mateus = utility.find_thresholds(
        modelo, X_train, y_train, wr=[0.25]
    )
    
    print(f"PEAB: t+={t_plus_peab:.4f}, t-={t_minus_peab:.4f}")
    print(f"Anchor: t+={t_plus_anchor:.4f}, t-={t_minus_anchor:.4f}")
    print(f"Mateus: t+={t_plus_mateus:.4f}, t-={t_minus_mateus:.4f}")
    
    # Verificar diferenças (tolerância de 0.01)
    assert abs(t_plus_peab - t_plus_anchor) < 0.01, "Thresholds PEAB vs Anchor divergem!"
    assert abs(t_plus_peab - t_plus_mateus) < 0.01, "Thresholds PEAB vs Mateus divergem!"