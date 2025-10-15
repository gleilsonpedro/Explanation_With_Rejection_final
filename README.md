
---

# PI-Explainer: Explicações de Modelos Lineares com Opção de Rejeição

Este projeto implementa e avalia um método para gerar **PI-Explicações** para modelos de Regressão Logística, com um diferencial crucial: a capacidade do modelo de **rejeitar** a classificação de instâncias ambíguas, sobre as quais ele não tem certeza. O objetivo é fornecer explicações que não sejam apenas **mínimas** (usando o menor número de features possível), mas também **robustas** (a explicação se mantém mesmo sob perturbações adversárias).

### Conceitos Fundamentais

* **PI-Explicação:** Inspirado no conceito de "Prime Implicant" da lógica booleana, é um conjunto *suficiente* de features e seus valores que, sozinhos, justificam a predição do modelo para uma dada instância.
* **Classificador com Rejeição:** Um modelo que, além de classificar em Classe 0 ou Classe 1, pode optar por não tomar uma decisão ("rejeitar"), caso a instância caia em uma "zona de incerteza". Isso aumenta a confiabilidade das predições que são de fato realizadas.
* **Robustez:** Uma explicação é robusta se a predição do modelo não se altera mesmo que todas as features *fora* da explicação sejam modificadas para seus piores valores possíveis (aqueles que mais empurram a decisão na direção oposta).
* **Minimalidade:** O objetivo de encontrar a PI-Explicação robusta com o menor número possível de features.

---

## A Jornada de uma Instância: Do Dado Cru à Explicação Final

Esta é a história de como uma única linha de dados é processada pelo nosso sistema para gerar uma explicação compreensível e confiável.

### Etapa 1: A Preparação do Terreno

Tudo começa com a seleção e preparação dos dados, um processo orquestrado por dois scripts principais.

* **`datasets.py`:** Através de um menu interativo na função `selecionar_dataset_e_classe()`, o usuário escolhe um dos 10 datasets de teste. Este script é responsável por:
    * Carregar os dados brutos.
    * Realizar limpezas específicas, como no **`Pima`**, onde valores impossíveis (ex: `glicose = 0`) são removidos.
    * Lidar com desafios de escala, como no **`creditcard`**, onde uma amostra estratificada de **20%** dos dados é utilizada para manter a viabilidade computacional.
    * Transformar problemas multi-classe (como `Wine` e `Sonar`) em problemas binários, permitindo que o usuário defina qual classe original será a Classe 0 e qual será a Classe 1.

* **`find_best_hyperparameters.py`:** Antes da análise principal, este script realiza uma busca exaustiva (Grid Search com Validação Cruzada) para encontrar a combinação ótima de hiperparâmetros (`penalty`, `C`, `solver`, etc.) para a `LogisticRegression` em *cada* um dos datasets. Os melhores parâmetros são salvos no arquivo `hiperparametros.json`, garantindo que nosso modelo principal seja sempre o mais performático possível.

### Etapa 2: A Construção e o Treinamento do Modelo

Com os dados e os melhores parâmetros em mãos, o script principal `pi_explainer_additive_bidirecional.py` assume o controle.

* **A Linha de Montagem (`Pipeline`):** A primeira ação na função `main()` é criar uma `Pipeline` do Scikit-learn. Esta é a nossa "linha de montagem" que garante a consistência e o rigor metodológico. Ela contém duas estações:
    1.  `StandardScaler`: Padroniza a escala das features.
    2.  `LogisticRegression`: O nosso classificador.
* **Treinamento (`pipeline.fit`)**: A `Pipeline` é treinada **apenas com os dados de treino**. O `StandardScaler` aprende as médias e desvios e, em seguida, a `LogisticRegression` aprende seus `pesos` com base nos dados já padronizados e nos hiperparâmetros otimizados do arquivo JSON. Isso previne qualquer "vazamento de dados" do conjunto de teste.
* **Definindo a Incerteza (`calcular_thresholds`)**: Após o treino, o sistema analisa as predições no conjunto de treino para definir a "zona de incerteza". A função `calcular_thresholds()` encontra os limiares `t+` e `t-` ideais, minimizando um custo que equilibra o erro de classificação versus o custo de rejeitar uma instância. Qualquer instância cujo score caia entre `t-` e `t+` será, a partir de agora, **rejeitada**.

### Etapa 3: A Geração da Explicação

Para cada instância no conjunto de teste, o processo de explicação individual começa.

* **A Contribuição de Cada Feature (`calculate_deltas`)**: O cérebro da explicação. Esta função calcula o "delta" de cada feature, que representa o quanto o valor daquela feature empurrou o score da instância para longe do "pior caso" possível. É uma medida de impacto.
* **A Primeira Versão (`one_explanation`)**: Com base nos deltas, o sistema constrói uma explicação inicial, adicionando as features de maior impacto até que a predição do modelo seja justificada.

### Etapa 4: O Refinamento (A Busca por Robustez e Minimalidade)

A explicação inicial é boa, mas não é garantidamente robusta ou mínima. Agora ela passa por um processo de refinamento de duas fases.

* **Para Instâncias CLASSIFICADAS (Caminho Único):**
    1.  **Fase 1 - Reforço (`executar_fase_1_reforco_unidirecional`):** O sistema testa a explicação. Se ela não for robusta a uma perturbação adversária, o algoritmo adiciona mais features (as próximas de maior impacto) até que a robustez seja alcançada.
    2.  **Fase 2 - Minimização (`executar_fase_2_minimizacao_unidirecional`):** Com uma explicação robusta em mãos, o sistema tenta "enxugá-la". Ele tenta remover features, uma por uma (começando pela de menor impacto), e só mantém a remoção se a explicação continuar robusta.

* **O Tratamento Especial para Instâncias REJEITADAS: A Busca Dupla**
    * Aqui reside uma das contribuições mais sofisticadas do projeto. Como uma instância rejeitada não tem uma "direção" clara, testamos ambos os cenários para encontrar a explicação mais enxuta possível.
    * A função `encontrar_explicacao_otimizada_para_rejeitada()` orquestra uma **busca dupla**:
        1.  **Caminho 1:** Roda o processo completo de reforço e minimização com os deltas ordenados para evitar que a instância vire **Classe 0**.
        2.  **Caminho 2:** Roda o processo completo novamente, mas com os deltas ordenados para evitar que a instância vire **Classe 1**.
    * **A Decisão Final:** O sistema compara as duas explicações robustas resultantes e **escolhe a que tiver o menor número de features**. Isso garante que a PI-Explicação para um caso de rejeição seja o mais minimalista possível.

### Etapa 5: O Relatório Final

Finalmente, a função `gerar_relatorio_consolidado()` coleta os resultados de todas as instâncias de teste e gera um arquivo `.txt` detalhado, contendo as estatísticas de desempenho, as explicações finais para cada instância e os logs completos do processo de geração, servindo como um registro transparente e completo de todo o experimento.