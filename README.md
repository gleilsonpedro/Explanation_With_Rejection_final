Olá! O nosso objetivo é ajustar o notebook `MINABRO_MLP.ipynb`.

A tarefa principal é modificar o notebook para que, ao final da execução de um experimento com um dataset, ele gere dois artefatos principais:

1.  **Um arquivo JSON completo:**
    *   **Onde salvar:** `json/MLP/`
    *   **O que salvar:** Um único arquivo JSON contendo todas as informações possíveis sobre o experimento.

2.  **Um relatório de texto detalhado:**
    *   **Onde salvar:** `results/report_MLP/`
    *   **Como gerar:** O relatório deve ser gerado a partir do arquivo JSON criado no passo anterior, preferencialmente em uma célula separada no final do notebook.
    *   **Formato:** O relatório deve ser similar em conteúdo e formato aos relatórios gerados pelo script `peab.py`.

**Detalhes que o relatório (e o JSON) devem conter:**

*   **Resumo do Experimento:**
    *   Nome do dataset.
    *   Número de instâncias e features.
    *   Tamanho do split de treino e teste.

*   **Configuração do Modelo (MLP):**
    *   Número de neurônios em cada camada oculta.
    *   Número de épocas de treinamento (backpropagation) que o modelo realmente executou (`n_iter_`).

*   **Zona de Rejeição:**
    *   Os valores dos limiares `t+` e `t-`.
    *   O "tamanho" da zona de rejeição (`t+` - `t-`).

*   **Desempenho:**
    *   Acurácia **antes** da rejeição.
    *   Acurácia **depois** da rejeição (calculada apenas nas instâncias aceitas).
    *   Número de instâncias classificadas como positivas, negativas e rejeitadas.

*   **Estatísticas das Explicações:**
    *   Tamanho médio das explicações para instâncias positivas.
    *   Tamanho médio das explicações para instâncias negativas.
    *   Tamanho médio das explicações para instâncias rejeitadas.

*   **Tempo de Execução:**
    *   Tempo total para gerar todas as explicações.
    *   Tempo médio por instância.

O fluxo de trabalho é: executar o notebook -> salvar JSON -> executar célula final -> gerar relatório de texto a partir do JSON.