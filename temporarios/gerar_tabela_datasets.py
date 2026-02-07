"""
Gera tabela LaTeX com estatísticas dos datasets do PEAB.
"""
import json
import os

# Carregar dados
with open('temporarios/dados_tabela_datasets.json', 'r', encoding='utf-8') as f:
    dados = json.load(f)

# Ordem dos datasets (conforme solicitado)
ordem_datasets = [
    'banknote',
    'vertebral_column',
    'pima_indians_diabetes',
    'heart_disease',
    'creditcard',
    'breast_cancer',
    'covertype',
    'spambase',
    'sonar',
    'mnist'
]

# Nomes bonitos para exibição
nomes_display = {
    'banknote': 'Banknote',
    'vertebral_column': 'Vertebral Column',
    'pima_indians_diabetes': 'Pima Indians Diabetes',
    'heart_disease': 'Heart Disease',
    'creditcard': 'Credit Card',
    'breast_cancer': 'Breast Cancer',
    'covertype': 'Covertype',
    'spambase': 'Spambase',
    'sonar': 'Sonar',
    'mnist': 'MNIST (3 vs 8)'
}

print("\n" + "="*80)
print("GERANDO TABELA LATEX - Estatísticas dos Datasets")
print("="*80 + "\n")

# Início da tabela LaTeX
latex = r"""\begin{table}[htbp]
\centering
\caption{Estatísticas dos Datasets com Rejeição (PEAB)}
\label{tab:datasets_estatisticas}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
\textbf{Dataset} & 
\textbf{Instâncias} & 
\textbf{Features} & 
\textbf{$t^+$} & 
\textbf{$t^-$} & 
\textbf{Zona Rej.} & 
\textbf{Taxa Rej. (\%)} & 
\textbf{Acc. s/ Rej. (\%)} &
\textbf{Acc. c/ Rej. (\%)} \\
\midrule
"""

# Adicionar linhas de dados
for dataset in ordem_datasets:
    if dataset not in dados:
        print(f"⚠️  {dataset} não encontrado, pulando...")
        continue
    
    d = dados[dataset]
    nome = nomes_display.get(dataset, dataset)
    
    # Formatar valores
    instancias = d['instancias']
    features = d['features']
    t_plus = f"{d['t_plus']:.3f}" if isinstance(d['t_plus'], (int, float)) else d['t_plus']
    t_minus = f"{d['t_minus']:.3f}" if isinstance(d['t_minus'], (int, float)) else d['t_minus']
    zona_rej = f"{d['zona_rejeicao']:.3f}" if isinstance(d['zona_rejeicao'], (int, float)) else d['zona_rejeicao']
    taxa_rej = f"{d['taxa_rejeicao']:.2f}" if isinstance(d['taxa_rejeicao'], (int, float)) else d['taxa_rejeicao']
    acc_sem = f"{d['acuracia_sem_rej']:.2f}" if isinstance(d['acuracia_sem_rej'], (int, float)) else d['acuracia_sem_rej']
    acc_com = f"{d['acuracia']:.2f}" if isinstance(d['acuracia'], (int, float)) else d['acuracia']
    
    # Adicionar linha
    latex += f"{nome} & {instancias} & {features} & {t_plus} & {t_minus} & {zona_rej} & {taxa_rej} & {acc_sem} & {acc_com} \\\\\n"

# Finalizar tabela
latex += r"""\bottomrule
\end{tabular}%
}
\begin{minipage}{\textwidth}
\footnotesize
\textbf{Legenda:} $t^+$ e $t^-$ são os thresholds de decisão otimizados; 
Zona Rej. = $t^+ - t^-$ (largura da zona de rejeição); 
Taxa Rej. = \% de instâncias rejeitadas no teste; 
Acc. s/ Rej. = acurácia sem rejeição (classificador base); 
Acc. c/ Rej. = acurácia com rejeição (apenas instâncias aceitas).
\end{minipage}
\end{table}
"""

# Salvar tabela
output_file = "results/tabelas_latex/tabela_datasets.tex"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(latex)

print(f"✅ Tabela LaTeX salva em: {output_file}")
print()

# Mostrar preview
print("="*80)
print("PREVIEW DA TABELA:")
print("="*80)
print(latex)

# Estatísticas resumidas
print("\n" + "="*80)
print("ESTATÍSTICAS RESUMIDAS:")
print("="*80)

total_inst = sum(d['instancias'] for d in dados.values() if isinstance(d['instancias'], int))
media_features = sum(d['features'] for d in dados.values() if isinstance(d['features'], int)) / len(dados)
media_taxa_rej = sum(d['taxa_rejeicao'] for d in dados.values() if isinstance(d['taxa_rejeicao'], (int, float))) / len(dados)
media_ganho_acc = sum((d['acuracia'] - d['acuracia_sem_rej']) for d in dados.values() if isinstance(d['acuracia'], (int, float)) and isinstance(d['acuracia_sem_rej'], (int, float))) / len(dados)

print(f"Total de instâncias (teste): {total_inst:,}")
print(f"Média de features: {media_features:.1f}")
print(f"Taxa de rejeição média: {media_taxa_rej:.2f}%")
print(f"Ganho médio de acurácia: {media_ganho_acc:.2f}%")
print("="*80 + "\n")

print("✅ PRONTO! Tabela gerada com sucesso!")
print(f"   Arquivo: {output_file}")
print("   Use \\input{results/tabelas_latex/tabela_datasets.tex} no LaTeX")
