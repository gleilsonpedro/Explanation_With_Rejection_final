# Problema: Arquivos JSON Grandes no GitHub

## 🚨 Erro Original:
```
remote: error: File json/pulp/newsgroups.json is 101.76 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File json/pulp/rcv1.json is 103.02 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: warning: File json/pulp/covertype.json is 51.95 MB; this is larger than GitHub's recommended maximum file size of 50.00 MB
```

## ✅ Solução Aplicada:

### 1. Adicionei ao `.gitignore`:
```gitignore
# Datasets grandes que geram JSONs > 50 MB
json/pulp/covertype.json
json/pulp/newsgroups.json
json/pulp/rcv1.json
json/pulp/creditcard.json
json/peab/covertype.json
json/peab/newsgroups.json
json/peab/rcv1.json
json/peab/creditcard.json
# ... (todos os métodos)
```

### 2. Removi do Git (mas mantive localmente):
```bash
git rm --cached json/pulp/covertype.json
git rm --cached json/pulp/newsgroups.json
git rm --cached json/pulp/rcv1.json
```

### 3. Fiz commit:
```bash
git commit -m "Adiciona JSONs grandes ao .gitignore (covertype, newsgroups, rcv1 > 50MB)"
```

## 📊 Tamanhos dos Arquivos:
| Arquivo | Tamanho | Status |
|---------|---------|--------|
| covertype.json | 54 MB | ⚠️ Acima de 50 MB (recomendado) |
| newsgroups.json | 107 MB | ❌ Acima de 100 MB (limite) |
| rcv1.json | 107 MB | ❌ Acima de 100 MB (limite) |

## 🎯 Próximos Passos:

### Agora você pode fazer push:
```bash
git push origin main
```

### Os arquivos permanecem localmente:
- ✅ Você **ainda tem** os arquivos em `json/pulp/`
- ✅ Eles só **não vão** para o GitHub
- ✅ Seu trabalho local continua funcionando normalmente

### Se precisar compartilhar esses resultados:
1. **Opção 1**: Usar Google Drive/OneDrive para arquivos grandes
2. **Opção 2**: Comprimir em ZIP e compartilhar
3. **Opção 3**: Usar Git LFS (mais complexo)

## 📝 Por Que Isso Aconteceu?

Esses datasets são **muito grandes**:
- **Covertype**: 581k instâncias → JSON de 54 MB
- **Newsgroups**: ~18k instâncias com texto → JSON de 107 MB
- **RCV1**: ~193k documentos → JSON de 107 MB

Cada instância no JSON armazena:
- Features originais
- Scores
- Decisões
- Explicações
- Metadados

**Resultado**: Arquivos JSON gigantes que não cabem no GitHub!

## ✅ Conclusão:

**Problema resolvido!** Os datasets grandes não irão mais para o GitHub, mas continuam funcionando localmente.

**Datasets nos experimentos principais** (que VÃO para o GitHub):
- Banknote, Breast Cancer, Heart Disease, Pima Indians, Sonar, Spambase, Vertebral Column
- Wine (se adicionar)
- MNIST (se adicionar com limite de instâncias)

Todos esses geram JSONs **< 10 MB** e podem ser versionados tranquilamente.
