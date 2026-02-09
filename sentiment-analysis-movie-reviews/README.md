# 🎬 Análise de Sentimentos em Avaliações de Produtos

Projeto de Processamento de Linguagem Natural (PLN) para classificação de sentimentos em reviews de filmes utilizando diferentes técnicas de Machine Learning e Deep Learning.

## 📋 Objetivo

Implementar e comparar três abordagens para classificação de sentimentos:
- **SVM + Bag of Words (BoW)**
- **SVM + Word2Vec Embeddings**
- **BERT (Fine-tuning)**

## 📊 Dataset

- **Fonte:** Reviews de filmes extraídas do IMDB
- **Total de reviews:** 545
- **Divisão:** 80% treino | 20% teste
- **Classificação:** Binária (Negativo/Positivo)

| Classe | Critério | Quantidade |
|--------|----------|------------|
| Negativo (0) | Notas 1-5 | 340 |
| Positivo (1) | Notas 6-10 | 205 |

## 🛠️ Metodologia

### 1. SVM + Bag of Words
- Vetorização com `CountVectorizer`
- N-gramas: unigramas e bigramas
- Kernel: Linear

### 2. SVM + Word2Vec
- Embeddings pré-treinados: `word2vec-google-news-300`
- Representação: Média dos vetores das palavras
- Kernel: RBF com `class_weight='balanced'`

### 3. BERT
- Modelo: `bert-base-uncased`
- Fine-tuning: 3 épocas
- Max length: 256 tokens

## 📈 Resultados

### Comparação Geral

| Modelo | Acurácia | F1-Macro | F1-Weighted |
|--------|----------|----------|-------------|
| SVM + BoW | 66.97% | 0.63 | 0.66 |
| SVM + Word2Vec | **78.90%** | **0.78** | **0.79** |
| BERT | **78.90%** | **0.78** | **0.79** |

### Métricas por Classe

| Modelo | Negativo (P/R/F1) | Positivo (P/R/F1) |
|--------|-------------------|-------------------|
| SVM + BoW | 0.71 / 0.79 / 0.75 | 0.58 / 0.46 / 0.51 |
| SVM + Word2Vec | 0.87 / 0.78 / 0.82 | 0.69 / 0.80 / 0.74 |
| BERT | 0.89 / 0.75 / 0.82 | 0.67 / 0.85 / 0.75 |

### Matrizes de Confusão

```
SVM + BoW:           SVM + Word2Vec:       BERT:
[[54 14]             [[53 15]              [[51 17]
 [22 19]]             [ 8 33]]              [ 6 35]]
```

## 💡 Conclusões

1. **Embeddings semânticos superam BoW:** Word2Vec e BERT tiveram desempenho ~12 pontos percentuais superior ao Bag of Words.

2. **BERT vs Word2Vec:** Desempenho similar em acurácia, porém BERT obteve melhor recall na classe Positivo (85% vs 80%).

3. **Balanceamento de classes:** O uso de `class_weight='balanced'` no SVM foi crucial para melhorar o recall da classe minoritária.

4. **Trade-off:** BERT requer mais recursos computacionais e tempo de treino, mas oferece melhor equilíbrio entre as classes.

## 🚀 Como Executar

### Requisitos
```bash
pip install pandas numpy scikit-learn gensim transformers torch matplotlib seaborn
```

### Executar o Notebook
1. Abra o notebook no Google Colab
2. Faça upload do arquivo `reviews_extraidas.csv`
3. Execute as células sequencialmente

## 📁 Estrutura do Projeto

```
├── README.md
├── analise_sentimentos.ipynb    # Notebook principal
├── reviews_extraidas.csv        # Dataset
└── results/
    ├── confusion_matrix_bow.png
    ├── confusion_matrix_word2vec.png
    └── confusion_matrix_bert.png
```

## 🔧 Tecnologias Utilizadas

- Python 3.10+
- Scikit-learn
- Gensim (Word2Vec)
- Transformers (BERT)
- PyTorch
- Pandas / NumPy
- Matplotlib / Seaborn

## 👤 Autor

[Pedro Fonseca de Andrade]

## 📄 Licença

Este projeto está sob a licença MIT.
