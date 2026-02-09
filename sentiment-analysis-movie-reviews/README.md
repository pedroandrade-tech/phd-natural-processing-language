# 🎬 Análise de Sentimentos em Reviews de Filmes

Projeto de Processamento de Linguagem Natural (PLN) para classificação de sentimentos em reviews do filme **"Babygirl" (2024)** utilizando diferentes técnicas de Machine Learning e Deep Learning.

## 📋 Objetivo

Implementar e comparar três abordagens para classificação binária de sentimentos:
- **SVM + Bag of Words (BoW)**
- **SVM + Word2Vec Embeddings**
- **BERT (Fine-tuning)**

## 📊 Dataset

- **Fonte:** Reviews extraídas do IMDB para o filme "Babygirl" (2024)
- **Total de reviews:** 544
- **Divisão:** 80% treino (435) | 20% teste (109)
- **Classificação:** Binária (Negativo/Positivo)

| Classe | Critério | Quantidade |
|--------|----------|------------|
| Negativo (0) | Notas 1-5 | 338 |
| Positivo (1) | Notas 6-10 | 206 |

## 🛠️ Metodologia

### 1. SVM + Bag of Words
- Vetorização com `CountVectorizer`
- N-gramas: unigramas e bigramas
- Max features: 5000
- Kernel: Linear

### 2. SVM + Word2Vec
- Embeddings pré-treinados: `word2vec-google-news-300` (300 dimensões)
- Representação: Média dos vetores das palavras
- Kernel: RBF
- Balanceamento: `class_weight='balanced'`

### 3. BERT
- Modelo: `bert-base-uncased`
- Fine-tuning: 3 épocas
- Max length: 256 tokens
- Batch size: 8

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

### Teste com Novas Reviews

| Review | BoW | Word2Vec | BERT |
|--------|-----|----------|------|
| "This movie was absolutely amazing!" | ❌ Neg | ❌ Neg | ✅ Pos (70%) |
| "Terrible film. Waste of time." | ✅ Neg | ✅ Neg | ✅ Neg (95%) |
| "It was okay, nothing special." | Neg | Neg | Pos (65%) |

## 💡 Conclusões

1. **Embeddings semânticos superam BoW:** Word2Vec e BERT tiveram desempenho ~12 pontos percentuais superior ao Bag of Words.

2. **BERT vs Word2Vec:** Desempenho similar em acurácia (78.90%), porém BERT obteve melhor recall na classe Positivo (85% vs 80%) e generalizou melhor para novas reviews.

3. **Balanceamento de classes:** O uso de `class_weight='balanced'` no SVM foi crucial para melhorar o recall da classe minoritária (Positivo).

4. **Limitações do BoW e Word2Vec:** Ambos tiveram dificuldade em classificar corretamente reviews positivas curtas fora do domínio de treino. BERT mostrou melhor capacidade de generalização.

5. **Trade-off:** BERT requer mais recursos computacionais (~1h de treino no Colab) mas oferece melhor equilíbrio entre as classes e generalização.

## 🚀 Como Executar

### Requisitos
```bash
pip install -r requirements.txt
```

### No Google Colab
1. Faça upload dos arquivos da pasta `data/`
2. Abra o notebook desejado
3. Execute as células sequencialmente

### Localmente
1. Clone o repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Ajuste os caminhos dos arquivos nos notebooks

## 📁 Estrutura do Projeto

```
sentiment-analysis-movie-reviews/
├── README.md
├── requirements.txt
├── data/
│   ├── reviews_extraidas.csv          # Dataset processado
│   └── sentiment-analysis-nlp-dataset.docx  # Dataset original
└── notebooks/
    ├── Projeto_01_BOW+SVM_2_classes.ipynb      # SVM + Bag of Words
    ├── Projeto_01_Word2Vec+SVM_2_classes.ipynb # SVM + Word2Vec
    └── Projeto_01_Bert.ipynb                   # BERT Fine-tuning
```

## 🔧 Tecnologias Utilizadas

- Python 3.10+
- Scikit-learn
- Gensim (Word2Vec)
- Transformers / HuggingFace (BERT)
- PyTorch
- Pandas / NumPy
- Matplotlib / Seaborn

## 👤 Autor

**Pedro Fonseca de Andrade**

## 📄 Licença

Projeto desenvolvido para fins acadêmicos.