# Análise de Sentimentos em Tweets

Biblioteca para coleta, processamento e classificação de tweets com análise de sentimentos.

## Funcionalidades

- 🐦 Coleta de tweets através da API do Twitter
- 🧹 Pré-processamento de texto (limpeza, normalização)
- ✨ Extração de features (TF-IDF, Word2Vec, BERT)
- 🤖 Modelos de classificação (Naive Bayes, Regressão Logística)
- 📊 Avaliação de modelos e geração de relatórios
- 🔍 Análise de erros de classificação

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/tweet-sentiment-analysis.git
cd tweet-sentiment-analysis
Instale as dependências:

bash
pip install -r requirements.txt
Configure seu Bearer Token do Twitter no arquivo config.ini

Uso
1. Coleta de dados
python
from src.extraction import coletar_tweets

coletar_tweets(bearer_token, "#bolsonaro", "pt", 1000)
2. Pré-processamento
python
from src.preprocessing import preprocessar_csv

preprocessar_csv("data/tweets.csv")
3. Vetorização
python
from src.features import vetorizar_tfidf, vetorizar_word2vec

vetorizar_tfidf("data/preprocessado.csv")
vetorizar_word2vec("data/preprocessado.csv")
4. Treinamento e avaliação
python
from src.models import treinar_modelos

treinar_modelos("data/labels.csv")
Estrutura do Projeto
text
/data              # Dados brutos e processados
  /raw            # Tweets coletados
  /processed      # Dados pré-processados
  /features       # Features extraídas
/models           # Modelos treinados
/src              # Código fonte
  /preprocessing  # Limpeza e normalização
  /features       # Extração de features
  /models         # Modelos de ML
  /evaluation     # Métricas e visualização
/docs             # Relatórios e documentação
Requisitos
Python 3.8+

Tweepy (para coleta de tweets)

Scikit-learn (para modelos clássicos)

Transformers (para BERT)

Gensim (para Word2Vec)

Pandas, NLTK, Matplotlib

Contribuição
Contribuições são bem-vindas! Siga os passos:

Fork o projeto

Crie sua branch (git checkout -b feature/AmazingFeature)

Commit suas mudanças (git commit -m 'Add some amazing feature')

Push para a branch (git push origin feature/AmazingFeature)

Abra um Pull Request

Licença
Distribuído sob a licença MIT. Veja LICENSE para mais informações.

Contato
Seu Nome - @seu_twitter - seu.email@example.com

Link do Projeto: https://github.com/seu-usuario/tweet-sentiment-analysis
