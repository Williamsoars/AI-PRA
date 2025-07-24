# 📊 Análise de Sentimentos em Tweets

Biblioteca Python para coleta, processamento e classificação de tweets com análise de sentimentos em português.

## ✨ Funcionalidades

| Módulo         | Recursos                                                                 |
|----------------|--------------------------------------------------------------------------|
| **Coleta**     | Kasanova/sentiment140 • Filtros avançados • Coleta histórica             |
| **Pré-process**| Limpeza de texto • Normalização • Tratamento de emojis/gírias            |
| **Features**   | TF-IDF • Word2Vec • BERTimbau (BERT em português)   • Transformers       |
| **Modelos**    | Regressão Logística • LSTM                                               |
| **Avaliação**  | Métricas detalhadas • Matriz de confusão • Análise de erros              |

## 🚀 Começando

### Pré-requisitos
- Python 3.8+


### Instalação
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/tweet-sentiment-analysis.git
cd tweet-sentiment-analysis
```
# Crie e ative o ambiente virtual
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows
```
### Instale as dependências
```
pip install -r requirements.txt
```
### ⚡ Uso Rápido
```
1. Configuração
Crie config.ini na raiz do projeto:

ini
[twitter]
bearer_token = seu_token_aqui

[models]
default = bert
2. Pipeline Completo
python
from src.pipeline import SentimentAnalysisPipeline

pipeline = SentimentAnalysisPipeline(
    query="#eleições2023 lang:pt",
    max_tweets=5000,
    model_type="bert"
)

results = pipeline.run()
pipeline.generate_report()
🧩 Módulos Principais
Coleta de Tweets
python
from src.extraction import TweetCollector

collector = TweetCollector()
tweets = collector.search(
    query="#política -is:retweet",
    start_date="2023-01-01",
    end_date="2023-01-31",
    save_path="data/raw/politica_jan2023.csv"
)
Pré-processamento
python
from src.preprocessing import TextCleaner

cleaner = TextCleaner(
    remove_stopwords=True,
    stemmer="rslp"
)

cleaned_text = cleaner.clean_dataset("data/raw/tweets.csv")
📊 Resultados
Exemplo de saída:

text
✅ Análise concluída!
```
### 📈 Métricas:
```
- Acurácia: 0.87
- F1-score: 0.86
- Precision: 0.85  
- Recall: 0.88

🔍 Top erros:
1. Ironia/sarcasmo (23%)
2. Contexto político (18%)
3. Gírias regionais (15%)
🤝 Contribuição
Faça o fork do projeto

Crie sua feature branch (git checkout -b feature/nova-feature)

Commit suas mudanças (git commit -m 'Add feature')

Push para a branch (git push origin feature/nova-feature)

Abra um Pull Request
```
### 📄 Licença
```
Distribuído sob licença MIT. Veja LICENSE para detalhes.
```
### 📬 Contato
```
Equipe de Análise de Dados - analise@email.com

https://img.shields.io/twitter/follow/seu_perfil?style=social

text
```
### Recursos incluídos:
```
1. Tabela de funcionalidades organizada
2. Passos de instalação completos
3. Exemplos de código prontos para uso
4. Seção de resultados com exemplos visuais
5. Badge do Twitter (adicione o link real)
6. Ícones e emojis para melhor legibilidade
7. Estrutura modular clara

Para adicionar badges personalizadas (como build status, coverage etc.), você pode usar serviços como:
- Shields.io
- GitHub Actions badges
- Codecov/PyPI badges

Basta adicionar no topo do arquivo após o título principal.
