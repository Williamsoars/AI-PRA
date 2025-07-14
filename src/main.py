from src.extraction.twitter_extractor import collect_tweets
from src.preprocessing.text_processor import preprocess_csv
from src.features.tfidf_vectorizer import vectorize_tfidf
from src.models.classical_models import train_models
from src.utils.report import generate_report
import configparser
from pathlib import Path

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # Coleta de tweets
    hashtag = input("Digite a hashtag ou termo (ex: #bolsonaro): ")
    language = input("Digite o código do idioma (ex: pt, en, es): ").lower()
    limit = input("Quantos tweets você quer coletar? [padrão: 1000]: ")
    limit = int(limit) if limit.isdigit() else 1000

    raw_data_path = Path(config['PATHS']['data_dir']) / 'raw_tweets.csv'
    collect_tweets(
        bearer_token=config['TWITTER']['bearer_token'],
        hashtag=hashtag,
        language=language,
        limit=limit,
        output_path=raw_data_path
    )

    # Pré-processamento
    processed_path = Path(config['PATHS']['data_dir']) / 'processed.csv'
    preprocess_csv(
        input_path=raw_data_path,
        output_path=processed_path,
        text_column='text'
    )

    # Vetorização
    tfidf_path = Path(config['PATHS']['models_dir']) / 'tfidf.pkl'
    vectorize_tfidf(
        input_path=processed_path,
        output_path=tfidf_path,
        max_features=int(config['MODELS']['tfidf_max_features'])
    )

    # Treinamento
    results = train_models(
        data_path=processed_path,
        label_column='label'  # Assumindo que existe uma coluna 'label'
    )

    # Relatório
    report_path = Path(config['PATHS']['reports_dir']) / 'report.md'
    generate_report(results, report_path)

if __name__ == "__main__":
    main()
