import pandas as pd
from pathlib import Path
import logging

def collect_tweets(
    bearer_token: str,
    hashtag: str,
    language: str = "pt",
    limit: int = 1000,
    output_path: Path = None
):
    """
    Simula a coleta de tweets usando o dataset Sentiment140.
    Filtra por hashtag e retorna DataFrame com coluna 'text'.
    """
    try:
        # Caminho para o dataset local (ajuste conforme necessário)
        dataset_path = Path("datasets/sentiment140.csv")

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset não encontrado em {dataset_path}")

        # Dataset original não tem cabeçalhos. Definimos manualmente.
        cols = ["target", "ids", "date", "flag", "user", "text"]
        df = pd.read_csv(dataset_path, encoding="latin-1", names=cols)

        # Filtragem por hashtag (sem distinção de idioma)
        df_filtered = df[df["text"].str.contains(hashtag, case=False, na=False)]

        # Pega apenas a coluna "text" e aplica o limite
        df_result = df_filtered[["text"]].head(limit)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_result.to_csv(output_path, index=False)
            logging.info(f"{len(df_result)} tweets salvos em {output_path}")

        return df_result

    except Exception as e:
        logging.error(f"Erro na leitura do Sentiment140: {str(e)}")
        raise


