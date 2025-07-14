from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pathlib import Path
import logging
import pandas as pd

def vectorize_tfidf(
    input_path: Path,
    output_path: Path,
    text_column: str = "cleaned",
    max_features: int = 5000
):
    try:
        df = pd.read_csv(input_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Coluna '{text_column}' não encontrada.")
        
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(df[text_column])
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump((X, vectorizer), f)
        
        logging.info(f"TF-IDF salvo em {output_path}. Shape: {X.shape}")
        return X, vectorizer
    except Exception as e:
        logging.error(f"Erro na vetorização TF-IDF: {str(e)}")
        raise
