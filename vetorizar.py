import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def vetorizar_tfidf(caminho_csv: str, salvar_em: str = "dados/tfidf.pkl", max_features: int = 5000):
    df = pd.read_csv(caminho_csv)
    
    if "cleaned" not in df.columns:
        raise ValueError("Coluna 'cleaned' não encontrada. Execute o pré-processamento antes.")
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df["cleaned"])
    
    os.makedirs(os.path.dirname(salvar_em), exist_ok=True)
    with open(salvar_em, "wb") as f:
        pickle.dump((X, vectorizer), f)
    
    print(f"[✓] TF-IDF salvo em {salvar_em}. Shape: {X.shape}")
