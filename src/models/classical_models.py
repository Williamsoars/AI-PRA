from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import pickle
import logging
import pandas as pd
from src.utils.evaluation import evaluate_model

def load_labels(data_path: Path, label_column: str = "label"):
    try:
        df = pd.read_csv(data_path)
        return df[label_column].values
    except Exception as e:
        logging.error(f"Erro ao carregar labels: {str(e)}")
        raise

def train_models(
    data_path: Path,
    label_column: str = "label",
    test_size: float = 0.2,
    random_state: int = 42
):
    results = []
    
    try:
        y = load_labels(data_path, label_column)
        
        # TF-IDF + Naive Bayes
        with open(Path("models/tfidf.pkl"), "rb") as f:
            X_tfidf, _ = pickle.load(f)
            
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=test_size, random_state=random_state
        )
        
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        nb_results = evaluate_model(nb, X_test, y_test, "Naive Bayes (TF-IDF)")
        results.append(("Naive Bayes (TF-IDF)", *nb_results))
        
        # TF-IDF + Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        lr_results = evaluate_model(lr, X_test, y_test, "Logistic Regression (TF-IDF)")
        results.append(("Logistic Regression (TF-IDF)", *lr_results))
        
        return results
        
    except Exception as e:
        logging.error(f"Erro no treinamento de modelos: {str(e)}")
        raise
