import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def carregar_rotulos(caminho_csv: str, coluna_label="label"):
    import pandas as pd
    df = pd.read_csv(caminho_csv)
    return df[coluna_label].values

def avaliar_modelo(modelo, X_test, y_test, nome="Modelo"):
    y_pred = modelo.predict(X_test)
    if hasattr(modelo, "predict_proba"):
        y_proba = modelo.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    print(f"\n== Avaliação: {nome} ==")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

def treinar_modelos(caminho_csv_labels: str):
    y = carregar_rotulos(caminho_csv_labels)

    # ==== TF-IDF + Naive Bayes
    with open("dados/tfidf.pkl", "rb") as f:
        X_tfidf, _ = pickle.load(f)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    avaliar_modelo(nb, X_test, y_test, "Naive Bayes (TF-IDF)")

    # ==== TF-IDF + Logistic Regression
    lr1 = LogisticRegression(max_iter=1000)
    lr1.fit(X_train, y_train)
    avaliar_modelo(lr1, X_test, y_test, "Logistic Regression (TF-IDF)")

    # ==== Word2Vec + Logistic Regression
    with open("dados/word2vec.pkl", "rb") as f:
        X_w2v, _ = pickle.load(f)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_w2v, y, test_size=0.2, random_state=42)

    lr2 = LogisticRegression(max_iter=1000)
    lr2.fit(X_train2, y_train2)
    avaliar_modelo(lr2, X_test2, y_test2, "Logistic Regression (Word2Vec)")
