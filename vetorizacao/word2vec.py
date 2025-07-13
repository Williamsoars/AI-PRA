import pandas as pd
import gensim
import os
import numpy as np
import pickle

def treinar_word2vec(df, coluna="cleaned", vector_size=100, min_count=2):
    frases = [linha.split() for linha in df[coluna]]
    modelo = gensim.models.Word2Vec(sentences=frases, vector_size=vector_size, window=5, min_count=min_count, workers=4)
    return modelo

def vetor_medio(frase, modelo, vector_size):
    palavras = frase.split()
    vetores = [modelo.wv[p] for p in palavras if p in modelo.wv]
    if vetores:
        return np.mean(vetores, axis=0)
    else:
        return np.zeros(vector_size)

def vetorizar_word2vec(caminho_csv: str, salvar_em: str = "dados/word2vec.pkl", vector_size: int = 100):
    df = pd.read_csv(caminho_csv)
    
    if "cleaned" not in df.columns:
        raise ValueError("Coluna 'cleaned' não encontrada.")
    
    modelo = treinar_word2vec(df, vector_size=vector_size)
    X = np.array([vetor_medio(txt, modelo, vector_size) for txt in df["cleaned"]])
    
    with open(salvar_em, "wb") as f:
        pickle.dump((X, modelo), f)
    
    print(f"[✓] Word2Vec salvo em {salvar_em}. Shape: {X.shape}")
