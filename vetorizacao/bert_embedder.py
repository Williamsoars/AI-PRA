import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import pickle
from tqdm import tqdm

def vetorizar_bert(caminho_csv: str, salvar_em: str = "dados/bert.pkl", modelo_name: str = "distilbert-base-uncased", coluna="text"):
    df = pd.read_csv(caminho_csv)
    tokenizer = DistilBertTokenizer.from_pretrained(modelo_name)
    model = DistilBertModel.from_pretrained(modelo_name)
    model.eval()

    textos = df[coluna].astype(str).tolist()
    vetores = []

    for texto in tqdm(textos, desc="BERT Embedding"):
        inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        vetor = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Média dos tokens
        vetores.append(vetor)

    X = np.stack(vetores)

    with open(salvar_em, "wb") as f:
        pickle.dump(X, f)

    print(f"[✓] Vetores BERT salvos em {salvar_em}. Shape: {X.shape}")
