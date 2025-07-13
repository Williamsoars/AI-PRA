import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stopwords_pt = set(stopwords.words("portuguese"))

def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)  # URLs
    texto = re.sub(r"@\w+", "", texto)     # Menções
    texto = re.sub(r"#\w+", "", texto)     # Hashtags
    texto = re.sub(r"[^\w\s]", "", texto)  # Pontuação
    texto = re.sub(r"\d+", "", texto)      # Números
    texto = re.sub(r"\s+", " ", texto)     # Espaços extras
    tokens = texto.strip().split()
    tokens = [t for t in tokens if t not in stopwords_pt]
    return " ".join(tokens)

def preprocessar_csv(caminho_csv: str, coluna: str = "text", salvar_em: str = "dados/preprocessado.csv"):
    df = pd.read_csv(caminho_csv)
    df["cleaned"] = df[coluna].astype(str).apply(limpar_texto)
    df.to_csv(salvar_em, index=False)
    print(f"[✓] Pré-processamento concluído. Salvo em: {salvar_em}")
