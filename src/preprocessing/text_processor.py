import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from pathlib import Path
import logging

nltk.download("stopwords")

class TextProcessor:
    def __init__(self, language='portuguese'):
        self.stopwords = set(stopwords.words(language))
        
    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+", "", text)  # URLs
        text = re.sub(r"@\w+", "", text)     # Menções
        text = re.sub(r"#\w+", "", text)     # Hashtags
        text = re.sub(r"[^\w\s]", "", text)  # Pontuação
        text = re.sub(r"\d+", "", text)      # Números
        text = re.sub(r"\s+", " ", text)     # Espaços extras
        tokens = text.strip().split()
        tokens = [t for t in tokens if t not in self.stopwords]
        return " ".join(tokens)

def preprocess_csv(
    input_path: Path,
    output_path: Path,
    text_column: str = "text",
    language: str = "portuguese"
):
    try:
        df = pd.read_csv(input_path)
        processor = TextProcessor(language)
        df["cleaned"] = df[text_column].astype(str).apply(processor.clean_text)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Pré-processamento concluído. Salvo em: {output_path}")
    except Exception as e:
        logging.error(f"Erro no pré-processamento: {str(e)}")
        raise
