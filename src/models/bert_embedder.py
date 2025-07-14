import torch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import pandas as pd

class BERTEmbedder:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.model.eval()
        
    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def vectorize_bert(
    input_path: Path,
    output_path: Path,
    text_column: str = "text",
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 32
):
    try:
        df = pd.read_csv(input_path)
        embedder = BERTEmbedder(model_name)
        texts = df[text_column].astype(str).tolist()
        vectors = []
        
        for i in tqdm(range(0, len(texts), desc="Processing texts"):
            batch = texts[i:i+batch_size]
            vectors.extend([embedder.embed_text(text) for text in batch])
        
        X = np.stack(vectors)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(X, f)
        
        logging.info(f"BERT vectors saved to {output_path}. Shape: {X.shape}")
        return X
        
    except Exception as e:
        logging.error(f"Error in BERT vectorization: {str(e)}")
        raise
