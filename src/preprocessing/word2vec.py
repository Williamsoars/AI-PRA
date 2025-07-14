import gensim
import numpy as np
import pickle
from pathlib import Path
import logging
import pandas as pd
from typing import List

class Word2VecEmbedder:
    def __init__(self, vector_size=100, window=5, min_count=2, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        
    def train(self, sentences: List[List[str]]):
        self.model = gensim.models.Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )
        return self.model
        
    def get_sentence_vector(self, sentence: str) -> np.ndarray:
        words = sentence.split()
        vectors = [self.model.wv[word] for word in words if word in self.model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.vector_size)

def vectorize_word2vec(
    input_path: Path,
    output_path: Path,
    text_column: str = "cleaned",
    vector_size: int = 100
):
    try:
        df = pd.read_csv(input_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        sentences = [text.split() for text in df[text_column]]
        embedder = Word2VecEmbedder(vector_size=vector_size)
        embedder.train(sentences)
        
        X = np.array([embedder.get_sentence_vector(text) for text in df[text_column]])
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump((X, embedder.model), f)
        
        logging.info(f"Word2Vec vectors saved to {output_path}. Shape: {X.shape}")
        return X, embedder.model
        
    except Exception as e:
        logging.error(f"Error in Word2Vec vectorization: {str(e)}")
        raise
