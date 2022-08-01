import fasttext
import fasttext.util
fasttext.FastText.eprint = lambda x: None

from sentence_transformers import SentenceTransformer

import numpy as np


class Embedder:
    def __init__(self, name="fasttext", use_tfidf=False):
        self.name = name
        if name == "fasttext_en":
            self.model = fasttext.load_model('cc.en.300.bin')
            self.emb_size = 300
        elif name == "fasttext_id":
            self.model = fasttext.load_model('cc.id.300.bin')
            self.emb_size = 300
        elif name == "sbert":
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.emb_size = 384
            # self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
            # self.emb_size = 512
        elif name == "everything":
            self.model = [
                fasttext.load_model('cc.en.300.bin'),
                fasttext.load_model('cc.id.300.bin'),
                SentenceTransformer('all-MiniLM-L6-v2')
            ]
            self.emb_size = 300 + 300 + 384
        else:
            raise ValueError(f"{name} embedder is not supported")

        if use_tfidf:
            self.emb_size += 300*2

    def get_sentence_vector(self, text):
        if self.name in ["fasttext_en", "fasttext_id"]:
            return self.model.get_sentence_vector(text)
        elif self.name == "sbert":
            return self.model.encode(text)
        elif self.name == "everything":
            ft_en, ft_id, sbert = self.model
            return np.concatenate([
                ft_en.get_sentence_vector(text),
                ft_id.get_sentence_vector(text),
                sbert.encode(text)
            ])
        else:
            raise ValueError(f"{self.name} embedder is not supported")
