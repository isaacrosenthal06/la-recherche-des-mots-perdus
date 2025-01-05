from insert_clean.utils import InsertNewBook
from embed.word2vec import WordEmbeddingModel, Word2Vec, NegativeSamplingLoss
import numpy as np
import io
from pathlib import Path
import pandas as pd


def embed_handler(author: str, title: str):
    
    insert_class = InsertNewBook(config_path='db/config.json',
                                 title = title,
                                 author = author)
    
    