import io
import re 
from sqlalchemy import create_engine, text
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

class WordEmbeddingModel:
    
    def __init__(self, 
                 title: str, 
                 author: str,
                 corpus: List[str], ## list of paragraphs
                 sentances = False ## sequence unit = paragraph or sentance?
                ):
    
        self.title          = title
        self.author         = author 
        
    def tokenize_text(self): 
        
        
    
    

class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                        embedding_dim,
                                        name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size,
                                        embedding_dim)

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
            # target: (batch,)
            word_emb = self.target_embedding(target)
            # word_emb: (batch, embed)
            context_emb = self.context_embedding(context)
            # context_emb: (batch, context, embed)
            dots = tf.einsum('be,bce->bc', word_emb, context_emb)
            # dots: (batch, context)
            return dots
            