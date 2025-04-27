import re 
from sqlalchemy import create_engine, text
import tqdm
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine, text
from typing import List
from pathlib import Path
import json
import nltk 
nltk.download('punkt_tab')

import numpy as np
from collections import Counter

import tensorflow as tf
import keras
from tensorflow.keras import layers
from nltk.corpus import stopwords
nltk.download('stopwords')
import os

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

class WordEmbeddingModel:
    
    def __init__(self, 
                 title: str, 
                 author: str,
                 corpus: List[str], ## list of paragraphs
                 config_path = 'shared/config.json',
                 sentences = False ## sequence unit = paragraph or sentance?
                ):

        # Get the path of the current script
        script_dir = Path(__file__).resolve().parent

        self.db_user        = os.environ.get("DB_USER")
        self.db_password    = os.environ.get("DB_PASSWORD")
        self.db_host        = os.environ.get("DB_HOST")
        self.db_port        = os.environ.get("DB_PORT")
        self.db_name        = os.environ.get("DB_NAME")
        self.title          = title
        self.author         = author 
        self.corpus         = corpus
        self.db_url = f'postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}'
        
    def get_sentences(self):
        
        """
        Process input paragraphs. Returns a list of token lists for each sentance.
        
        """
        
        sentence_all = []
        sentence_pk = []
        token_set = set()
        max_len = 0
        pk = None
        stop_words = set(stopwords.words('english'))

        for p in self.corpus: 
            
            sentences = self.split_paragraph_into_sentences(p)
            
            ## tokenize each sentance and store
            for s in sentences:
                
                ## if first sentence, get pk, will iterate on each sentence
                if s == sentences[0] and not pk:
                    result = self.execute_statement(f"""
                        SELECT pk from sentences 
                        WHERE sentence_text = :sentence
                        ORDER BY pk ASC 
                    """, {"sentence": s.strip()}).fetchone()
                    
                    pk = result[0]
                
                s = re.sub(r'[^\w\s]', '', s).lower().strip()
                
                if s:
                
                    tokens = word_tokenize(s)
                    
                    ## drop stop words
                    filtered_sentence = [word for word in tokens if word not in stop_words]

                    # Join the filtered words back into a sentence
                    s_no_stop = " ".join(filtered_sentence)
                
                    sentence_all.append(s_no_stop)
                    sentence_pk.append(pk)
                    
                    pk += 1
                    for token in tokens:
                        token_set.add(token)
                    
                    if len(tokens) > max_len:
                        max_len = len(tokens)

        return sentence_all, sentence_pk, token_set, max_len
    
    
    def vectorize(self, sentences, vocab_size, max_sentence):
        
        ### first vectorization: integer vocab indices for each word in a sentence
        vectorize_layer_int = keras.layers.TextVectorization(
            max_tokens = vocab_size,
            output_mode = 'int',
            output_sequence_length = max_sentence
        )
        
        dset_len = len(sentences)
        
        text_ds = np.array(sentences)
        
        if dset_len > 1024: 
            vectorize_layer_int.adapt(text_ds.batch(1024))
        else: 
            vectorize_layer_int.adapt(text_ds)
           
        vectorized_sentences = vectorize_layer_int(text_ds)       
        vec_vocab = vectorize_layer_int.get_vocabulary()
        
        ## second: get tf-idf weights
        tf_idf_layer =  keras.layers.TextVectorization(
            standardize = None,
            max_tokens = vocab_size,
            output_mode = 'tf_idf'
        )
        
        if dset_len > 1024: 
            tf_idf_layer.adapt(text_ds.batch(1024))
        else: 
            tf_idf_layer.adapt(text_ds)
            
        tf_idf_sentences  = tf_idf_layer(text_ds)
        
        
        return vec_vocab, vectorized_sentences, tf_idf_sentences   
    
    def generate_training_data(self, sentences, window_size, num_ns, vocab_size, power_law_probs_tensor, vocab_tensor, seed):
        
        targets = []
        contexts = []
        labels = []
        
        ## build sampling table to downsample high frequency tokens
        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

        ## loop over sentences
        for s in tqdm.tqdm(sentences):

            # generate positive skip grams
            positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                s,
                vocabulary_size=vocab_size,
                sampling_table=sampling_table,
                window_size=window_size,
                negative_samples=0)
            
            # Iterate over each positive skip-gram pair to produce training examples
            # with a positive context word and negative samples.
            for target_word, context_word in positive_skip_grams:
                
                context_class = tf.constant(context_word.numpy(), dtype = "int64")
                context_class = tf.expand_dims([context_class], axis=0)  # Shape: [1]
                
                # Create a tensor for the word distribution and sample negative words
                logits = tf.math.log(power_law_probs_tensor)  # Log of the probabilities
                
                negative_sampling_candidates =  tf.zeros([num_ns], dtype = 'int64')
                while tf.reduce_any(tf.equal(negative_sampling_candidates, context_word)) or tf.reduce_any(tf.equal(negative_sampling_candidates, 0)):
                    
                    sampled_negative_indices = tf.random.categorical(
                        logits[None, :], num_ns, dtype=tf.int64, seed=seed)  # Sample indices
                    
                    negative_sampling_candidates = tf.gather(vocab_tensor, sampled_negative_indices)
                
                # Build context and label vectors (for one target word)
                context = tf.concat([tf.squeeze(context_class,1), tf.squeeze(negative_sampling_candidates, 0)], axis = 0)
                label = tf.constant([1] + [0]*num_ns, dtype="int64")

                # Append each element from the training example to global lists.
                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

        return targets, contexts, labels
        
    
    def compute_power_law_probabilities(self, vectorized_sentences, alpha=0.75):
        # Flatten the list of sentences to get all words
        all_words = vectorized_sentences.flatten() if isinstance(vectorized_sentences, np.ndarray) else tf.reshape(vectorized_sentences, [-1]).numpy()
        
        # Count the frequency of each word
        word_freqs = Counter(all_words)
        
        # Apply power-law transformation to frequencies
        power_law_probs = {word: (freq ** alpha) for word, freq in word_freqs.items()}
        
        # Normalize the power-law probabilities so they sum to 1
        total_power_law = sum(power_law_probs.values())
        normalized_power_law_probs = {word: prob / total_power_law for word, prob in power_law_probs.items()}
        
        vocab = list(normalized_power_law_probs.keys())
        probs = list(normalized_power_law_probs.values())

        power_law_probs_tensor = tf.convert_to_tensor(probs, dtype=tf.float32)
        vocab_tensor = tf.convert_to_tensor(vocab, dtype=tf.int64)  # Using token indices
    
        return vocab_tensor, power_law_probs_tensor
    
    
    def build_and_train(self, vocab_size, embedding_dim, num_ns, targets, contexts, labels, batch_size, epochs): 
        
        word2vec = Word2Vec(vocab_size, embedding_dim)
        loss_fn = NegativeSamplingLoss(num_ns=num_ns)
        word2vec.compile(optimizer='adam',
                 loss=loss_fn,
                 metrics=['accuracy'])
        
        ## log of training
        script_dir = Path(__file__).resolve().parent.parent
        log_dir = script_dir / 'logs'
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        
        ## prep data
        BATCH_SIZE = batch_size
        BUFFER_SIZE = 10000
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
        
        ## fit model
        word2vec.fit(dataset, epochs=epochs, callbacks=[tensorboard_callback])
        
        return word2vec
        

    
    def execute_statement(self, qry, dict):

        try: 
            
            engine = create_engine(self.db_url, isolation_level="AUTOCOMMIT")
            
            with engine.connect() as connection:
                
                try:
                    # execute
                    print(qry)
                    response = connection.execute(text(qry), dict)

                    print("Transaction committed successfully!")
                    return response
                    
                except Exception as e:
                    print("An error occurred when executing:", e)
                    return False 
        
        except Exception as e:
            
            print(f"Error occurred when connecting: {e}")
            return False 
    
    def split_paragraph_into_sentences(self, paragraph):
        ## identify quoted text
        quoted_text = re.findall(r'["\']["\']([^\']+[^\']*)["\']["\']', paragraph)  # Extract quoted text
        
        ## replace quoted text with placeholders so they aren't split
        placeholder = "<QUOTE>"
        for idx, quote in enumerate(quoted_text):
            paragraph = paragraph.replace(f"'{quote}'", f'{placeholder}{idx}{placeholder}')

        # split on typical endings
        sentences = re.split(r'(?<=\w[.!?])(?<!(?:Mr|Dr|Jr|Sr|vs|Ms)\.)(?<!(?:Prof)\.)(?<!(?:Inc|Mrs|etc)\.)\s+(?!(?:Mr|Dr|Prof|Inc|Jr|Sr|vs|Mrs|Ms|etc)\.)(?=(?!<QUOTE>\d+<QUOTE>))', paragraph.strip())

        # replace placeholders with original text
        for idx, quote in enumerate(quoted_text):
            sentences = [s.replace(f'{placeholder}{idx}{placeholder}', f"'{quote}'") for s in sentences]

        return sentences
                

    

class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        
        ## hidden layer
        self.target_embedding = layers.Embedding(vocab_size,
                                        embedding_dim,
                                        name="w2v_embedding")
        ## context layer
        self.context_embedding = layers.Embedding(vocab_size,
                                        embedding_dim)

    ## compute similarity score for forward passes 
    ## similarity = inner product of target and context
    ## similarity scores feed into sigmoid, mapping similarity to [0,1]
    def call(self, pair):
        target, context = pair
        ## dimensions
        ## target: (batch size, dummy?) 
        ## context: (batch size, context)
        if len(target.shape) == 2:
            
            ## reduce dimension if needed
            target = tf.squeeze(target, axis=1)
        
        ## word_emb: (batch, embed)
        word_emb = self.target_embedding(target)
        
        ## context_emb: (batch, context, embed)
        context_emb = self.context_embedding(context)
        
        ## dot products: (batch, context)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        
        return dots
    
class NegativeSamplingLoss(tf.keras.losses.Loss):
    def __init__(self, num_ns, **kwargs):
        """
        Custom loss function for Word2Vec with negative sampling.
        
        Args:
            num_ns (int): Number of negative samples per positive sample.
        """
        super(NegativeSamplingLoss, self).__init__(**kwargs)
        self.num_ns = num_ns

    def call(self, y_true, y_pred):
        """
        Compute the negative sampling loss.
        
        Args:
            y_true : labeled value
            y_pred (tf.Tensor): Predicted logits from the model. Shape: (batch_size, 1 + num_ns).
        
        Returns:
            tf.Tensor: Loss value for the batch.
        """
        ## get predicted logits
        pos_logits = y_pred[:, 0]  
        neg_logits = y_pred[:, 1:]  

        ## log(p(y=1 | word, true context)) -- log-liklihood of labelling true given true context
        ## pos_loss: (batch_size,) 
        pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(pos_logits), logits=pos_logits
        )

        ## log(p(y = 0 | word, noise context)) -- log-liklihood of labelling noise given noise context
        ## neg_loss : (batch_size, num_ns)
        neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(neg_logits), logits=neg_logits
        )

        total_loss = tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)
        return total_loss
    
            