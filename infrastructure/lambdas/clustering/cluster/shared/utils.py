import re 
from sqlalchemy import create_engine, text
from typing import List
from pathlib import Path
import json
import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import BayesianGaussianMixture
from fuzzywuzzy import process
from nltk.corpus import wordnet
from collections import Counter
import spacy 
import nltk 
from nltk.corpus import stopwords
import pickle
import boto3
import os
nltk.download('stopwords')
nltk.download('wordnet')

s3 = boto3.client('s3', 
    endpoint_url='http://localhost:4566', 
    region_name='us-east-1',  
    aws_access_key_id='test',  
    aws_secret_access_key='test') 

def upload_s3(local_file, bucket_name, file_key):
    
    try:
        s3.upload_file(local_file, bucket_name, file_key)

        return True
        
    except Exception as e:
        raise f'An error occurred in uploading to s3: {e}'

## update to read file types appropriately
def read_s3(bucket_name, file_key, file_type):
    
    try:
        response = s3.get_object(Bucket = bucket_name, Key = file_key)
        
        body = response['Body'].read().decode('utf-8')
        
        return body
    
    except Exception as e:
        raise f'An error occurred in reading from s3: {e}'
    
def confirm_cluster(title, author):
        
    statement = """
        SELECT title, author, FROM Clusters
        WHERE (title = :title AND author = :author)
    """
        
    val_dict = {
        'title': title,
        'author': author
    }
           
    response, response_status = self.execute_statement(statement, dict = val_dict)
        
    response_vals = response.fetchone()
        
    if response_status and response_vals:
            
        print(f'Book has been clustered: {response_vals}')
            
        return response_vals, True
        
    else: 
        print(f'Book has not been clustered')
            
        return None, False


def execute_statement(qry, dict):
    

    db_user        = os.environ.get("DB_USER")
    db_password    = os.environ.get("DB_PASSWORD")
    db_host        = os.environ.get("DB_HOST")
    db_port        = os.environ.get("DB_PORT")
    db_name        = os.environ.get("DB_NAME")
    db_url = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    
    try: 
            
        engine = create_engine(db_url, isolation_level="AUTOCOMMIT")
            
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
    
    
class Cluster:
    
    def __init__(self,
                 title: str, 
                 author: str,
                 vocabulary: str,
                 tf_idf_wts: str,
                 vectors: str,
                 embedding_dim: int):
        
        script_dir = Path(__file__).resolve().parent

        self.db_user        = os.environ.get("DB_USER")
        self.db_password    = os.environ.get("DB_PASSWORD")
        self.db_host        = os.environ.get("DB_HOST")
        self.db_port        = os.environ.get("DB_PORT")
        self.db_name        = os.environ.get("DB_NAME")
        self.title          = title
        self.author         = author 
        self.embedding_dim  = embedding_dim
        
        
        self.db_url = f'postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}'
        
        vocab_bucket = "vocab"
        tf_idf_bucket = "tf-idf"
        vector_bucket = "vectors"
        
        self.word_embeddings = read_s3(bucket_name=vector_bucket, file_key=vectors, file_type='tsv')
        self.tf_idf          = read_s3(bucket_name=tf_idf_bucket, file_key=tf_idf_wts, file_type='csv')       
        self.vocab           = read_s3(bucket_name=vocab_bucket, file_key=vocabulary, file_type = 'tsv')
        

    def embed_sentences(self):
        
        ## convert to arrays 
        self.tf_idf['token_indices']    = self.tf_idf['token_indices'].apply(literal_eval).apply(np.array) 
        self.tf_idf['weights']          = self.tf_idf['weights'].apply(literal_eval).apply(np.array) 
        self.tf_idf['sentence_weight']  = np.nan
        self.tf_idf['sentence_vec']     = [np.empty((1, self.embedding_dim)) for _ in range(0, len(self.tf_idf))]
        
        for index, row in self.tf_idf.iterrows():
            
            words = row['token_indices']
            wts = row['weights']
            
            total_wt = 0
            
            ## print first five tokens
            wtd_vectors = np.empty((len(words), self.embedding_dim))
            for i, ele in enumerate(words):
                
                if index == 0 and i < 5:
                    print(f"Applying weights for word {self.vocab[ele- 1]}")
                    
                vector = self.word_embeddings[ele- 1]
                wt = wts[i]
                
                wtd_vec = wt * vector
                    
                wtd_vectors[i] = wtd_vec
                
                total_wt += wt 

            if total_wt != 0:
                weighted_avg = np.sum(wtd_vectors, axis = 0, dtype=np.float64) / total_wt
            else: 
                if len(wtd_vectors) > 1:
                    raise ValueError("TF-idf weights of zero for sentences with multiple tokens")
                weighted_avg = wtd_vec
            
            
            self.tf_idf.at[index, 'sentence_vec']     = weighted_avg
            self.tf_idf.at[index,'sentence_weight']   = total_wt
            
        return self.tf_idf 
    
    def plot_embeddings(self, embeddings, labels = None):
        
        ## dimensionality reduction 
        pca = PCA(n_components = 3)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # plot 3d
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c = labels, cmap='viridis')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        
        fig_path = f'/tmp/{self.title}_{self.author}_cluster_plot.png'

        plt.savefig(fig_path) 
        
        upload_s3(local_file=fig_path, bucket_name='cluster-plots', file_key = f'{self.title}_{self.author}_cluster_plot.png')
        plt.close()
        
    def fit_clustering(self, embeddings, max_clusters, alpha):
        
        dpgmm = BayesianGaussianMixture(
            n_components=max_clusters,               
            covariance_type='full',        
            weight_concentration_prior=alpha,  ## skew to larger number of clusters  
            random_state=42
        )

        ## fit to sentence embeddings
        dpgmm.fit(embeddings)
        
        return(dpgmm)
    
    def save_model(self, model):
                
        file_path = f'/tmp/DPGMM_{self.title}_{self.author}'
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            
            upload_s3(local_file = file_path, bucket_name = 'cluster_models', file_key =  f'DPGMM_{self.title}_{self.author}')
            
        except Exception as e:
            print(f"Error saving model: {e}")
        
    
    def get_gaussian_probs(self, sentence_embeddings, threshold, model):
        
        dpgmm = model
        
        ## get probability of new text belonging to a cluster for each gaussian cluster 
        probs = dpgmm.predict_proba(sentence_embeddings)
        
        ## identify clusters meeting a specific probability threshold
        above_threshold = [np.where(row > threshold)[0] for row in probs]
        
        return above_threshold , probs

class Predict:
    
    def __init__(self,
                 new_text: str,
                 title: str, 
                 author: str,
                 vocab: str,
                 vectors: str,
                 embeddings_path: str,
                 cluster_model_path: str
                ):
        
        script_dir = Path(__file__).resolve().parent

        self.db_user        = os.environ.get("DB_USER")
        self.db_password    = os.environ.get("DB_PASSWORD")
        self.db_host        = os.environ.get("DB_HOST")
        self.db_port        = os.environ.get("DB_PORT")
        self.db_name        = os.environ.get("DB_NAME")
        self.title          = title
        self.author         = author

        self.new_text       = new_text
        
        vocab_bucket      = "vocab"
        vector_bucket     = "vectors"
        embeddings_bucket = 'clustered-sentences'
        
        self.vectors = read_s3(bucket_name=vector_bucket, file_key=vectors, file_type='tsv')       
        self.vocab   = read_s3(bucket_name=vocab_bucket, file_key=vocab, file_type = 'tsv')
        ## convert columns from string to np arrays
        self.embeddings_df = read_s3(bucket_name=embeddings_bucket, file_key=embeddings_path, file_type = 'csv')
        self.embeddings_df['token_indices']    = self.embeddings_df['token_indices'].str.replace(r"\s+", ",", regex=True).str.replace(r",]", "]", regex=True).str.replace(r"\[,", "[", regex=True).apply(literal_eval).apply(np.array) 
        self.embeddings_df['weights']          = self.embeddings_df['weights'].str.replace(r"\s+", ",", regex=True).str.replace(r",]", "]", regex=True).str.replace(r"\[,", "[", regex=True).apply(literal_eval).apply(np.array) 
        self.embeddings_df['sentence_vec']     = self.embeddings_df['sentence_vec'].str.replace(r"\s+", ",", regex=True).str.replace(r",]", "]", regex=True).str.replace(r"\[,", "[", regex=True).apply(literal_eval).apply(np.array) 
        self.embeddings_df['cluster_labels']   = self.embeddings_df['cluster_labels'].str.replace(r"\s+", ",", regex=True).str.replace(r",]", "]", regex=True).str.replace(r"\[,", "[", regex=True).apply(literal_eval).apply(list) 
        self.embeddings_df['cluster_probs']    = self.embeddings_df['cluster_probs'].str.replace(r"\s+", ",", regex=True).str.replace(r",]", "]", regex=True).str.replace(r"\[,", "[", regex=True).apply(literal_eval).apply(list) 

        self.plot_path      = Path(__file__).resolve().parent / 'plots'
        self.db_url = f'postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}' 
        
        ## read in spelling corpus 
        spelling_bucket = "spelling"
        spell_key = 'spelling_corpus.txt'
        
        self.spelling_words = Counter(re.findall(r'\w+', read_s3(bucket_name=spelling_bucket, file_key=spell_key, file_type = 'txt').lower()))
        
        model_bucket = "cluster-models"
        
        ## load dpgmm model 
        with open(read_s3(bucket_name=model_bucket, file_key=cluster_model_path, file_type='.pkl'), 'rb') as f:
            self.cluster_model = pickle.load(f)
        
    
    def embed_new_sentence(self, window = 5):
        
        nlp = spacy.load("en_core_web_sm")
        
        ## if a parapgraph split into sentences
        sentences =  self.split_paragraph_into_sentences(self.new_text)
        
        stop_words = set(stopwords.words('english'))
        
        embeddings = []
        row = 0
        ## tokenize
        for s in sentences:
            
            s = re.sub(r'[^\w\s]', '', s).strip()
            
            if s:
                
                processed_s = nlp(s) 
                
                orig_tokens         = [token.text.lower() for token in processed_s]
                orig_tokens_no_stop = [word for word in orig_tokens if word not in stop_words]
                proper_nouns        = [token.text.lower() for token in processed_s if token.pos_ == "PROPN"] 
                spell_corrected = []
                vectors = []
                
                for token in orig_tokens_no_stop: 
                    if token not in self.spelling_words and token not in proper_nouns and len(wordnet.synsets(token)) == 0:
                        spell_corrected.append(self.spell_correct(token))
                    else: 
                        spell_corrected.append(token)
                
                for i in range(len(orig_tokens_no_stop)):
                    
                    token = spell_corrected[i]
                    
                    print(f'Embedding token: {token}')
                    
                    context = []
                    rem = len(spell_corrected) - 1 - i
                    rem = min(rem, window)
                    iter = 1
                    while rem > 0:
                        context.append(spell_corrected[i + iter])
                        rem -= 1
                        iter += 1

                    pad = i 
                    pad = min(pad, window)
                    
                    iter = 1
                    while pad > 0:
                        context.append(spell_corrected[i - iter])
                        pad -= 1
                        iter += 1
                        
                    
                    index = None
                    vocab_list = self.vocab.flatten().tolist()
                    ## handling OOV words
                    if token not in vocab_list:
                        
                        print(f"Token {token} not found in vocab looking for a replacement.")
                        
                        ## try fuzzy matching -- will be applied to proper nouns (with a score of 90 it is likely just one or two chars are being being changed) 
                        best_match, score = process.extract(token, vocab_list, limit = 1)[0]
                        
                        if score > 90:
                            index =  vocab_list.index(best_match)
                            
                        ## next try - find a synonym and compare to context 
                        if not index:
                            
                            if wordnet.synsets(token):
                                
                                synonyms = wordnet.synsets(token)
                                synonym_words = set([lemma.name().lower() for syn in synonyms for lemma in syn.lemmas()])
                                synonym_words_set = {syn for syn in synonym_words if syn in vocab_list}
                                
                                if len(synonym_words_set) > 1:
                                    
                                    print(f'Synonyms identified: {synonym_words_set}')
                                    
                                    ## check if any synonyms have similar context words 
                                    ## check if any context words from text are in vocab -- take intersection 
                                    common_elements = set(context) & set(vocab_list)
                                        
                                    if len(common_elements) > 0:
                                        
                                        common_dict = {}
                                        for syn in synonym_words_set:
                                            syn_index = vocab_list.index(syn)
                                            common_dict[str(syn_index)] = {
                                                'total_matches': 0,
                                                'distinct_context_matches': 0, 
                                                'row_context_indices' : [],
                                                'row_syn_indices':  []
                                            }
                                            
                                            for ele in common_elements:
                                                ele_index = np.where(self.vocab == ele)
                                                mask1 = self.embeddings_df['token_indices'].apply(lambda x: ele_index in x.tolist()) 
                                                mask2 = self.embeddings_df['token_indices'].apply(lambda x: syn_index in x) 
                                                mask3 = (abs(self.embeddings_df['token_indices'].apply(lambda x: x.tolist().index(ele_index) if ele_index in x.tolist() else None) - self.embeddings_df['token_indices'].apply(lambda x: x.tolist().index(syn_index) if syn_index in x.tolist() else None)) <= window) 
                                                
                                                syn_ele_df = self.embeddings_df[mask1 & mask2 & mask3]
                                                    
                                                for index, row in syn_ele_df.iterrows():
                                                    
                                                    context_indices = [i for i, x in enumerate(row['token_indices']) if x == ele_index]
                                                    syn_indices     = [i for i, x in enumerate(row['token_indices']) if any([abs(context_index - i) <= window for context_index in context_indices]) & x == syn_index ]
                                                    
                                                    ## one-to-one mapping for unqiue row/list index 
                                                    ## if row_syn index has duplicate values this indicates multiple context words appear in the same window for the same sentence
                                                    row_syn_indices     = [len(self.embeddings_df) * index + x for x in syn_indices]
                                                             
                                                    common_dict[str(syn_index)]['row_syn_indices'].extend(row_syn_indices)
                                                
                                                ## add n matches to dict
                                                common_dict[str(syn_index)]['total_matches'] += sum(mask)
                                                common_dict[str(syn_index)]['distinct_context_matches'] += 1 
                                        
                                        ## identify maximum number of distinct contexts appearing in the same window, for each synonym
                                        max_syn_dups = []
                                        max_distinct_list = []
                                        max_total_list = []
                                        for syn in common_dict.keys():
                                            
                                            ## criteria 1, record with the max number of context matches withina given window
                                            max_syn_dup = 0
                                            ## count duplicate row_syn indices, this indicates multiple context shows up in window of synonym
                                            row_targ_set = set(common_dict[syn]['row_syn_indices'])
                                            
                                            for val in row_targ_set:
                                                if len([x for x in common_dict[syn]['row_syn_indices'] if x == val]) > max_syn_dup:
                                                        max_syn_dup = len([x for x in common_dict[syn]['row_syn_indices'] if x == val])
                                            
                                            
                                            ## criteria 2, record with the maximum distinct matches 
                                            max_distinct_match = common_dict[syn]['distinct_context_matches']
                                            ## criteria 3, record with max context matches total 
                                            max_distinct_match = common_dict[syn]['total_matches']
                                            
                                            max_syn_dups.append(max_syn_dup)
                                            max_distinct_list.append(max_distinct_match)
                                            max_total_list.append(max_distinct_match)
                                            
                                            
                                                        
                                        ## identify record 
                                        max_syn_dups_indices = [i for i, x in enumerate(max_syn_dups) if x == max(max_syn_dups)]
                                        if len(max_row_targ_indices) == 1:
                                            index = common_dict.keys()[max_syn_dups_indices[0]]
                                        else:
                                            max_distinct_list_filetered = [x for i, x in enumerate(max_distinct_list) if i in max_syn_dups_indices]
                                            max_distinct_indices = [i for i, x in enumerate(max_distinct_list_filetered) if x == max(max_distinct_list_filetered)]
                                            if len(max_distinct_indices) == 1:
                                                index = common_dict.keys()[max_distinct_indices[0]]
                                            else:
                                                max_total_list_filetered = [x for i, x in enumerate(max_total_list) if i in max_distinct_indices]
                                                max_total_indices = [i for i, x in enumerate(max_total_list_filetered) if x == max(max_total_list_filetered)]
                                                if len(max_total_indices) == 1:
                                                    index = common_dict.keys()[max_total_indices[0]]
                                        

                        ## use context to find best match 
                        if not index:
                            
                            potential_targets = {}
                            common_elements = set(context) & set(vocab_list)
                            
                            for ele in common_elements: 
                                ele_index = vocab_list.index(ele)
                                mask = self.embeddings_df['token_indices'].apply(lambda x: ele_index in x.tolist())
                                
                                ele_df = self.embeddings_df[mask]
                                
                                ## get all potential targets (words within window)
                                for index, row in ele_df.iterrows():
                                    
                                    context_indices = [i for i, x in enumerate(row['token_indices']) if x == ele_index]
                                    potential_targs = [x for i, x in enumerate(row['token_indices']) \
                                                         if any([abs(context_index - i) <= window for context_index in context_indices])]
                                    targ_indices    = [i for i, x in enumerate(row['token_indices']) \
                                                         if any([abs(context_index - i) <= window for context_index in context_indices])]
                                
                                    
                                    ## add context match info for each potential targ
                                    for i in range(len(potential_targs)):
                                        
                                        ## create on-to-one mapping df row, list element mapping 
                                        ## row 2 of df = 2 * 1000 = 2000 for 100 record df
                                        row_index = index * len(self.embeddings_df)
                                        ## list index its value itself will be one to one so long as max sentence length is less than number of sentences
                                        ## this seems fair
                                        ele_index = targ_indices[i]
                                        
                                        targ = potential_targs[i]

                                        if str(targ) not in potential_targets.keys():
                                            potential_targets[str(targ)] = {
                                                'contexts': set([ele_index]),
                                                'row_indices': [row_index],
                                                'row_targ_indices' : [row_index + ele_index]
                                            }
                                        else:
                                            potential_targets[str(targ)]['contexts'] = potential_targets[str(targ)]['contexts'] | {ele_index}
                                            potential_targets[str(targ)]['row_indices'].append(row_index)
                                            potential_targets[str(targ)]['row_targ_indices'].append(row_index + ele_index)
                            
                            ## find best target -- criteria 1, record with the max number of context matches withina given window
                            max_row_targ_list = []
                            max_distinct_list = []
                            max_total_list = []
                            for targ in potential_targets.keys():
                                
                                ## criteria 1, record with the max number of context matches withina given window
                                max_row_targ = 0
                                ## count duplicate row_targ indices, this indicates target shows up in window of multiple context words
                                row_targ_set = set(potential_targets[targ]['row_targ_indices'])
                                
                                
                                for val in row_targ_set:
                                    if len([x for x in potential_targets[targ]['row_targ_indices'] if x == val]) > max_row_targ:
                                        max_row_targ = len([x for x in potential_targets[targ]['row_targ_indices'] if x == val])
                                
                                ## criteria 2, record with the maximum distinct matches 
                                max_distinct_match = len(potential_targets[targ]['contexts'])
                                
                                ## criteria 3, record with max context matches total 
                                max_total_match = len(potential_targets[targ]['row_indices'])
                                
                                max_row_targ_list.append(max_row_targ)
                                max_distinct_list.append(max_distinct_match)
                                max_total_list.append(max_total_match)
                            
                            ## identify record 
                            key_list = list(potential_targets.keys())
                            max_row_targ_indices = [i for i, x in enumerate(max_row_targ_list) if x == max(max_row_targ_list)]
                            if len(max_row_targ_indices) == 1:
                                index = int(key_list[int(max_row_targ_indices[0])])
                            else:
                                max_distinct_list_filtered = [x for i, x in enumerate(max_distinct_list) if i in max_row_targ_indices]
                                max_distinct_indices = [i for i, x in enumerate(max_distinct_list_filtered) if x == max(max_distinct_list_filtered)]
                                if len(max_distinct_indices) == 1:
                                    index = int(key_list[max_row_targ_indices[max_distinct_indices[0]]])
                                else:
                                    max_total_list_filetered = [x for i, x in enumerate(max_total_list) if i in max_distinct_indices]
                                    max_total_indices = [i for i, x in enumerate(max_total_list_filetered) if x == max(max_total_list_filetered)]
                                    index = int(key_list[max_row_targ_indices[max_distinct_indices[max_total_indices[0]]]])
                        
                        ## if no match found, exclude word from the vector average    
                        if not index:
                            print("No replacement word found")
                        else:
                            print(index)
                            print(f'{vocab_list[int(index)]} identified as best match for {token}')
                        
                    else:
                            
                        index = vocab_list.index(token)
                    
                    if index: 
                        vec = self.vectors[int(index)] 
                    
                        ## add idf weights for each word here
                        vectors.append(vec)
            
                embedded_sentence = np.mean(np.array(vectors), axis = 0, dtype=np.float64)
                
                embeddings.append(embedded_sentence)
            
            return embeddings
                
    
    def classify_sentence(self, sentence_embeddings, threshold):
        
        dpgmm = self.cluster_model 
        
        ## get probability of new text belonging to a cluster for each gaussian cluster 
        probs = dpgmm.predict_proba(sentence_embeddings)
        
        ## identify clusters meeting a specific probability threshold
        above_threshold = [np.where(row > threshold)[0] for row in probs]
        
        return above_threshold
    
    def compute_similarity(self, sentence_embedding, clusters):
        
        ##normalize 
        mask =self.embeddings_df['cluster_labels'].apply(lambda clus_list: any(clus in clusters for clus in clus_list))
        
        cluster_mask_df = self.embeddings_df[mask]
        
        sentence_pks = cluster_mask_df['pk']
    
        book_vecs = np.array(cluster_mask_df['sentence_vec'].tolist())
        
        norm_book_vecs = np.linalg.norm(book_vecs, axis = 1)
        
        norm_input = np.linalg.norm(sentence_embedding)
        
        similarities = [np.dot(book_vec, np.squeeze(sentence_embedding, axis = 0))/ (norm_book_vecs[i] * norm_input) for i, book_vec in enumerate(book_vecs)]
        
        return similarities, sentence_pks
        
    
    def insert_request(self, pk:int, response:str, score:float):
        
        request = self.new_text 
        
        statement = """
               INSERT INTO responses (fk_books, fk_paragraphs, fk_sentences, request_date, request_text, response_text, score) 
               VALUES (
                   (SELECT pk FROM books WHERE title = :book_title 
                                               AND author = :book_author),
                   (SELECT fk_paragraph FROM sentences WHERE fk_books = (SELECT pk FROM books WHERE title = :book_title 
                                                                AND author = :book_author)
                                                        AND pk = :pk),
                   :pk,
                   NOW(),
                   :request,
                   :response,
                   cast(:score as double precision)
               )
        """
        
        
        self.execute_statement(statement, {
            "request" : request,
            "response" : response[0],
            "book_title": self.title,
            "book_author": self.author,
            "score" : score,  
            "pk": pk
        })
                                        
    def spell_correct(self, raw_word):
        
        word_set = set(self.spelling_words)
        candidates = None 
        edits = set()
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        
        ## incrementally apply edits
        while not candidates:
            ## potential ways to split the text
            splits     = [(raw_word[:i], raw_word[i:]) for i in range(len(raw_word) + 1)]
            
            ## get every edit permutation for 1 edits 
            deletes    = [left + right[1:]               for left, right in splits if right]
            transposes = [left + right[1] + right[0] + right[2:] for left, right in splits if len(right)>1]
            replaces   = [left + c + right[1:]           for left, right in splits if right for c in letters]
            inserts    = [left + c + right               for left, right in splits for c in letters]
            
            edit_perms = set(deletes + transposes + replaces + inserts)
            
            edits = edit_perms 
            
            intersection = edits & word_set
            
            if len(intersection):
                
                candidates = intersection 
        
        ## find candidate with best prob 
        probs = [self.compute_word_prob(candidate) for candidate in candidates]
        max_candidate = probs.index(max(probs))
        
        return(list(candidates)[max_candidate])
        
        
    def compute_word_prob(self, word):
        N = len(self.spelling_words)
        
        return self.spelling_words[word]/N 
        
    def split_paragraph_into_sentences(self, paragraph):
        
        ## identify quoted text
        quoted_text = re.findall(r'["\']([^"]*)[\'"]', paragraph)  # Extract quoted text 
            
        ## replace quoted text with placeholders so they aren't split
        placeholder = "<QUOTE>"
        for idx, quote in enumerate(quoted_text):
            paragraph = paragraph.replace(f'"{quote}"', f'{placeholder}{idx}{placeholder}')

        # split on typical endings
        sentences = re.split(r'(?<=\w[.!?])\s+(?!(?:Mr|Dr|Prof|Inc|Jr|Sr|vs|etc|Ms|Mrs)\.)(?=(?!<QUOTE>\d+<QUOTE>))', paragraph.strip())

        # replace placeholders with original text
        for idx, quote in enumerate(quoted_text):
                sentences = [s.replace(f'{placeholder}{idx}{placeholder}', f'"{quote}"') for s in sentences]

        return sentences
    
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
   