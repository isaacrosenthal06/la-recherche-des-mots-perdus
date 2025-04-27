from shared.utils import Cluster
from shared.utils import execute_statement, upload_s3
import numpy as np
from pathlib import Path
import boto3


def cluster_handler(event, context):
    
    title  = event['title']
    author = event['author']
    
    statement = """
        SELECT embedding_dim FROM Embeddings
        WHERE author = :author AND
        title = :title
    """
    
    response = execute_statement(
        statement,
        dict = {
            'title' : title,
            'author' : author
        }
    )
    
    embedding_dim = response.fetchone()
    
    
    vocab        = f'{title}_{author}_metadata.tsv'
    tf_idf_df    = f'{title}_{author}_tf_idf_wts.csv'
    word_vectors = f'{title}_{author}_vectors.tsv'
    
    cluster_class = Cluster(title = title,
                            author = author,
                            vocabulary = vocab,
                            tf_idf_wts = tf_idf_df,
                            vectors = word_vectors,
                            embedding_dim=embedding_dim)
    
    embedded_sentences = cluster_class.embed_sentences()
    
    ##fit dpgmm
    try:
        dpgmm = cluster_class.fit_clustering(np.array(embedded_sentences['sentence_vec'].tolist()), max_clusters =50, alpha = 3)
        
        ## save dpgmm 
        cluster_class.save_model(dpgmm)
        
        ## label existing data
        cluster_labels , probs =  cluster_class.get_gaussian_probs(np.array(embedded_sentences['sentence_vec'].tolist()), threshold= .2, model = dpgmm)

        embedded_sentences['cluster_labels'] = cluster_labels 
        embedded_sentences['cluster_probs'] = embedded_sentences.apply(lambda row: [probs[row.name][clus_index] for clus_index in row['cluster_labels']], axis = 1)
        embedded_sentences['max_prob_cluster'] = embedded_sentences.apply(lambda row: row['cluster_labels'][[i for i, val in enumerate(row['cluster_probs']) if val == max(row['cluster_probs'])][0]], axis = 1)

        cluster_class.plot_embeddings(np.array(embedded_sentences['sentence_vec'].tolist()), labels = embedded_sentences['max_prob_cluster'])
        
        ## save df with cluster labels
        file_path  = f'/tmp/{cluster_class.title}_{cluster_class.author}_clustered.csv'
        
        bucket_name = 'clustered-sentences'
        file_key    = f'{cluster_class.title}_{cluster_class.author}_clustered.csv'
        
        embedded_sentences.to_csv(file_path, index=False)
        
        upload_s3(file_path, bucket_name, file_key)
        
    except Exception as e:
        
        return {
            'statusCode': 500,
            'data':       None,
            'body': f'An error occurred in clustering: {e}'
        }
    
    
    