from utils import Cluster
import numpy as np
from pathlib import Path

def cluster_handler(title, author, embedding_dim): 
    
    vocab =f'processing/embeddings/{title}_{author}_metadata.tsv'
    tf_idf_df =f'processing/embeddings/{title}_{author}_tf_idf_wts.csv'
    word_vectors =f'processing/embeddings/{title}_{author}_vectors.tsv'
    
    cluster_class = Cluster('db/config.json',
                            title = title,
                            author = author,
                            vocabulary = vocab,
                            tf_idf_wts = tf_idf_df,
                            vectors = word_vectors,
                            embedding_dim=embedding_dim)
    
    embedded_sentences = cluster_class.embed_sentences()
    
    ##fit dpgmm
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
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir / 'clustered'
    embedded_sentences.to_csv(f'{file_path}/{cluster_class.title}_{cluster_class.author}_labeled.csv', index=False)
    
    return "Success"
    
    
    
    