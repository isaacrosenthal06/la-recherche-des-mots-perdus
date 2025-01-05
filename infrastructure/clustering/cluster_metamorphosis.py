from utils import Cluster
from utils import Predict
import numpy as np

## cluster embeddings 
metamorphosis_cluster = Cluster('db/config.json', 
                                title = "Metamorphosis", 
                                author = 'Kafka',
                                vocabulary= 'processing/embeddings/Metamorphosis_metadata.tsv',
                                tf_idf_wts = 'processing/embeddings/Metamorphosis_tf_idf_wts.csv',
                                vectors = 'processing/embeddings/Metamorphosis_vectors.tsv')

embedded_sentences = metamorphosis_cluster.embed_sentences()

## fit dpgmm model
dpgmm_model = metamorphosis_cluster.fit_clustering(np.array(embedded_sentences['sentence_vec'].tolist()), max_clusters =50, alpha = 3)

## classify sentences 
cluster_labels , probs =  metamorphosis_cluster.get_gaussian_probs(np.array(embedded_sentences['sentence_vec'].tolist()), threshold= .2, model = dpgmm_model)

embedded_sentences['cluster_labels'] = cluster_labels 
embedded_sentences['cluster_probs'] = embedded_sentences.apply(lambda row: [probs[row.name][clus_index] for clus_index in row['cluster_labels']], axis = 1)
embedded_sentences['max_prob_cluster'] = embedded_sentences.apply(lambda row: row['cluster_labels'][[i for i, val in enumerate(row['cluster_probs']) if val == max(row['cluster_probs'])][0]], axis = 1)

metamorphosis_cluster.plot_embeddings(np.array(embedded_sentences['sentence_vec'].tolist()), labels = embedded_sentences['max_prob_cluster'])

## predict new text 
new_text = "Gregor changing into a vermin"

metamorphosis_predict = Predict(new_text = new_text,
                                config_path = 'db/config.json', 
                                title = metamorphosis_cluster.title,
                                author = metamorphosis_cluster.author,
                                vocab = metamorphosis_cluster.vocab,
                                vectors = metamorphosis_cluster.word_embeddings,
                                embeddings_df = embedded_sentences,
                                cluster_model = dpgmm_model
                                )

new_text_embedded = metamorphosis_predict.embed_new_sentence(window = 3)


predicted_cluster = metamorphosis_predict.classify_sentence(new_text_embedded, threshold = .2)

similarity_scores, sentence_pks = metamorphosis_predict.compute_similarity(new_text_embedded, predicted_cluster[0].tolist())

best_10 = sorted(similarity_scores, reverse = True)[:10]
best_10_pks = [list(sentence_pks)[similarity_scores.index(val)] for val in best_10]

best_responses = []

for i, pk in enumerate(best_10_pks):
    
    ## look up vocab 
    statement = """

                SELECT sentence_text FROM sentences
                WHERE fk_books = (SELECT pk FROM books WHERE title = :book_title 
                                               AND author = :book_author)
                AND pk = :pk
    """
    
    dict = {
        "book_title": metamorphosis_predict.title,
        "book_author": metamorphosis_predict.author,
        'pk': pk  
    }
    
    response = metamorphosis_predict.execute_statement(statement, dict).fetchone()
    best_responses.append(response)
    
for i,response in enumerate(best_responses):
    metamorphosis_predict.insert_request(response = response, score = best_10[i])
    

    
    