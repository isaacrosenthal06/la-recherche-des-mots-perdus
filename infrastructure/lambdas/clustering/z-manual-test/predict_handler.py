from utils import Predict
import numpy as np

def predict_handler(new_text, title, author, window_size, n_responses):
    
    vocab =f'processing/embeddings/{title}_{author}_metadata.tsv'
    word_vectors =f'processing/embeddings/{title}_{author}_vectors.tsv'
    labeled_path = f'clustering/clustered/{title}_{author}_labeled.csv'
    model_path = f'clustering/models/DPGMM_{title}_{author}.pkl'
    
    
    predict_class = Predict(new_text,
                            'db/config.json',
                            title = title,
                            author = author,
                            vocab = vocab,
                            vectors = word_vectors,
                            embeddings_path=labeled_path,
                            cluster_model_path=model_path
                            )
    
    
    ## embed new text 
    new_text_embedded = predict_class.embed_new_sentence(window = window_size)
    
    ## cluster new text
    predicted_cluster = predict_class.classify_sentence(new_text_embedded, threshold = .2)
    
    ## compute similarities
    similarity_scores, sentence_pks = predict_class.compute_similarity(new_text_embedded, predicted_cluster[0].tolist())
    
    ## get n best responses
    best_n = sorted(similarity_scores, reverse = True)[:n_responses]
    best_n_pks = [list(sentence_pks)[similarity_scores.index(val)] for val in best_n]

    best_responses = []

    for i, pk in enumerate(best_n_pks):
        
        ## look up vocab 
        statement = """

                    SELECT sentence_text FROM sentences
                    WHERE fk_books = (SELECT pk FROM books WHERE title = :book_title 
                                                AND author = :book_author)
                    AND pk = :pk
        """
        
        dict = {
            "book_title": predict_class.title,
            "book_author": predict_class.author,
            'pk': pk  
        }
        
        response = predict_class.execute_statement(statement, dict).fetchone()
        print(response)
        best_responses.append(response)
    
    ## insert responses into database
   ##for i,response in enumerate(best_responses):
        ##predict_class.insert_request(pk = best_n_pks[i], response = response, score = best_n[i])
    
    return f"{n_responses} responses added to the db"