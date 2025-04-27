from shared.utils import Predict
from shared.utils import execute_statement, confirm_cluster
import numpy as np

def predict_handler(event, context):
    
    new_text    = event['text']
    title       = event['title']
    author      = event['author']
    n_responses = event['n_responses']
    
    ## get window size 
    statement = """
        SELECT window_size FROM Embeddings
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
    
    window_size = response.fetchone()
    
    vocab        =f'{title}_{author}_metadata.tsv'
    word_vectors =f'{title}_{author}_vectors.tsv'
    labeled_path = f'{title}_{author}_labeled.csv'
    model_path   = f'DPGMM_{title}_{author}.pkl'
    
    cluster_bool, response = confirm_cluster(title, author)
    
    if not cluster_bool:
        return {
            'statusCode':500,
            'data':      None,
            'body': f'Cluster does not exist, not predicting'   
        }
    
    try:
        predict_class = Predict(new_text,
                                title = title,
                                author = author,
                                vocab = vocab,
                                vectors = word_vectors,
                                embeddings_path=labeled_path,
                                cluster_model_path=model_path,
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
        for i, response in enumerate(best_responses):
            predict_class.insert_request(pk = best_n_pks[i], response = response, score = best_n[i])
            
        ## return respoonse 
        response_data  = [
            {'rank': i, 'response': response}
            for i, response in enumerate(best_responses)
        ]
        
        return {
            'statusCode': 200,
            'data': json.dumps(response_data),
            'message': f'success, best {n_responses} returned'
            }
    
    except Exception as e:
        
        return {
            'statusCode': 500,
            'data': None,
            'message': f'An error occurred in prediction: {e}'
            
        }
          