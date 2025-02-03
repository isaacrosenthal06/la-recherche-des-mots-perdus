from shared.utils import InsertNewBook
from shared.word2vec import WordEmbeddingModel
import numpy as np
import io
from pathlib import Path
import pandas as pd
import json 

def insert_and_embed(event, context): 
    
    title       = event['title']
    author      = event['author']
    url         = event['book_url']

    insert_class = InsertNewBook(config_path='shared/config.json',
                                 title = title,
                                 author = author)
    
    ## confirm not in db
    identified_book, books_found = insert_class.confirm_not_in_db_books(title_new = title, author_new=author, threshold = .7)
    if books_found:
        
        ## update with title and author existing in the db
        title  = identified_book[0]
        author = identified_book[1]
        
        response_data = {'author': author, 'title': title}
        
        ## check if book is already embedded (note that books in the Books table, should have associated embeddings)    
        identified_book, embedding_found = insert_class.confirm_not_in_db_embeddings(title_new = title, author_new= author, threshold = .7)
        
        if embedding_found:
        
            response = {
                "statusCode": 200,
                "data"      : json.dumps(response_data),
                "message"   : 'Book and embeddings already exists in database'
            }
        else:
            response = {
                "statusCode": 400,
                "data"      : json.dumps(response_data),
                "message"   : 'Book exists in database but embeddings do not'
            }
            
        return response
    
    ## if not in db add new record associated with this title and author
    else: 
        
        paragraphs = insert_class.scrape_book_text(link = url, title = title, author = author)
        
        ## add books and text to the db 
        insert_class.insert_book_db()
        insert_class.insert_text_db(paragraphs)
        
    
        ## learn embeddings 
        try:
            
            print(f'Learning embeddings for {title} by {author}')
            embeddings_class = WordEmbeddingModel(title = insert_class.title, author = insert_class.author, corpus = paragraphs)
            
            all_sentences, sentence_pks, tokens, max_len = embeddings_class.get_sentences()
            
            print(f"First pk {sentence_pks[0]} : {all_sentences[0]}")
            
            in_vocab_size = len(tokens)
            max_sentence_size = max_len
            
            vocab, vectorized, tf_idf_weights = embeddings_class.vectorize(sentences=all_sentences,
                                        vocab_size = in_vocab_size, 
                                        max_sentence= max_sentence_size) 
        
            vocab_np = np.array(vocab)
            sentences_np = np.array(vectorized)
            weights_np = np.array(tf_idf_weights)

            print('\n')
            print(f"vocab.shape: {vocab_np.shape}")
            print(f"sentences.shape: {sentences_np.shape}")
            print(f"weights.shape: {weights_np.shape}")
            
            ## save tf_idf_weights and pks 
        
            ## loop through each vectorized sentence, add its primary key, list of words and corresponding tf_idf weights to a dataframe 
            rows = []
            for i in range(0, len(vectorized)):
                pk = sentence_pks[i]
                
                token_indices = vectorized[i].numpy()
                
                ## remove padding
                indices = [z for z in token_indices if z != 0]
                
                tf_idf = tf_idf_weights[i].numpy()        

                weights = []
                for index in indices:
                    ## subtract 1 to adjust for the 1-based indexing in vectorized and 0-based in tf-idf weights
                    weights.append(tf_idf[index- 1])
                    
                row = {
                    "pk": pk,
                    "token_indices": indices,
                    "weights": weights 
                }
                
                rows.append(row)
                
            df = pd.DataFrame(rows)
            
            script_dir = Path(__file__).resolve().parent
            embeddings_dir = script_dir / 'embeddings'

            df.to_csv(f'{embeddings_dir}/{embeddings_class.title}_{embeddings_class.author}_tf_idf_wts.csv', index=False)
            
            ## get power law probabilities
            vocab_tensor, power_law_probs_tensor = embeddings_class.compute_power_law_probabilities(vectorized)
            
            ## determine window size (len vocab/len sentences -- Books with large vocab bet few sentences will have a wider window)
            ##                       (max_sentence_size/5 -- Books with large sentence sizes will have a wider window)
            window_size = int(min((len(vocab_np))/len(sentences_np), max_sentence_size/5))
            
            ## set a minimum of 8
            window_size= max(window_size, 5)
            
            print(f"Using window size of {window_size}")
            
            
            ## train 
            targets, contexts, labels = embeddings_class.generate_training_data(sentences = vectorized,
                                                                                window_size = window_size,
                                                                                num_ns = 5,
                                                                                vocab_size = in_vocab_size,
                                                                                power_law_probs_tensor = power_law_probs_tensor,
                                                                                vocab_tensor = vocab_tensor,
                                                                                seed = 1234)
            
            for i in range(0, 4):
                print(f"Target: {targets[i].numpy()}")
                print(f"contexts: {contexts[i].numpy()}")
                print(f"labels: {labels[i].numpy()}")

            targets = np.array(targets)
            contexts = np.array(contexts)
            labels = np.array(labels)

            print('\n')
            print(f"targets.shape: {targets.shape}")
            print(f"contexts.shape: {contexts.shape}")
            print(f"labels.shape: {labels.shape}")    
            
            ## embedding dimension. Books with larger vocabularies and sentence sizes will be represented better with more dimensions
            ##  Min of 100 dimensions and max of 700
            embedding_dim = int(min(700, max(100, np.sqrt((len(vocab_np)) * max_sentence_size)/5)))
            
            print(f"Using embedding dimension size of {embedding_dim}")
            
            w2v_model = embeddings_class.build_and_train(vocab_size=in_vocab_size, 
                                                embedding_dim=embedding_dim, 
                                                num_ns = 5, 
                                                targets= targets,
                                                contexts=contexts,
                                                labels=labels,
                                                batch_size=300,
                                                epochs = 18)
            
            weights = w2v_model.get_layer('w2v_embedding').get_weights()[0]
            
            ## write embedding layer and vocab to output
            out_v = io.open(f'{embeddings_dir}/{embeddings_class.title}_{embeddings_class.author}_vectors.tsv', 'w', encoding='utf-8')
            out_m = io.open(f'{embeddings_dir}/{embeddings_class.title}_{embeddings_class.author}_metadata.tsv', 'w', encoding='utf-8')

            for index, word in enumerate(vocab):
                if index == 0:
                    continue  # skip 0, it's padding.
                vec = weights[index]
                out_v.write('\t'.join([str(x) for x in vec]) + "\n")
                out_m.write(word + "\n")
            
            out_v.close()
            out_m.close()
            
            ## add to the embeddings table
            ## at this point, author and title will necessarily be consistent with what is in the Books db
            ## since we only embed if the embedding does not already exist, and we embed using title and authors
            ## that were either inputs or identified as existing in the books table
            statement = """INSERT INTO Embeddings (title, author, fk_books, embedding_dim, window_size) 
                        VALUES (
                            :book_title, 
                            :book_author,
                            (SELECT pk FROM Books WHERE author = :author 
                                                  AND   title  = :title),
                            :embed_dim,
                            :window_size
                        ) 
                        ON CONFLICT ON CONSTRAINT unique_book
                        DO UPDATE SET title = EXCLUDED.title, author = EXCLUDED.author;
            """
            
            embeddings_class.execute_statement(statement, 
                                               dict = {
                                                   'book_title' : title,
                                                   'book_author': author,
                                                   'embed_dim'  : embedding_dim,
                                                   'window_size': window_size
                                               })
            
            response_data = {'author': author, 'title': title}
            
            response = {
                "statusCode": 200,
                "data"      : json.dumps(response_data),
                "message"   : 'Book and embeddings added'
            }
            
            
            return(response)
            
            
        ## if embeddings fail, delete book from db     
        except Exception as e:
            
            ## delete from books table
            statement = """DELETE FROM BOOKS (title, author, fk_books, embedding_dim, window_size) 
                        WHERE title = :book_title AND author = :book_author
            """
            
            embeddings_class.execute_statement(statement,
                                               dict = {
                                                   'book_title': title,
                                                   'book_author': author
                                               })
            
            response_data = {'author': author, 'title': title}
            
            response = {
                "statusCode": 400,
                "data"      : json.dumps(response_data),
                "message"   : 'Embeddings failed, deleted from database'
            }
            
            return(response)
            
                
            
            
            
            
            
            
        
        
        
        