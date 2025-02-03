from utils import InsertNewBook
from word2vec import WordEmbeddingModel
import numpy as np
import io
from pathlib import Path
import pandas as pd

def embed_handler(author: str, title: str, scrape = True, txt = None):
    
    insert_class = InsertNewBook(config_path='db/config.json',
                                 title = title,
                                 author = author,
                                 find = scrape,
                                 txt_file = txt)
    
    ## confirm not in db
    identified_book, found = insert_class.confirm_not_in_db(title_new = title, author_new=author, threshold = .7)
    if found:
        
        title = identified_book[0]
        author = identified_book[1]
        
        return f'{title} by {author} already exists in the db ', title, author
    
    else: 
        if scrape:
        
            text, title_found, author_found = insert_class.scrape_book_text()
            
            identified_book, found = insert_class.confirm_not_in_db(title_new = title_found, author_new=author_found, threshold = .7)
            if found:
        
                title = identified_book[0]
                author = identified_book[1]
                return f'{title} by {author} already exists in the db ', title, author
            else:
                ## set title to identified book from gutenberg, to ensure we have consistency if the same book is scraped
                insert_class.title = title_found
                insert_class.author = author_found
        
        else: 
            
            text = insert_class.clean_txt_file()
            
        
        if not text:
            raise ValueError("No text found")
        
        ##insert into db
        insert_class.insert_book_db()
        insert_class.insert_text_db(text)
        
        ###################
        ## Learn embeddings
        ##################
        print(f'Learning embeddings for {title} by {author}')
        embeddings_class = WordEmbeddingModel(title = insert_class.title, author = insert_class.author, corpus = text)
        
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
        
        return f'Book inserted and embeddings learned for {embeddings_class.title} by {embeddings_class.author}', embeddings_class.title, embeddings_class.author, embedding_dim, window_size
            