from insert_clean.utils import InsertNewBook
from embed.word2vec import WordEmbeddingModel, Word2Vec, NegativeSamplingLoss
import numpy as np
import io
from pathlib import Path
import pandas as pd

## insert book into db 
insert_metamorphosis = InsertNewBook(config_path='db/config.json', title = "Metamorphosis", author = 'Kafka') 

inserted = False
if not inserted:
    ## clear existing records 
    insert_metamorphosis.execute_statement("""
                                        DELETE FROM books 
                                        WHERE title = :book_title
                                        AND author = :book_author
                                        """, {
                                                "book_title": insert_metamorphosis.title, 
                                                "book_author": insert_metamorphosis.author
                                                })

    insert_metamorphosis.execute_statement("""
                                        DELETE FROM sentences 
                                        WHERE fk_books = (SELECT pk FROM books WHERE title = :book_title
                                                                                AND author = :book_author)
                                        """, {
                                                "book_title": insert_metamorphosis.title, 
                                                "book_author": insert_metamorphosis.author
                                                })
    insert_metamorphosis.execute_statement("""
                                        DELETE FROM paragraphs 
                                        WHERE fk_books = (SELECT pk FROM books WHERE title = :book_title
                                                                                AND author = :book_author)
                                        """, {
                                                "book_title": insert_metamorphosis.title, 
                                                "book_author": insert_metamorphosis.author
                                                })

text = insert_metamorphosis.scrape_book_text()

if not text:
    raise ValueError("No text scraped")
    
else:
    
    if not inserted:
        insert_metamorphosis.insert_book_db()

        insert_metamorphosis.insert_text_db(text)
    
    ###################
    ## Learn embeddings
    ##################
    
    ## process sentences
    embeddings_metamorphosis = WordEmbeddingModel("Metamorphosis", author = 'Kafka', corpus = text)
    
    all_sentences, sentence_pks, tokens, max_len = embeddings_metamorphosis.get_sentences()
    
    print(f"First pk {sentence_pks[0]} : {all_sentences[0]}")
    
    in_vocab_size = len(tokens)
    max_sentence_size = max_len
    
    ## vectorize sentences
    
    vocab, vectorized, tf_idf_weights = embeddings_metamorphosis.vectorize(sentences=all_sentences,
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

    df.to_csv(f'{embeddings_dir}/{embeddings_metamorphosis.title}_tf_ids_wts.csv', index=False)
    #print(vocab)
    
    # for tf_idf_wt in tf_idf_weights[:5]:
    #     print(f"tf_idf: {tf_idf_wt}")
    #     print(len(tf_idf_wt))
    
    # ## print first 5 vectorized sentences
    # for vector in vectorized[:5]:
    #     print(f"{vector} => {[vocab[i] for i in vector]}")
        
    ## get power law probabilities
    vocab_tensor, power_law_probs_tensor = embeddings_metamorphosis.compute_power_law_probabilities(vectorized)    
    
    ## train 
    targets, contexts, labels = embeddings_metamorphosis.generate_training_data(vectorized, 5, 5, in_vocab_size, power_law_probs_tensor, vocab_tensor, 1234)
    
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
    
    metamorphosis_w2v = embeddings_metamorphosis.build_and_train(vocab_size=in_vocab_size, 
                                             embedding_dim=100, 
                                             num_ns = 5, 
                                             targets= targets,
                                             contexts=contexts,
                                             labels=labels,
                                             batch_size=300,
                                             epochs = 18)
    
    
    
    weights = metamorphosis_w2v.get_layer('w2v_embedding').get_weights()[0]
    
    out_v = io.open(f'{embeddings_dir}/{embeddings_metamorphosis.title}_vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open(f'{embeddings_dir}/{embeddings_metamorphosis.title}_metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    
    out_v.close()
    out_m.close()
    