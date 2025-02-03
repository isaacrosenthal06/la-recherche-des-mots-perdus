import handler as handler

response, title, author, embedding_dim, window_size = handler.embed_handler(author = 'Kafka', title = 'Metamorphosis')

print(response)
print(embedding_dim)
print(window_size)