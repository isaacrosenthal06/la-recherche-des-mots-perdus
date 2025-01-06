import predict_handler

## predict new text 
new_text = "loss of self, alienation, and absurdity"

predict_handler.predict_handler(new_text= new_text, title = "Metamorphosis", author = "Franz Kafka", window_size = 5, n_responses = 10)