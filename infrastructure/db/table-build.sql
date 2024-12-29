CREATE TABLE IF NOT EXISTS Books (
    pk SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NOT NULL, 
    CONSTRAINT unique_book UNIQUE (title, author)
);

CREATE TABLE IF NOT EXISTS Paragraphs (
    pk SERIAL PRIMARY KEY,
    fk_books INT NOT NULL,  
    paragraph_text TEXT NOT NULL,
    attributes TEXT, 
    length INT NOT NULL,
    FOREIGN KEY (fk_books) REFERENCES Books(pk) ON DELETE CASCADE, 
    CONSTRAINT unique_paragraph UNIQUE (fk_books, paragraph_text)
);

CREATE TABLE IF NOT EXISTS Sentences (
    pk SERIAL PRIMARY KEY,
    fk_books INT NOT NULL,
    fk_paragraph INT NOT NULL,   
    sentence_text TEXT NOT NULL,
    attributes TEXT, 
    length INT NOT NULL,
    FOREIGN KEY (fk_books) REFERENCES Books(pk) ON DELETE CASCADE, 
    FOREIGN KEY (fk_paragraph) REFERENCES Paragraphs(pk), 
    CONSTRAINT unique_sentence UNIQUE (fk_paragraph, sentence_text)
);

-- Need to add more info to request
CREATE TABLE IF NOT EXISTS responses (
    pk SERIAL PRIMARY KEY,
    fk_books INT NOT NULL,
    request_date DATE NOT NULL,  
    request_text TEXT,
    request_status status,
    response_text TEXT,
    CONSTRAINT unique_request UNIQUE (request_date, request_text)  
)
