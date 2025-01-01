ALTER TABLE paragraphs DROP CONSTRAINT unique_paragraph;
ALTER TABLE sentences DROP CONSTRAINT unique_sentence;
ALTER TABLE paragraphs DROP COLUMN text_search_vector;
ALTER TABLE sentences DROP COLUMN text_search_vector;

-- Create a tsvector column for full-text search
ALTER TABLE paragraphs ADD COLUMN text_search_vector tsvector;

-- Populate the tsvector column with the text data
UPDATE paragraphs SET text_search_vector = to_tsvector('english', paragraph_text);

-- Create the full-text index
CREATE INDEX unique_paragraph_fulltext_idx ON paragraphs USING gin(text_search_vector);

-- Add MD5 column for uniqueness
ALTER TABLE paragraphs ADD COLUMN paragraph_text_md5 TEXT;

-- Populate with MD5 hashes
UPDATE paragraphs SET paragraph_text_md5 = md5(paragraph_text);

-- Create the unique constraint on MD5 hash
ALTER TABLE paragraphs ADD CONSTRAINT unique_paragraph UNIQUE (fk_books, paragraph_text_md5);

-- Create a tsvector column for full-text search
ALTER TABLE sentences ADD COLUMN text_search_vector tsvector;

-- Populate the tsvector column with the text data
UPDATE sentences SET text_search_vector = to_tsvector('english', sentence_text);

-- Create the full-text index
CREATE INDEX unique_sentence_fulltext_idx ON sentences USING gin(text_search_vector);

-- Add MD5 column for uniqueness
ALTER TABLE sentences ADD COLUMN sentence_text_md5 TEXT;

-- Populate with MD5 hashes
UPDATE sentences SET sentence_text_md5 = md5(sentence_text);

-- Create the unique constraint on MD5 hash
ALTER TABLE sentences ADD CONSTRAINT unique_sentence UNIQUE (fk_books, fk_paragraph, sentence_text_md5)

-- Responses
ALTER TABLE responses DROP CONSTRAINT unique_request;

-- Create a tsvector column for full-text search
ALTER TABLE responses ADD COLUMN text_search_request_vector tsvector;
ALTER TABLE responses ADD COLUMN text_search_response_vector tsvector;

-- Populate the tsvector column with the text data
UPDATE responses SET text_search_request_vector = to_tsvector('english', request_text);
UPDATE responses SET text_search_response_vector = to_tsvector('english', response_text);

-- Create the full-text index
CREATE INDEX unique_request_fulltext_idx ON responses USING gin(text_search_request_vector);
CREATE INDEX unique_request_fulltext_idx ON responses USING gin(text_search_response_vector);

-- Add MD5 column for uniqueness
ALTER TABLE responses ADD COLUMN request_text_md5 TEXT;

UPDATE responses SET request_text_md5 = md5(request_text);

ALTER TABLE responses ADD CONSTRAINT unique_request UNIQUE (request_date, request_text_md5);

ALTER TABLE responses ADD COLUMN fk_sentences INT;
ALTER TABLE responses ADD COLUMN fk_paragraphs INT;