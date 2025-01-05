ALTER TABLE responses ADD COLUMN fk_sentences INT;
ALTER TABLE responses ADD COLUMN fk_paragraphs INT;
ALTER TABLE responses ADD COLUMN score FLOAT