import requests
import json
from bs4 import BeautifulSoup
import re 
from sqlalchemy import create_engine, text
from typing import List
from pathlib import Path

class InsertNewBook:
    
    def __init__(self, 
                 config_path: str, 
                 title: str, 
                 author: str,
                 find = True,
                 txt_file = None):
        
        # Get the path of the current script
        script_dir = Path(__file__).resolve().parent.parent.parent

        # Define the relative path to the config file
        config_file = script_dir / config_path
    
        with open(config_file, "r") as file:
            config = json.load(file)

        self.db_user        = config["DB_USER"]
        self.db_password    = config["DB_PASSWORD"]
        self.db_host        = config["DB_HOST"]
        self.db_port        = config["DB_PORT"]
        self.db_name        = config["DB_NAME"]
        self.title          = title
        self.author         = author 
        self.db_url = f'postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}'
        
        if not find:
            try: 
                
                with open(txt_file, 'r') as file:
                    # Read the content of the file
                    content = file.read()
                
                self.book_txt = content
            
            except Exception as e:
                
                print("Error in reading text file")
    

    def contains_chapter_start(self, text: str) -> bool:
        """
        Checks if a string contains text indicating the start of a novel chapter or part.
        
        Parameters:
            text (str): The input string to check.
        
        Returns:
            bool: True if the string matches a chapter/part start pattern, False otherwise.
        """
        pattern = r'\b(?:chapter|part|section|book|prologue|epilogue|[ivxlcdm]|[IVXLCDM]|)\b\s*(?:\d+|[ivxlcdm]+|[IVXLCDM]+|one|two|three|[a-z]+)?'
        return bool(re.search(pattern, text, re.IGNORECASE))


    ## scrape text of a book 
    def scrape_book_text(self) -> List[str]:
        
        """
        Downloads a full text of a book from a given URL and saves it to a file. 
        From project Gutenberg, limited to books on project Gutenberg
        
        Parameters:
            title -- title of the  book 
            author -- author of the book
            language -- language of publication
        
        
        Returns:
            list of strings (each paragraph of book)

        """
        if f"{self.title}{self.author}" == '':
            print("No parameters specified please specify")
            return 
        
        try: 
            
            base_url = 'https://www.gutenberg.org/'
            
            book_url = f"{base_url}/ebooks/search/?query={'+'.join([self.title, self.author])}&submit_search=Go!"
            
            print(book_url)
            
            book_response = requests.get(book_url)
            
            book_response.raise_for_status()
            
            soup = BeautifulSoup(book_response.text, 'html.parser')
            
            book_links = soup.find_all('li', class_ = 'booklink')
            
            desired_link = None
            
            for link in book_links:
                
                found_title = link.find('span', class_ = 'title').get_text()
                
                found_author = link.find('span', class_ = "subtitle").get_text()
                
                desired_link = link.a['href']
                break
                
                #y_n_response = input(f"Are you looking for {found_title} by {found_author} (yes/no)").strip().lower()
                
                #if y_n_response in ['yes', 'y']:
                    #desired_link = link.a['href']
                    #break
            
                #elif y_n_response in ['no', 'n']:
                    #continue
                    
                #else:
                    
                   # print("Please enter 'yes' or 'no'.")
                    
                    #y_n_response = input(f"Are you looking for {found_title} by {found_author} (yes/no)").strip().lower()
                
                
                
            desired_page = f"{base_url}{desired_link}"
            
            desired_response = requests.get(desired_page)
            
            desired_response.raise_for_status()
            
            desired_soup = BeautifulSoup(desired_response.text, 'html.parser')
            
            desired_html = desired_soup.find('a', {
                'type'  : 'text/html',
                'title' : 'Read online'
            })['href']
            
            ## last response/soup I swear 
            text_response = requests.get(f"{base_url}{desired_html}")
            
            text_response.raise_for_status()
            
            text_soup = BeautifulSoup(text_response.text, 'html.parser')
            
            chapters = text_soup.find_all('div', class_ = 'chapter')
            
            paragraphs = [] 
            start_reading = False

            for chap in chapters:
                
                if self.contains_chapter_start(chap.h2.get_text()):
                    start_reading = True 
                    
                if start_reading:  
                    
                    ps = chap.find_all('p')
                    
                    for p in ps:
                        
                        if p.get('class') != ['footnote']:
                        
                            p_text = re.sub("  +", " ", p.get_text().replace('\r',' ').replace('\n',' '))
                            
                            # Replace both single and double quotes with two single quotes
                            p_text = re.sub(r"[\'\"]", "''", p_text)
                            
                            paragraphs.append(p_text)
            
            return(paragraphs)

        except requests.RequestException as e:
            print(f"Book query request failed: {e}")
            
        except Exception as e:
            print(f"An unexpected error occcured: {e}")
    
    def clean_txt_file(self) -> List[str]:
        
        paragraphs = text.split("\n\n")
        
        filtered_paragraphs = [
            para.strip() for para in paragraphs
            if re.search(r"[.!?](\s|$)", para.strip()) and not self.contains_chapter_start(para.strip())
        ] 
        
        return paragraphs
        
    def execute_statement(self, qry, dict):

        try: 
            
            engine = create_engine(self.db_url, isolation_level="AUTOCOMMIT")
            
            with engine.connect() as connection:
                
                try:
                    # execute
                    print(qry)
                    connection.execute(text(qry), dict)

                    print("Transaction committed successfully!")
                    return True
                    
                except Exception as e:
                    
                    print("An error occurred when executing:", e)
                    stop
                    return False 
        
        except Exception as e:
            
            print(f"Error occurred when connecting: {e}")
            stop
            return False 
            


    def insert_book_db(self) -> str: 
        
        """
        Adds a book and author to the books table
        
        Returns:
            Status of qry

        """
        
        statement = f"""INSERT INTO books (title, author) 
                        VALUES (:book_title, :book_author) 
                        ON CONFLICT ON CONSTRAINT unique_book
                        DO UPDATE SET title = EXCLUDED.title, author = EXCLUDED.author;
                    """
        
        self.execute_statement(statement, {
                                            "book_title": self.title, 
                                            "book_author": self.author 
                                          })
        
        
    def insert_text_db(self, paragraphs) -> str:
        """
        Adds text to the paragraphs and sentances tables
        
        Returns:
            Status of qry

        """
        
        for p in paragraphs:
            
            sentences =  self.split_paragraph_into_sentences(p)
            
            sentence_count = len([s for s in sentences if s])
            
            p_statement = f"""
                            INSERT INTO paragraphs (fk_books, paragraph_text, length, text_search_vector, paragraph_text_md5) 
                            VALUES (
                                (SELECT pk FROM books WHERE title = :book_title 
                                                      AND author = :book_author),
                                :para_text,
                                :sentence_ct,
                                to_tsvector('english', :para_text),
                                md5(:para_text)
                                
                            ) 
                            ON CONFLICT ON CONSTRAINT unique_paragraph
                            DO UPDATE SET paragraph_text = EXCLUDED.paragraph_text, length = EXCLUDED.length;
                           """
                           
            ## execute paragraph insert 
            self.execute_statement(p_statement, {
                                                            "book_title": self.title, 
                                                            "book_author": self.author,
                                                            "para_text": p.strip(),
                                                            "sentence_ct": sentence_count 
                                                        })
            
            
            for s in sentences: 
                if s:
                    
                    sentence_length = len(s)
                    
                    s_statement = f"""
                                    INSERT INTO sentences (fk_books, fk_paragraph, sentence_text, length, text_search_vector, sentence_text_md5) 
                                    VALUES (
                                        (SELECT pk FROM books WHERE title = :book_title 
                                                            AND author = :book_author),
                                        (SELECT pk FROM paragraphs WHERE fk_books = (SELECT pk FROM books WHERE title = :book_title 
                                                                                                          AND author = :book_author) 
                                                                   AND paragraph_text = :para_text),
                                                            
                                        :sentence_text,
                                        :sentence_len,
                                        to_tsvector('english', :sentence_text),
                                        md5(:sentence_text)
                                    ) 
                                    ON CONFLICT ON CONSTRAINT unique_sentence
                                    DO UPDATE SET sentence_text = EXCLUDED.sentence_text, length = EXCLUDED.length;
                                  """
                    self.execute_statement(s_statement, {
                                                            "book_title": self.title, 
                                                            "book_author": self.author,
                                                            "para_text": p.strip(),
                                                            "sentence_text": s.strip(),
                                                            "sentence_len": sentence_length 
                                                        })

    def split_paragraph_into_sentences(self, paragraph):
        ## identify quoted text
        quoted_text = re.findall(r'["\']([^"]*)[\'"]', paragraph)  # Extract quoted text
        
        ## replace quoted text with placeholders so they aren't split
        placeholder = "<QUOTE>"
        for idx, quote in enumerate(quoted_text):
            paragraph = paragraph.replace(f'"{quote}"', f'{placeholder}{idx}{placeholder}')

        # split on typical endings
        sentences = re.split(r'(?<=\w[.!?])\s+(?!(?:Mr|Dr|Prof|Inc|Jr|Sr|vs|etc)\.)(?=(?!<QUOTE>\d+<QUOTE>))', paragraph.strip())

        # replace placeholders with original text
        for idx, quote in enumerate(quoted_text):
            sentences = [s.replace(f'{placeholder}{idx}{placeholder}', f'"{quote}"') for s in sentences]

        return sentences