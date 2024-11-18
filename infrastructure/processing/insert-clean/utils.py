import requests
from bs4 import BeautifulSoup
import re 

import re

def contains_chapter_start(text):
    """
    Checks if a string contains text indicating the start of a novel chapter or part.
    
    Parameters:
        text (str): The input string to check.
    
    Returns:
        bool: True if the string matches a chapter/part start pattern, False otherwise.
    """
    pattern = r'\b(?:chapter|part|section|book|prologue|epilogue)\b\s*(?:\d+|[ivxlcdm]+|one|two|three|[a-z]+)?'
    return bool(re.search(pattern, text, re.IGNORECASE))


## scrape text of a book 
def scrape_book_text(
    title = '',
    author = '',
    language = ''):
    
    """
    Downloads a full text of a book from a given URL and saves it to a file. 
    From project Gutenberg
    
    Parameters:
        url (str): The URL of the book's text page.
    
    
    Returns:
        String of all text in the book

    """
    if f"{title}{author}{language}" == '':
        print("No parameters specified please specify")
        return 
    
    try: 
        
        base_url = 'https://www.gutenberg.org/'
        
        book_url = f"{base_url}/ebooks/search/?query={'+'.join([title, author])}&submit_search=Go!"
        
        book_response = requests.get(book_url)
        
        book_response.raise_for_status()
        
        soup = BeautifulSoup(book_response.text, 'html.parser')
        
        book_links = soup.find_all('li', class_ = 'booklink')
        
        desired_link = None
        
        for link in book_links:
            
            title = link.find('span', class_ = 'title').get_text()
            
            author = link.find('span', class_ = "subtitle").get_text()
            
            y_n_response = input(f"Are you looking for {title} by {author} (yes/no)").strip().lower()
            
            if y_n_response in ['yes', 'y']:
                desired_link = link.a['href']
                break
        
            elif y_n_response in ['no', 'n']:
                continue
                
            else:
                
                print("Please enter 'yes' or 'no'.")
                
                y_n_response = input(f"Are you looking for {title} by {author} (yes/no)").strip().lower()
            
            
            
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
            
            if contains_chapter_start(chap.h2.get_text()):
                start_reading = True 
                
            if start_reading:  
                
                ps = chap.find_all('p')
                
                for p in ps:
                    
                    p_text = re.sub("  +", " ", p.get_text()).replace('\r','').replace('\n','')
                    
                    paragraphs.append(p_text)
        
        return(paragraphs)

    except requests.RequestException as e:
        print(f"Book query request failed: {e}")
        
    except Exception as e:
        print(f"An unexpected error occcured: {e}")
        
text = scrape_book_text(
    "Crime and punishment"
)

print(text)

print(len(text))


print(text[len(text)//2])