from shared.utils import InsertNewBook
import json 

def find_guten_book(event, context):
    
    title       = event['title']
    author      = event['author']
    find        = event['scrape_text']
    
    insert_class = InsertNewBook(title = title,
                                 author = author,
                                 find = find)
    
    ## confirm not in db
    identified_book, found = insert_class.confirm_not_in_db_books(title_new = title, author_new=author, threshold = .7)
    if found:
        
        title, author = identified_book
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'title': title,
                'author': author,
                'url': None,
                'message': 'Author and title identified already exist'
            })
        }
        
        
    try:
        urls, titles, authors = insert_class.find_gutenberg_books()
                
        if urls:
                
            response_data = [
                {"author": author, "title": title, "url": url}
                for author, title, url in zip(authors, titles, urls)
            ]
                
                    
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'results': response_data,
                    'message': 'Success, URLs identified'
                })
            }
        
        else:
            return {
                'statusCode': 204,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'message': 'No URLs identified'
                })
            }
            
    except Exception as e:
            
        return {
            "statusCode": 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'message': f'An unexpected error occurred {e}'
            })
        }