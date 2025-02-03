import json
import sys
from handler import find_guten_book

def test_lambda():
    
    test_event = {
        'title': 'Pride and Prejudice',
        'author': 'Jane Austen',
        'scrape_text': True
    }
    
    ## general checks 
    ## expected keys
    expected_keys = {"statusCode", "headers", "body"}

    response = find_guten_book(test_event, None)
    
    assert isinstance(response, dict), "Response should be a dictionary"
    assert expected_keys.issubset(response.keys()), f"Missing expected keys: {expected_keys - response.keys()}"
    
    ## status codes
    assert response["statusCode"] in {200, 204, 500}, f"Unexpected statusCode: {response['statusCode']}"
   
    try:
        
        body = json.loads(response["body"])
        
        assert isinstance(body, dict), "Body should be a JSON object"
        assert "message" in body, "Response body should contain 'message'"
        
        assert body['message'] in {'Author and title identified already exist', \
                                    'Success, URLs identified', \
                                    'No URLs identified', \
                                    'An unexpected error occurred'}, f"Unexpected message: {body['message']}"
        
    except json.JSONDecodeError:
        assert False, "Response body is not valid JSON"
    
    ## specific checks 
    ## assert success, this book is on guten
    assert response["statusCode"] == 200, f"Unexpected failure: {response['statusCode']}"
    
    assert body['message'] in {'Author and title identified already exist', \
                                    'Success, URLs identified'}
    
    assert "results" in body , "No results for Pride and Prejudice" 
    
    
    print(body['results'])
    

    print("All tests passed.")

if __name__ == "__main__":
    try:
        test_lambda()
    except AssertionError as e:
        print(f"Test failed: {e}")
        sys.exit(1)