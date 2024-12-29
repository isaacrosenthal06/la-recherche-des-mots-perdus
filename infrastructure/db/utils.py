import json
from sqlalchemy import create_engine, text

class DBUtils:
    
    def __init__(self, config_path):
    
        with open(config_path, "r") as file:
            config = json.load(file)

        self.db_user        = config["DB_USER"]
        self.db_password    = config["DB_PASSWORD"]
        self.db_host        = config["DB_HOST"]
        self.db_port        = config["DB_PORT"]
        self.db_name        = config["DB_NAME"]

        self.db_url = f'postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}'

            
    def execute_statement(self, qry):

        try: 
            
            engine = create_engine(self.db_url, isolation_level="AUTOCOMMIT")
            
            with engine.connect() as connection:
                
                try:
                    # Execute some SQL statements
                    print(qry)
                    connection.execute(text(qry))

                    # Commit the transaction
                    print("Transaction committed successfully!")
                    
                except Exception as e:
                    # Rollback the transaction in case of an error
                    print("An error occurred:", e)
        
        except Exception as e:
            # Roll back the transaction if any error occurs
            print(f"Error occurred: {e}")

        
    def execute_script(self, script = '', qry = ''):
        
        try: 
            
            # split queries
            if script != '':
                with open(f"{script}.sql", "r") as file:
                    sql_script = file.read()
            
            else: 
                sql_script = script
                
            queries = sql_script.split(';')
            
            # Connect to PostgreSQL
            engine = create_engine(self.db_url)
            
            with engine.connect() as connection:
                # Begin a transaction
                trans = connection.begin()
                
                try:
                    
                    for qry in queries: 
            
                        # Execute some SQL statements
                        print(qry)
                        connection.execute(text(qry))
                        print('qry executed successfully')

                        
                    # Commit the transaction
                    print("Transaction committed successfully!")
                    trans.commit()
                    
                except Exception as e:
                    # Rollback the transaction in case of an error
                    print("An error occurred, rolling back transaction:", e)
                    trans.rollback()
            
        
        except Exception as outer_e:
            print("Connection error:", outer_e)
        
        
        
    