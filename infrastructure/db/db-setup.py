from utils import DBUtils
import os

## initialize DButils
db_utils = DBUtils()

## create database
with open("./db-build.sql", "r") as file:
    db_build_script = file.read()

try:
    db_utils.execute_statement(qry = db_build_script)

except Exception as e:
    
    print(f"Error encountered, likely database already exists: {e}")
    
## create enums
try:
    db_utils.execute_script(qry = './enum-build')

except Exception as e:
    print(f"Error encounter, likely that type already exits: {e}")

## create tables
try: 
    db_utils.execute_script(script = './table-build')
except Exception as e:
    print(f"Error encounter, likely that table already exits: {e}")

## run modification scripts
for filename in os.listdir('./db_alter_scripts'):
    name_without_extension = os.path.splitext(filename)[0]
    
    try: 
        db_utils.execute_script(script = f'./db_alter_scripts/{name_without_extension}')
    except Exception as e:
        print(f'Error occurred, likely a object already exists: {e}')
    
    
