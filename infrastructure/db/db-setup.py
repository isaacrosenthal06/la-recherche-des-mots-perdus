from utils import DBUtils
import os

## initialize DButils
db_utils = DBUtils('infrastructure/db/config.json')

## create database
with open("infrastructure/db/db-build.sql", "r") as file:
    db_build_script = file.read()

try:
    db_utils.execute_statement(qry = db_build_script)

except Exception as e:
    
    print(f"Error encountered, likely database already exists: {e}")
    
## create enums
try:
    db_utils.execute_script(qry = 'infrastructure/db/enum-build')

except Exception as e:
    print(f"Error encounter, likely that type already exits: {e}")

## create tables
try: 
    db_utils.execute_script(script = 'infrastructure/db/table-build')
except Exception as e:
    print(f"Error encounter, likely that table already exits: {e}")

## run modification scripts
for filename in os.listdir('infrastructure/db/db_alter_scripts'):
    name_without_extension = os.path.splitext(filename)[0]
    
    db_utils.execute_script(script = f'infrastructure/db/db_alter_scripts/{name_without_extension}')
    
    
