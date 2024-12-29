from utils import DBUtils

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
db_utils.execute_script(script = 'infrastructure/db/table-build')
