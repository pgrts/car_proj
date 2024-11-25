from sqlalchemy import create_engine, text
import os 
from dotenv import load_dotenv
import pandas as pd
import numpy as np

dir_path = os.path.join(os.getcwd(), '..', 'database')

# Create the directory
os.makedirs(dir_path, exist_ok=True)

print(f"Directory created at: {os.path.abspath(dir_path)}")

load_dotenv()  # Load variables from .env file
db_password = os.getenv('DB_PASSWORD')
engine = create_engine(f'postgresql+psycopg2://postgres:{db_password}@localhost:5432/cars')

def latest_cbm_f():
    
    cbm_files =  os.listdir(os.path.join(os.getcwd(), '..', 'cb_models'))
    cbm_dates = [x.lstrip('cb_model_').rstrip('.cbm') for x in cbm_files]
    latest_file = [x for x in cbm_files if max(cbm_dates) in x][0]

    return os.path.join(os.getcwd(), '..', 'cb_models', latest_file)    

latest_cbb = latest_cbm_f()
model_sfx = '_' + latest_cbb.lstrip(os.path.join(os.getcwd(), '..', 'cb_models')).rstrip('.cbm')
pred_col = 'pred' + model_sfx
main_table = 'car_test'
main_table_o = 'car_test_outliers'

with engine.connect() as conn:
    df_vehicles = pd.read_sql(main_table, conn)     
    #print(df_vehicles.head())
#df_vehicles = pd.read_sql(main_table, engine)
df_vehicles.to_csv(os.path.join(dir_path, 'df_vehicles.csv'), index=False)

interval = 3

