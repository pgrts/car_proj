import pandas as pd
import numpy as np
import subprocess
from datetime import date, datetime
import time 

from catboost import CatBoostRegressor
from sqlalchemy import create_engine, text, inspect, MetaData, Table
import logging

import os
import zipfile
import json
import requests

from sklearn.metrics import root_mean_squared_error
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()  
db_password = os.getenv('DB_PASSWORD')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

cats = [x.lower() for x in ['ABS', 'Trim2', 'ESC', 'SteeringLocation', 'BatteryInfo', 'DaytimeRunningLight', 'PedestrianAutomaticEmergencyBraking', 'TransmissionStyle', 
'WheelBaseType', 'Trim', 'ChargerLevel', 'AutomaticPedestrianAlertingSound', 'TractionControl', 'AirBagLocFront', 'Pretensioner', 'TransmissionSpeeds', 'AdaptiveDrivingBeam',
 'Model', 'BlindSpotMon', 'EntertainmentSystem', 'BodyCabType', 'FuelTypeSecondary', 'LaneDepartureWarning', 'TPMS', 'Seats', 'FuelInjectionType', 'EDR', 'LowerBeamHeadlampLightSource', 
 'ParkAssist', 'AirBagLocCurtain', 'RearAutomaticEmergencyBraking', 'RearCrossTrafficAlert', 'SemiautomaticHeadlampBeamSwitching', 'CIB', 'AirBagLocSide', 'BrakeSystemDesc', 'KeylessIgnition',
  'EngineConfiguration', 'AirBagLocKnee', 'RearVisibilitySystem', 'VehicleType', 'AdaptiveCruiseControl', 'AirBagLocSeatCushion', 'BlindSpotIntervention', 'ForwardCollisionWarning', 
  'SeatRows', 'BatteryType', 'LaneKeepSystem', 'GVWR', 'ElectrificationLevel', 'DynamicBrakeSupport', 'LaneCenteringAssistance', 'BedType', 'BrakeSystemType', 'Series2', 'CoolingType', 
  'Doors', 'EngineCylinders', 'CAN_AACN', 'Turbo', 'BodyClass', 'DriveType', 'ValveTrainDesign', 'FuelTypePrimary', 'Make', 'AutoReverseSystem', 'EVDriveUnit', 'Series', 'SeatBeltsAll', 
  'PlantCity', 'PlantCountry', 'PlantState', 'Note', 'OtherEngineInfo', 'GVWR_to', 'EngineModel', 'DestinationMarket', 'ActiveSafetySysNote', 'state', 'region', 'condition', 'paint_color']]

nums = [x.lower() for x in ['ModelYear', 'WheelSizeRear', 'BasePrice', 'WheelSizeFront', 'CurbWeightLB', 'WheelBaseShort', 'WheelBaseLong', 'BatteryPacks', 'SAEAutomationLevel', 'odometer', 
'EngineHP', 'TopSpeedMPH', 'TrackWidth', 'ChargerPowerKW', 'EngineKW', 'EngineHP_to', 'BatteryKWh', 'BedLengthIN', 'BatteryV', 'DisplacementCC', 'Wheels', 'Windows', 'days_since', 'state_income']]

def df_to_table(df, final_table_name, engine, exist = 'replace'):
    with engine.connect() as connection:
        df.to_sql(final_table_name, con=connection, if_exists=exist, index=False)
        print(f"Table '{final_table_name}' created successfully using SQLAlchemy.")
        
def scrape_regions(df):
    dfls = []
    
    for region_url in df.region_url.unique():
        try:
            # Log each region being scraped
            logging.info(f'Scraping region: {region_url}')
            link = region_url + '/search/cta?auto_title_status=1&bundleDuplicates=1&query=vin#search=1~gallery~0~0'
            
            # Ensure the df_from_link function is working as expected
            tdf = df_from_link(link)
            
            # Check if the dataframe is empty after scraping
            if tdf.empty:
                logging.warning(f'No data found for region: {region_url}')
            else:
                logging.info(f'Scraped {len(tdf)} rows from {region_url}')
            
            # Append region data to dfls
            tdf['region_url'] = region_url
            dfls.append(tdf)
            
            # Sleep to prevent being flagged for scraping too quickly
            time.sleep(1)
        
        except Exception as e:
            logging.error(f'Error scraping {region_url}: {e}')
    
    return dfls

def clean_mask_price(df, mini, maxi):
    # Clean the 'price' column by removing commas and dollar signs, then convert to numeric
    try:
        df['price'] = df['price'].astype(str).str.replace(',', '').str.replace('$', '').astype(int)
    except:
        print('error with casting initial scrape price column to integer')
        return None, None
    
    price_mask = (df['price'] > mini) & (df['price'] < maxi)

    return df[price_mask], df[~price_mask]  

def df_from_link(link):
    response=requests.get(link)

    soup = BeautifulSoup(response.text, 'html.parser')

    # Finding all listing elements with class 'cl-static-search-result'
    listings = soup.find_all('li', class_='cl-static-search-result')

    # Preparing lists to store the data
    links = []
    prices = []
    locations = []

    # Loop through each listing and extract the required data
    for listing in listings:
        # Extracting link
        link_tag = listing.find('a', href=True)
        link = link_tag['href'] if link_tag else 'No link'

        # Extracting price
        price_tag = listing.find('div', class_='price')
        price = price_tag.text.strip() if price_tag else 'No price'

        # Extracting location
        location_tag = listing.find('div', class_='location')
        location = location_tag.text.strip() if location_tag else 'No location'

        # Append data to lists
        links.append(link)
        prices.append(price)
        locations.append(location)

    # Creating a DataFrame to store the extracted data
    dd = pd.DataFrame({
        'link': links,
        'price': prices,
        'location': locations
    })

    return dd

def table_exists(table_name, connection):
    inspector = inspect(connection)
    tables = inspector.get_table_names()  # Get list of table names in the current schema
    return table_name in tables

def table_exists_engine(table_name, engine):
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()

def get_table_schema(table_name, engine):
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    schema = {col['name']: col['type'] for col in columns}
    return schema

def compare_table_schemas(table1, table2, engine):
    # Get schemas for both tables
    table1_schema = get_table_schema(table1, engine)
    table2_schema = get_table_schema(table2, engine)
    
    # Extract column sets and initialize flags
    table1_columns = set(table1_schema.keys())
    table2_columns = set(table2_schema.keys())
    all_columns_match = True
    
    # Compare column names
    if table1_columns != table2_columns:
        missing_in_table1 = table2_columns - table1_columns
        missing_in_table2 = table1_columns - table2_columns
        
        if missing_in_table1:
            print(f"Columns in {table2} but missing in {table1}: {missing_in_table1}", flush=True)
        if missing_in_table2:
            print(f"Columns in {table1} but missing in {table2}: {missing_in_table2}", flush=True)
        all_columns_match = False
    else:
        print("Column names match between tables.", flush=True)
    
    # Compare column types and any additional schema differences
    for column in table1_columns.intersection(table2_columns):
        table1_type = str(table1_schema[column])
        table2_type = str(table2_schema[column])
        
        if table1_type != table2_type:
            all_columns_match = False
            print(f"Type mismatch for column '{column}': {table1} type = '{table1_type}', {table2} type = '{table2_type}'", flush=True)
        else:
            print(f"Column '{column}' type matches: '{table1_type}'", flush=True)
    
    # Final status
    if all_columns_match and table1_columns == table2_columns:
        print("Schemas match exactly!", flush=True)
        return True
    else:
        print("Schemas DO NOT MATCH EXACTLY!!", flush=True)
        return False

def batch_vin(vin_input):
    url = 'https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/'
    post_fields = {'format': 'json', 'data': vin_input}
    r = requests.post(url, data=post_fields)
    vin_return = json.loads(r.text)
    return pd.DataFrame(vin_return['Results'])

def clean_vin_output(df):
    # Replace empty strings with 'nan' (if you want actual NaNs, use np.nan instead of 'nan')
    df = df.replace('', 'nan')
    #df = df.rename(columns={'vin': 'vin'})
    
    # Separate rows with invalid ErrorCodes, or nulls in key columns, into df_bad
    df_bad = df[~df['ErrorCode'].isin(['0', '1', '6']) | df[['Make', 'Model', 'ModelYear', 'VIN']].isnull().any(axis=1)]
    df_good = df[df['ErrorCode'].isin(['0', '1', '6'])].dropna(subset=['Make', 'Model', 'ModelYear', 'VIN'])

    # Define the value filter for acceptable vehicle types and body classes
    value_filter = (
        df_good['VehicleType'].isin(['TRUCK', 'MULTIPURPOSE PASSENGER VEHICLE (MPV)', 'PASSENGER CAR']) &
        df_good['BodyClass'].isin(['Pickup', 'Sport Utility Vehicle (SUV)/Multi-Purpose Vehicle (MPV)',
                                   'Crossover Utility Vehicle (CUV)', 'Sedan/Saloon',
                                   'Hatchback/Liftback/Notchback', 'Coupe', 'Convertible/Cabriolet',
                                   'Minivan', 'Wagon', 'Cargo Van', 'Van'])
    )

    # Apply the value filter, adding rows that don't meet it to df_bad
    df2 = df_good[value_filter]
    df_bad = pd.concat([df_bad, df_good[~value_filter]], ignore_index=True)
    df_bad.columns = df_bad.columns.str.lower()
    df2.columns = df2.columns.str.lower()
    return df2, df_bad

def process_vin_batch(vin_batch, engine, vins_accepted, vins_rejected, datestr):
    # Decode batch and clean
    vin_df = batch_vin(';'.join(vin_batch))
    valid_vin_df, reject_vin_df = clean_vin_output(vin_df)
    
    with engine.connect() as conn:
        # Store valid VINs in database
        if not valid_vin_df.empty:
            valid_vin_df['date_scraped'] = datestr
            valid_vin_df.to_sql(vins_accepted, engine, if_exists='append', index=False)
            print(f"Added {len(valid_vin_df)} records to {vins_accepted}")

        # Store rejected VINs in database with scrape date
        if not reject_vin_df.empty:
            reject_vin_df['date_scraped'] = datestr
            reject_vin_df.to_sql(vins_rejected, engine, if_exists='append', index=False)
            print(f"Added {len(reject_vin_df)} records to {vins_rejected}") 

def link_parser(link):
    response=requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Dictionary to hold the data
    data = {}

    # Define expected fields (labels) and initialize all as None
    fields = ['VIN:', 'condition:', 'drive:', 'fuel:', 'odometer:', 'paint color:', 'title status:', 'transmission:', 'type:']
    data = {field: None for field in fields}

    title_tag = soup.find('title')
        
    if title_tag:
        title = title_tag.text.strip()
        if title == 'blocked':
            return None
    else:
        title = None  # Just in case no title tag is found (shouldn't happen)
        
    time_tag = soup.find('time', class_='date timeago')
    if time_tag and 'datetime' in time_tag.attrs:
        posting_date_str = time_tag['datetime']
        try:
            # Use strptime for ISO format: "2024-10-18T17:06:07-0500"
            posting_date = datetime.strptime(posting_date_str, '%Y-%m-%dT%H:%M:%S%z')
        except ValueError:
            posting_date = None
        data['posting_date'] = posting_date
    else:
        data['posting_date'] = None
        
    geo_position_tag = soup.find('meta', attrs={'name': 'geo.position'})
    if geo_position_tag:
        geo_position = geo_position_tag.get('content', None)
        if geo_position:
            lat, long = geo_position.split(';')
            data['lat'] = lat
            data['long'] = long
    else:
        data['lat'] = None
        data['long'] = None

    geo_placename_tag = soup.find('meta', attrs={'name': 'geo.placename'})
    if geo_placename_tag:
        data['geo_placename'] = geo_placename_tag.get('content', None)
    else:
        data['geo_placename'] = None

    geo_region_tag = soup.find('meta', attrs={'name': 'geo.region'})
    if geo_region_tag:
        data['geo_region'] = geo_region_tag.get('content', None)
    else:
        data['geo_region'] = None

    # Finding only the relevant divs with class "attrgroup"
    attr_groups = soup.find_all('div', class_='attrgroup')

    # Loop through each attrgroup and extract 'labl' and 'valu'
    for group in attr_groups:
        attrs = group.find_all('div', class_='attr')  # Find individual attributes in the group
        for attr in attrs:
            labl_tag = attr.find('span', class_='labl')
            valu_tag = attr.find('span', class_='valu')

            # Check if both labl and valu are present
            if labl_tag and valu_tag:
                labl = labl_tag.text.strip()  # Keep the original labl with the colon
                valu = valu_tag.text.strip()

                # Ensure we're only storing known fields
                if labl in data:
                    data[labl] = valu
                    
    data['title'] = title
    data['link'] = link
    
    ff = pd.DataFrame([data])
    return ff

def clean_listing_output(df2, datestr):
    # Column renaming dictionary
    column_rename_dc = {
        'VIN:': 'vin', 'condition:': 'condition', 'drive:': 'drive', 'fuel:': 'fuel',
        'paint color:': 'paint_color', 'type:': 'type', 'transmission:': 'transmission',
        'title status:': 'title_status', 'odometer:': 'odometer'
    }


    # Rename and clean DataFrame
    df = df2.rename(columns=column_rename_dc).reset_index(drop=True)
    df = df.replace('', 'nan')

    if 'odometer' in df and df['odometer'].dtype == 'object':
        # Replace commas and handle non-numeric values
        df['odometer'] = df['odometer'].str.replace(',', '', regex=False)  # Remove commas
        
        # Convert to numeric, forcing errors to NaN (useful for invalid or missing values)
        df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')

        # Optionally, you can fill NaN values with a default value (e.g., -1 or 0) if needed
        df['odometer'] = df['odometer'].fillna(-1).astype(int)  # Replace NaNs with -1 and convert to int

    # Truncate VIN to 16 characters if longer, else mark for rejection if less than 16
    df['vin'] = df['vin'].str[:17]
    
    df['date_scraped'] = datestr

    # Filter listings with valid VIN and odometer
    valid_df = df[(df['vin'].notnull()) & 
                  (df['vin'].str.len() == 17) & 
                  (df['odometer'].between(15000, 400000))]
    reject_df = df[~df.index.isin(valid_df.index)]  # Entries that don't meet criteria

    return valid_df, reject_df

def posting_date(df):
    #df['posting_date'] = pd.to_datetime(df['posting_date']).dt.tz_localize(None)
    df['posting_date'] = pd.to_datetime(df['posting_date'], utc=True).dt.tz_localize(None)
    reference_date = '2021-01-01'
    df['days_since'] = (df['posting_date'] - reference_date).dt.days
    df['reference_date'] = reference_date
    return df

def find_price_diffs(scraped_links, big_rej_df):
    merged_df = pd.merge(scraped_links, big_rej_df, on='link', suffixes=('_scraped', '_rejected'))

    # Filter rows where prices are different
    price_diff_df = merged_df[merged_df['price_scraped'] != merged_df['price_rejected']]

    # Get the indices from scraped_links where prices are different
    return price_diff_df.index    

# use match_col to identify matches on that column BUT find price changes. rejects if price is the same AND vin (or link or match_col) is the same. 
def filter_and_reject(valid_df, reject_df, data_df, match_col='vin', use_odo=False):

    if use_odo==True:
        all_cols = [match_col, 'price', 'odometer']
    else:
        all_cols = [match_col, 'price']
    # Check for exact matches in 'vin' and 'price' between valid_df and data_df
    matches = valid_df.merge(
        data_df[all_cols], 
        on=all_cols, 
        how='inner'
    )
    
    # Remove matching rows from valid_df
    valid_df = valid_df[~valid_df[match_col].isin(matches[match_col])]
    
    # Add matching rows to reject_df
    reject_df = pd.concat([reject_df, matches], ignore_index=True)
    
    return valid_df, reject_df
  
def all_col_vals(tablename, conn, col= 'vin'):
    quer = text(f'SELECT DISTINCT {col} FROM {tablename}')
    return pd.read_sql(quer, conn)[col].tolist()
def latest_prices(new_data):

    new_data = new_data.sort_values(by='posting_date', ascending=False)
    
    # Drop duplicates based on 'vin' and 'odometer', keeping the first (newest) posting_date
    latest_data = new_data.drop_duplicates(subset=['vin', 'odometer'], keep='first')

    old_data = new_data[~new_data.index.isin(latest_data.index)]

    return latest_data, old_data

# {'cb_model_2024_11_15.cbm': 'pred_2024_11_15', ....}
def latest_cbm_files():
    return dict(zip([os.path.join(os.path.join(os.getcwd(), '..', 'cb_models'), file) for file in os.listdir(os.path.join(os.getcwd(), '..', 'cb_models')) if os.path.isfile(os.path.join(os.path.join(os.getcwd(), '..', 'cb_models'), file))], ['pred_' + x.lstrip('cb_model_').rstrip('.cbm') for x in os.listdir(os.path.join(os.getcwd(), '..', 'cb_models'))]))

# WHEN WE RETRAIN MODEL 
# Alter Table: Add pred_col
# Resave / Overwrite main_data / whatever schema we use to use the new column

pred_cols = list(latest_cbm_files().values())
mod_ints = ['price', 'odometer', 'modelyear', 'state_income', 'days_since'] + pred_cols

mod_texts = ['dynamicbrakesupport',
 'edr',
 'esc',
 'evdriveunit',
 'electrificationlevel',
 'engineconfiguration',
 'valvetraindesign',
 'vehicletype',
 'state',
 'enginemodel',
 'entertainmentsystem',
 'forwardcollisionwarning',
 'fuelinjectiontype',
 'fueltypeprimary',
 'fueltypesecondary',
 'region',
 'wheelbasetype',
 'gvwr',
 'gvwr_to',
 'keylessignition',
 'lanecenteringassistance',
 'lanedeparturewarning',
 'lanekeepsystem',
 'lowerbeamheadlamplightsource',
 'make',
 'model',
 'condition',
 'paint_color',
 'note',
 'otherengineinfo',
 'parkassist',
 'pedestrianautomaticemergencybraking',
 'plantcity',
 'plantcountry',
 'plantstate',
 'pretensioner',
 'rearautomaticemergencybraking',
 'rearcrosstrafficalert',
 'rearvisibilitysystem',
 'abs',
 'activesafetysysnote',
 'adaptivecruisecontrol',
 'adaptivedrivingbeam',
 'airbagloccurtain',
 'airbaglocfront',
 'airbaglocknee',
 'airbaglocseatcushion',
 'airbaglocside',
 'autoreversesystem',
 'automaticpedestrianalertingsound',
 'seatbeltsall',
 'semiautomaticheadlampbeamswitching',
 'series',
 'batteryinfo',
 'series2',
 'steeringlocation',
 'tpms',
 'batterytype',
 'tractioncontrol',
 'bedtype',
 'blindspotintervention',
 'blindspotmon',
 'bodycabtype',
 'bodyclass',
 'brakesystemdesc',
 'brakesystemtype',
 'can_aacn',
 'cib',
 'chargerlevel',
 'coolingtype',
 'daytimerunninglight',
 'destinationmarket',
 'transmissionstyle',
 'trim',
 'trim2',
 'drivetype',
 'turbo',
 'title',
 'link'] + ['location', 'drive', 'type', 'title_status', 'transmission', 'fuel', 'region_url', 'geo_placename', 'vin']
mod_floats = ['trackwidth',
 'baseprice',
 'batterykwh',
 'displacementcc',
 'enginehp',
 'enginehp_to',
 'enginekw',
 'wheelbaselong',
 'wheelbaseshort',
 'seats',
 'seatrows',
 'transmissionspeeds',
 'enginecylinders',
 'batterypacks',
 'batteryv',
 'bedlengthin',
 'chargerpowerkw',
 'curbweightlb',
 'saeautomationlevel',
 'topspeedmph',
 'wheelsizefront',
 'wheelsizerear',
 'wheels',
 'windows',
 'doors']
mod_dts = ['reference_date', 'date_scraped', 'posting_date']

def retrain_model(main_data, main_data_outliers, user='postgres', password=db_password, host='localhost', port='5432', db_name='cars'):

    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}')

    with engine.connect() as conn:
        df = pd.read_sql_query(text(f"select * FROM car_test UNION select * from car_test_outliers"), conn)
        
    good_index = set(df.index)
    bad_index = set()

    for mod_file, col in latest_cbm_files().items():

        if col not in df.columns:
            cbm = CatBoostRegressor()
            cbm.load_model(mod_file)
            df[col] = cbm.predict(model_prep(df[cats+nums])).round().astype(int)
            df['error'] = df[col] - df['price']
            df['error_percent'] = df['error'] / df['price']


            with engine.connect() as conn:
                for table in [main_data, main_data_outliers, 'schema_example']:
                    scheme_query = text(f'''ALTER TABLE {table} ADD COLUMN {col} BIGINT; ''')
                    conn.execute(scheme_query)
                    conn.commit()
            
    epm_msk = epm(df)
    abse_mask = abse(df)

    combined_mask = epm_msk | abse_mask

    # Get indices of outliers
    outlier_indices = df[combined_mask].index

    # Update bad_index and good_index
    bad_index.update(outlier_indices)
    good_index.difference_update(outlier_indices)

    # Create DataFrames for good and bad rows
    good_df = df.loc[good_index]
    bad_df = df.loc[bad_index]

    backup_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'table_backups', col))
    print(f"Backup directory: {backup_dir}")
    os.makedirs(backup_dir, exist_ok=True)  # Ensure backup directory exists

    with engine.connect() as conn:
        for table, df in {main_data: good_df, main_data_outliers: bad_df}.items():
            backup_file = os.path.join(backup_dir, f"{table}.dump")
            print(f"Backing up table {table} to {backup_file}...")

            # Set the PGPASSWORD environment variable temporarily for pg_dump
            env = os.environ.copy()
            env['PGPASSWORD'] = password

            # Run pg_dump for each table
            subprocess.run([
                "pg_dump", "-U", user, "-h", host, "-p", port, "-d", db_name,
                "-t", table, "-Fc", "-f", backup_file
            ], check=True, env=env)

            print(f"Backup of table {table} completed.")
            
            prep_cd_sql(df, mod_ints, mod_floats, mod_texts).to_sql(table, engine, index=False, if_exists='replace')  

def abse(df, lb=-25000, ub=25000):
    return (df['error'] < lb) | (df['error'] > ub)

def epm(df, lb=-0.75, ub=1.5):
    return (df['error_percent'] < lb) | (df['error_percent'] > ub) 

def create_preds(df, model, pred_col):
    #load df, load model
    df[pred_col] = model.predict(model_prep(df[cats+nums])).round().astype(int)
    df['error'] = df[pred_col] - df['price']
    df['error_percent'] = df['error'] / df['price']
    return df
    
def divert_outliers(df):
    # Initialize good_index with all indices and bad_index as empty
    good_index = set(df.index)
    bad_index = set()

    for model_file, pred_col in latest_cbm_files().items():
        # Load the model
        cbb = CatBoostRegressor()
        cbb.load_model(model_file)
        
        # Generate predictions
        df = create_preds(df, cbb, pred_col)
        
        # Apply masks to identify outliers
        epm_msk = epm(df)
        abse_mask = abse(df)
        
        combined_mask = epm_msk | abse_mask
        
        # Get indices of outliers
        outlier_indices = df[combined_mask].index
        
        # Update bad_index and good_index
        bad_index.update(outlier_indices)
        good_index.difference_update(outlier_indices)
    
    # Create DataFrames for good and bad rows
    good_df = df.loc[good_index]
    bad_df = df.loc[bad_index]
    
    return good_df, bad_df

def prep_cd_sql(df, int_cols, float_cols, text_cols, dt_cols=['reference_date', 'date_scraped', 'posting_date']):

    #ii, ff, tt = remove_null_cols(null_cols, int_cols, float_cols, text_cols)
    df = df.replace({'None':np.nan, 'nan':np.nan})
    
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].astype('Int64')
        df[col] = df[col].replace(-1, np.nan)

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['reference_date', 'date_scraped']:
        df[col] = df[col].replace('None', np.nan)
        df[col] = pd.to_datetime(df[col])

    for col in text_cols:
        df[col] = df[col].replace('nan', np.nan)
        df[col] = df[col].replace('Not Applicable', np.nan)
        df[col] = df[col].replace('None', np.nan)
        
    df['posting_date'] = pd.to_datetime(df['posting_date'])

    return df[int_cols+float_cols+text_cols+dt_cols]

def do_lots_stuff(main_data, datestr, engine, reg_ref = 'region_reference', big_rejects = 'big_rejects'):
                  
    #engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}')
    
    datestr_sql = '_' + datestr.replace('-', '_')

    backup_tablename = main_data + datestr_sql
    links_accepted = 'links_accepted' + datestr_sql
    links_rejected = 'links_rejected' + datestr_sql
    listings_accepted = 'listings_accepted' + datestr_sql    
    listings_rejected = 'listings_rejected' + datestr_sql
    vins_accepted = 'vins_accepted' + datestr_sql   
    vins_rejected = 'vins_rejected' + datestr_sql
    initial_scrape = 'scrape_results' + datestr_sql

    backup_query = f"""
    CREATE TABLE "{backup_tablename}" AS
    TABLE "{main_data}";"""
    
    check_query = f"""
    SELECT EXISTS (
        SELECT 1 
        FROM information_schema.tables 
        WHERE table_name = '{backup_tablename}'
    );
    """

    try:
        with engine.connect() as conn:
            
            result = conn.execute(text(check_query)).fetchone()

            if result[0]:  # If the table exists
                print(f"Backup table '{backup_tablename}' already exists. Skipping creation.")
            else:
                # Proceed to create the backup table
                conn.execute(text(backup_query))
                conn.commit()
                print(f"Backup of '{main_data}' created as '{backup_tablename}'.")
                
            print(f"Reading data from table '{main_data}'...")

            main_query = text(f'''SELECT "link", "vin", "price", "odometer" FROM "{main_data}";''')
            rj_query = text(f'''SELECT "link", "vin", "price", "odometer" FROM "{big_rejects}";''')

            backup_df = pd.read_sql(main_query, conn)
            big_rej_df =  pd.read_sql(rj_query, conn)

            reg_ref_df = pd.read_sql(reg_ref, conn)      

            if table_exists_engine(initial_scrape, engine):
                print(f'{initial_scrape} exists. Skipping Scrape')
                full_scrape = pd.read_sql(initial_scrape, conn)
            else:
                print(f'starting scrape to {initial_scrape}')
                full_scrape = pd.concat(scrape_regions(reg_ref_df))
                if full_scrape.empty:
                    print(f'{initial_scrape} failed. empty df')
                    return 
                else:
                    print(f'{initial_scrape} created. scrape proceeds')
                    full_scrape.to_sql(initial_scrape, engine, index=False)

            if full_scrape.empty:
                print("Error: full_scrape is empty after initialization.")
                return 

            if table_exists_engine(links_accepted, engine):
                print(f'{links_accepted} exists. Skipping split')
            else:
                df1, df2 = clean_mask_price(full_scrape, 1000, 175000)

                if df1.empty or df2.empty:
                    print("Error: The DataFrame (df1 or df2) is empty.")
                    return False

                # get price repeats on same link = move to rejected. same link + diff price= keep
                try:
                    scraped_links, reject_links = filter_and_reject(df1, df2, backup_df[['link', 'price', 'odometer']], match_col='link')
                except:
                    print('filter adn reject fail')

                if scraped_links.empty or reject_links.empty:
                    print("Error: The DataFrame (scraped_links or reject_links) is empty.")
                    return False
                
                # remove already rejected links from scraped_links
                scraped_links_final = scraped_links[~scraped_links.link.isin(big_rej_df.link)]
                reject_links_final = pd.concat([reject_links, scraped_links[scraped_links.link.isin(big_rej_df.link)]])

                if scraped_links_final.empty or reject_links_final.empty:
                    print("Error: The DataFrame (scraped_links_final or reject_links_final) is empty.")
                    return False

                new_length = (len(scraped_links_final) + len(reject_links_final))
                old_length = (len(scraped_links) + len(reject_links))
                
                if new_length != old_length:
                    print(f'Error length mismatch when removing from big_rejects. new length: {new_length} old length: {old_length}')
                    return False

                scraped_links_final['date_scraped'] = datestr
                reject_links_final['date_scraped'] = datestr
                scraped_links_final.to_sql(links_accepted, engine, index=False, if_exists='replace')
                print(str(len(scraped_links_final)) + f' rows added to {links_accepted}')
                print(links_accepted + ' created')    

                reject_links_final.to_sql(links_rejected, engine, index=False, if_exists='replace')
                print(links_rejected + ' created')

            all_links = list(set(big_rej_df.link.unique().tolist()))   
            print('length of all rejected and in data links ' + str(len(all_links)))
            
            bool1 = compare_table_schemas('links_schema', links_accepted, engine)
            #print(bool1)
            bool2 = compare_table_schemas('links_schema', links_rejected, engine)
            #print(bool2)

            all_vins = big_rej_df['vin'].drop_duplicates().tolist()
            
            if not bool1 or not bool2:
                print(f'BOO {links_accepted} or {links_rejected} do not match links_schema')
                return False
                
            if not table_exists(listings_rejected, conn):
                conn.execute(text(f"""
                    CREATE TABLE {listings_rejected} AS TABLE "listings_schema" WITH NO DATA;
                """))
                conn.commit()
                print(f"Created the listing target table '{listings_rejected}'.")
            else:
                all_links.extend(all_col_vals(listings_rejected, conn, col='link'))
                all_vins.extend(all_col_vals(listings_rejected, conn, col='vin'))
                print(f"The listing target table '{listings_rejected}' already exists.")       

            if not table_exists(listings_accepted, conn):
                conn.execute(text(f"""
                    CREATE TABLE {listings_accepted} AS TABLE "listings_schema" WITH NO DATA;
                """))
                conn.commit()
                print(f"Created the listing target table '{listings_accepted}'.")
            else:
                all_links.extend(all_col_vals(listings_accepted, conn, col='link'))
                print(f"The listing target table '{listings_accepted}' already exists.")                                              
            
            # Check and create good VIN table if it doesn't exist
            if not table_exists(vins_accepted, conn):
                conn.execute(text(f"""
                    CREATE TABLE \"{vins_accepted}\" AS TABLE "vins_schema" WITH NO DATA;
                """))
                print(f"Created the good VIN table '{vins_accepted}'.")
                conn.commit()
            else:
                all_vins.extend(all_col_vals(vins_accepted, conn, col='vin'))
                print(f"The good VIN table '{vins_accepted}' already exists.")

            if not table_exists(vins_rejected, conn):
                conn.execute(text(f"""
                    CREATE TABLE \"{vins_rejected}\" AS TABLE "vins_schema" WITH NO DATA;
                """))
                print(f"Created the good VIN table '{vins_rejected}'.")
                conn.commit()
            else:
                all_vins.extend(all_col_vals(vins_rejected, conn, col='vin'))
                print(f"The good VIN table '{vins_rejected}' already exists.")                
    
            # TODO: Refactor, This could be done above using def all_col_vals with else:
            if table_exists_engine(listings_accepted, engine):
                list_acc_quer = text(f"SELECT link FROM {listings_accepted};")
                listing_links = pd.read_sql(list_acc_quer, conn)['link'].tolist()
                all_links.extend(listing_links)
            else:
                print('all links not extended ' + f'{listings_accepted} does not exist')
                return False

            if table_exists_engine(links_rejected, engine):
                link_rej_quer = text(f"SELECT link FROM {links_rejected};")
                r_links = pd.read_sql(link_rej_quer, conn)['link'].tolist()
                all_links.extend(r_links)
            else:
                print('all links not extended ' + f'{links_rejected} does not exist')
                return False

            if table_exists_engine(listings_rejected, engine):
                list_rej_quer = text(f"SELECT link FROM {listings_rejected};")
                r_listings = pd.read_sql(list_rej_quer, conn)['link'].tolist()
                all_links.extend(r_listings)
            else:
                print('all links not extended ' + f'{listings_rejected} does not exist')
                return False

            link_source_quer = text(f'SELECT DISTINCT link, price FROM {links_accepted}')
            link_source_df = pd.read_sql(link_source_quer, conn)
            remaining_link_df = link_source_df[~link_source_df['link'].isin(all_links)][['link', 'price']]
            print(f"Found {len(remaining_link_df)} remaining links to process.")

    except Exception as e:
        print(f"Error fetching links from tables: {e}")
        return False
      
    all_vins = list(set(all_vins))

    vin_batch = [] 
    for row in remaining_link_df[['link', 'price']].itertuples():

        link = row.link
        price_val = row.price

        try:
            parsed_df = link_parser(link)
            parsed_df['price'] = price_val

            if parsed_df is not None:
                df1, df2 = clean_listing_output(parsed_df, datestr)

                valid_df, reject_df = filter_and_reject(df1, df2, big_rej_df, match_col='vin')
                reject_df = reject_df.drop_duplicates(subset='vin')
  
                with engine.connect() as conn:
                    if not valid_df.empty:
                        valid_df = valid_df.drop(columns='price')
                        

                        vin = valid_df.at[0, 'vin']

                        if (vin not in vin_batch) and (vin not in all_vins):
                            valid_df.to_sql(listings_accepted, con=engine, if_exists='append', index=False)
                            vin_batch.append(vin)
                            print(f"VIN added to batch: {vin}")
                            print(f"Current batch size: {len(vin_batch)}")
                        else:
                            print(f"VIN already in database: {vin}, adding to {listings_rejected}")
                            valid_df.to_sql(listings_rejected, con=engine, if_exists='append', index=False)

                        # Process batch if full
                        if len(vin_batch) == 50:
                            all_vins.extend(vin_batch)
                            process_vin_batch(vin_batch, engine, vins_accepted, vins_rejected, datestr)
                            vin_batch.clear()

                    # Insert rejected listings
                    if not reject_df.empty:
                        reject_df = reject_df.drop(columns='price')
                        reject_df.to_sql(listings_rejected, con=engine, if_exists='append', index=False)
                        print(f"Added reject records to {listings_rejected}")

                print(f"Processed link: {link}")
            else:
                print(f"Link {link} returned blocked or empty data.")
        
        except Exception as e:
            print(f"Error processing {link}: {e}")
            return False
        finally:
            time.sleep(0.5)

    # Process any remaining vins in the batch after all links are processed
    if vin_batch:
        process_vin_batch(vin_batch, engine, vins_accepted, vins_rejected, datestr)

    return True

def reject_more_values(gd_links, bd_links, gd_listings, bd_listings, gd_vins, reg_ref, new_table_name, engine, 
    big_reg = "big_rejects", main_data = 'all_cars'):
    logging.basicConfig(level=logging.INFO)
    try:
        with engine.connect() as conn:

            # Step 1: Perform left joins and create the new table
            try:
                if not table_exists_engine(new_table_name, engine):  
                    join_to_new_table = text(f"""
                    CREATE TABLE {new_table_name} AS 
                    WITH combined_data AS (
                        SELECT 
                            l."condition", 
                            l."drive", 
                            l."fuel", 
                            COALESCE(l."odometer", -1) AS "odometer",  -- Keep odometer as integer
                            l."paint_color", 
                            l."title_status", 
                            l."transmission", 
                            l."type", 
                            l."posting_date", 
                            l."lat", 
                            l."long", 
                            l."geo_placename", 
                            l."geo_region", 
                            l."title", 
                            l."link", 
                            v.*, 
                            k.price, 
                            k.location, 
                            k.region_url, 
                            r.state, 
                            r.state_income, 
                            r.region
                        FROM {gd_vins} v
                        LEFT JOIN {gd_listings} l USING ("vin")
                        LEFT JOIN {gd_links} k USING ("link")
                        LEFT JOIN {reg_ref} r ON k."region_url" = r."region_url"
                    )
                    SELECT * FROM combined_data;
                    """)
                    conn.execute(join_to_new_table)
                    conn.commit()  # Commit after creating the new table
                    print(f"Created new table {new_table_name} and committed changes.")
                else:
                    print(f"Table {new_table_name} already exists. Skipping creation.")
            except Exception as e:
                
                logging.error(f"Error creating table {new_table_name}: {e}")
                return False
            
            try:
                move_to_big_reg_bd = text(f'''INSERT INTO {big_reg}(
                                        "odometer", 
                                        "posting_date", 
                                        "lat", 
                                        "long", 
                                        "geo_placename", 
                                        "geo_region", 
                                        "title", 
                                        "link", 
                                        "vin", 
                                        "date_scraped", 
                                        "price", 
                                        "location", 
                                        "region_url"
                                    )
                                       SELECT 
                    COALESCE(l."odometer", -1) AS "odometer",  -- Keep odometer as integer
                    l."posting_date", 
                    l."lat", 
                    l."long", 
                    l."geo_placename", 
                    l."geo_region", 
                    l."title", 
                    l."link", 
                    l."vin", 
                    l."date_scraped", 
                    k."price", 
                    k."location", 
                    k."region_url"
                FROM {bd_listings} l
                LEFT JOIN {bd_links} k USING ("link");''')
                conn.execute(move_to_big_reg_bd)
                conn.commit() 
                print(f"Moved rows from {bd_links} and {bd_listings} to {big_reg} and committed changes.")        
            except Exception as e:
                logging.error(f"Error moving {bd_links} and {bd_listings} to {big_reg}: {e}")
                return False 

            try:
                move_to_big_reg = text(f'''INSERT INTO {big_reg}(
                                        "odometer", 
                                        "posting_date", 
                                        "lat", 
                                        "long", 
                                        "geo_placename", 
                                        "geo_region", 
                                        "title", 
                                        "link", 
                                        "vin", 
                                        "date_scraped", 
                                        "price", 
                                        "location", 
                                        "region_url"
                                    )
                                       SELECT 
                    COALESCE(l."odometer", -1) AS "odometer",  -- Keep odometer as integer
                    l."posting_date", 
                    l."lat", 
                    l."long", 
                    l."geo_placename", 
                    l."geo_region", 
                    l."title", 
                    l."link", 
                    l."vin", 
                    l."date_scraped", 
                    k."price", 
                    k."location", 
                    k."region_url"
                FROM {gd_listings} l
                LEFT JOIN {gd_links} k USING ("link")
                WHERE l."link" NOT IN (SELECT "link" FROM {new_table_name});''')
                conn.execute(move_to_big_reg)
                conn.commit()  
                print(f"Moved rows from {gd_links} and {gd_listings} to {big_reg} and committed changes.")  
            except Exception as e:
                logging.error(f"Error moving {gd_links} and {gd_listings} rows to {big_reg}: {e}")
                return False  

            try:
                # Alter table and update additional columns
                conn.execute(text(f"""
                    ALTER TABLE {new_table_name}
                    ADD COLUMN IF NOT EXISTS days_since INTEGER,
                    ADD COLUMN IF NOT EXISTS reference_date TEXT;
                """))
                conn.commit()  # Commit after altering the table
                print(f"Altered table {new_table_name} to add new columns and committed changes.")

                conn.execute(text(f"""
                    UPDATE {new_table_name}
                    SET 
                        days_since = EXTRACT(DAY FROM (posting_date - DATE '2021-01-01')),
                        reference_date = '2021-01-01'
                    WHERE posting_date IS NOT NULL;
                """))
                conn.commit()  # Commit after updating the new table
                print(f"Updated table {new_table_name} with days_since and reference_date, and committed changes.")
            except Exception as e:
                logging.error(f"Error updating {new_table_name}: {e}")
                return False
            
            try:
                # Read from the source table
                query = text(f"""SELECT * FROM {new_table_name};""")
                df = pd.read_sql(query, conn)
                
                #print([x for x in df2.columns if x not in df.columns])
                gdf, bdf = divert_outliers(df)
                gdf = prep_cd_sql(gdf, mod_ints, mod_floats, mod_texts)
                bdf = prep_cd_sql(bdf, mod_ints, mod_floats, mod_texts)
                gdf[mod_ints+mod_floats+mod_texts+mod_dts].to_sql(main_data, engine, index=False, if_exists='append')
                bdf[mod_ints+mod_floats+mod_texts+mod_dts].to_sql((main_data + '_outliers'), engine,  index=False, if_exists='append')

                print(f"{len(gdf)} rows added from {new_table_name} to {main_data}.")
                print(f"{len(bdf)} rows added from {new_table_name} to {main_data}_outliers.")
            except Exception as e:
                # Log or print an error if the operation fails
                logging.error(f"Failed to append data to {main_data}: {e}")
                return False
        
    except Exception as e:
        logging.error(f"Error in reject_more_values function: {e}")
        return False
    print('reject_more_values function succeeded.')
    return True
    
def backup_and_cleanup_database(source_db_url, backup_db_url, subst='_2024_'):
    logging.basicConfig(level=logging.INFO)

    try:
        # Create engine for the source and backup databases
        source_engine = create_engine(source_db_url)
        backup_engine = create_engine(backup_db_url)

        # Reflect the tables in the source database
        metadata = MetaData()
        metadata.reflect(bind=source_engine)

        # Get the names of tables that contain the specified substring
        tables_to_backup = [table_name for table_name in metadata.tables if subst in table_name]

        if not tables_to_backup:
            logging.info("No tables found with the specified pattern.")
            return False
        else:
            print(tables_to_backup)

        # Start copying tables to the backup database
        with backup_engine.begin() as backup_conn:
            for table_name in tables_to_backup:
                # Get the table definition from the source database
                source_table = Table(table_name, metadata, autoload_with=source_engine)

                # Create the table in the backup database
                source_table.metadata.create_all(backup_engine, tables=[source_table])
                logging.info(f"Created table {table_name} in backup database.")

                # Copy data from source to backup
                data = pd.read_sql_table(table_name, source_engine)
                data.to_sql(table_name, backup_engine, if_exists='append', index=False)
                logging.info(f"Copied data to table {table_name} in backup database.")

                # Confirm successful backup before deleting from source
                if data.shape[0] > 0:  # Check if data was copied
                    with source_engine.begin() as source_conn:
                        # Execute the DROP TABLE command
                        source_conn.execute(text(f"DROP TABLE IF EXISTS {table_name};"))
                    logging.info(f"Deleted table {table_name} from source database after backup.")
                    print(f"Successfully deleted table {table_name} from source database.")

    except Exception as e:
        logging.error(f"An error occurred during the backup process: {e}")
        print(f"Error: {e}")
        return False
    
    return True

def backup_and_cleanup_database2(backup_db_url, backup_zip_path):
    logging.basicConfig(level=logging.INFO)

    # Extract the database name from the URL for the dump file
    db_name = backup_db_url.split('/')[-1]
    dump_file_path = f"{db_name}.sql"  # Dump file name

    # Extract username and password from the URL
    user = backup_db_url.split(':')[1].split('//')[1].split('@')[0]
    password = backup_db_url.split(':')[2].split('@')[0]

    # Set the PGPASSWORD environment variable
    os.environ['PGPASSWORD'] = password

    try:
        # Run pg_dump with explicit username
        subprocess.run([
            'pg_dump',
            '--username=' + user,  # Specify the username explicitly
            '--host=localhost',     # Specify the host explicitly
            '--port=5432',          # Specify the port explicitly
            '--dbname=' + db_name,  # Use just the database name
            '--file=' + dump_file_path,
            '--format=plain',       # Change to 'custom' if needed
            '--no-owner',           # Omit ownership information
            '--no-privileges'       # Omit privilege information
        ], check=True)

        logging.info(f"Successfully created backup: {dump_file_path}")

        # Compress the dump file into a ZIP file
        with zipfile.ZipFile(backup_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(dump_file_path, arcname=os.path.basename(dump_file_path))
            logging.info(f"Compressed backup database into {backup_zip_path}.")

        # If backup is successful, clear the original database
        source_engine = create_engine(backup_db_url)
        with source_engine.begin() as conn:
            # Drop all tables in the public schema
            conn.execute(text("DROP SCHEMA public CASCADE;"))
            conn.execute(text("CREATE SCHEMA public;"))
            logging.info("Cleared all tables from the original database.")

    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while backing up the database: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Clean up the dump file if you want to remove it after zipping
        if os.path.exists(dump_file_path):
            os.remove(dump_file_path)
            logging.info(f"Removed temporary dump file: {dump_file_path}")

def model_prep(df2):
    df2[cats] = df2[cats].astype(str)
    df2[nums] = df2[nums].astype('float64')
    return df2

def rmse(df, pred_col):
    return root_mean_squared_error(df[pred_col], df['price'])

def get_first(ls):
    if (len(ls) == 1):
        return ls[0]
    else: 
        return get_first(list(set(ls)))

def detect_down_payments(df, pred_col, mult = 1.5, add=1500):
    #(df.odometer < 250000) & (df['modelyear'] > 2000) & (df['price'] < 10000) & 
    down_payments_likely = (df[pred_col] > ((mult * df['price']) + add))
    return df[~down_payments_likely], df[down_payments_likely]
    
def dump_backup(substring, user='postgres', password=db_password, host='localhost', port='5432', db_name='cars', delete_after_backup=False):

    backup_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'table_backups',substring))
    print(f"Backup directory: {backup_dir}")
    os.makedirs(backup_dir, exist_ok=True)  # Ensure backup directory exists

    # Create a SQLAlchemy engine
    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}')

    # Query to find all tables with '{substring}' in their name
    with engine.connect() as connection:
        result = connection.execute(text("""
            SELECT tablename
            FROM pg_tables
            WHERE tablename LIKE :substring AND schemaname = 'public';
        """), {"substring": f"%{substring}%"})
        
        # Fetch all matching table names as a list
        tables = [row['tablename'] for row in result.mappings()]
        print("Tables to back up:", tables)

    for table in tables:
        backup_file = os.path.join(backup_dir, f"{table}.dump")
        print(f"Backing up table {table} to {backup_file}...")

        # Set the PGPASSWORD environment variable temporarily for pg_dump
        env = os.environ.copy()
        env['PGPASSWORD'] = password

        # Run pg_dump for each table
        subprocess.run([
            "pg_dump", "-U", user, "-h", host, "-p", port, "-d", db_name,
            "-t", table, "-Fc", "-f", backup_file
        ], check=True, env=env)

        print(f"Backup of table {table} completed.")

        # Delete the table after backup if delete_after_backup is set to True
        if delete_after_backup:
            with engine.connect() as connection:
                connection.execute(text(f'DROP TABLE IF EXISTS "{table}"'))
                connection.commit()  # Commit deletion
                print(f"Table {table} deleted after backup.")

    print("Backup process completed.")

datestr = '2024-11-31'
datestr_sql = '_' + datestr.replace('-', '_')
source_db_url = f'postgresql+psycopg2://postgres:{db_password}@localhost:5432/cars'
backup_db_url = f'postgresql://postgres:{db_password}@localhost:5432/cars_backup'
backup_zip_path = f'C:\\Users\\pgrts\\Desktop\\python\\car_proj\\scraper\\backup{datestr}.zip'
engine = create_engine(source_db_url)

main_data = 'car_test'


if do_lots_stuff(main_data, datestr, engine):
    print('initial scrape complete. tables created')
    if reject_more_values('links_accepted' + datestr_sql, 
        'links_rejected' + datestr_sql, 
        'listings_accepted' + datestr_sql, 
        'listings_rejected' + datestr_sql, 
        'vins_accepted' + datestr_sql, 
        'region_reference', 
        'new_data' + datestr_sql, engine,
            main_data = main_data):
        print('reject complete')
        dump_backup(substring=datestr_sql, delete_after_backup=True)
        print('backup dumped')

def restore_all_dumps(substring, user='postgres', password=db_password, host='localhost', port='5432', db_name='cars'):
    # Backup directory based on substring
    backup_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'table_backups', substring))
    print(f"Backup directory: {backup_dir}")

    # Check if directory exists
    if not os.path.exists(backup_dir):
        print(f"Backup directory does not exist: {backup_dir}")
        return

    # Get all .dump files in the directory
    dump_files = [f for f in os.listdir(backup_dir) if f.endswith('.dump')]

    if not dump_files:
        print("No .dump files found in the directory.")
        return

    # Set the PGPASSWORD environment variable for pg_restore
    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password

    # Iterate through all dump files and restore them
    for dump_file in dump_files:
        dump_path = os.path.join(backup_dir, dump_file)
        print(f"Restoring {dump_path}...")

        cmd = [
            "pg_restore",
            "-U", user,
            "-h", host,
            "-p", port,
            "-d", db_name,
            dump_path
        ]

        try:
            subprocess.run(cmd, check=True, env=env)
            print(f"Successfully restored {dump_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error restoring {dump_file}: {e}")

    print("Restoration process completed.")


'''    
backup_db_url = f'postgresql://postgres:{db_password}@localhost:5432/cars_backup'
backup_zip_path = f'C:\\Users\\pgrts\\Desktop\\python\\car_proj\\scraper\\backup{datestr}.zip'
if backup_and_cleanup_database(source_db_url, backup_db_url):
    backup_and_cleanup_database2(backup_db_url, backup_zip_path)

engine.dispose()  # Ensure that the engine is disposed even if an error occurs

if do_lots_stuff(main_data, datestr, engine):
    print('initial scrape complete. tables created')
    if reject_more_values('links_accepted' + datestr_sql, 
        'links_rejected' + datestr_sql, 
        'listings_accepted' + datestr_sql, 
        'listings_rejected' + datestr_sql, 
        'vins_accepted' + datestr_sql, 
        'region_reference', 
        'new_data' + datestr_sql, engine,
            main_data = main_data):
        print('reject complete')
        

''' 