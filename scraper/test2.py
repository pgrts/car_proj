from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np

from datetime import date, datetime
import time 

from sqlalchemy import create_engine, text
import psycopg2

import json
import requests
from bs4 import BeautifulSoup
import re
import os
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file
db_password = os.getenv('DB_PASSWORD')


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

def rmse(df, pred='pred', actual='price'):
    return int(root_mean_squared_error(df[pred], df[actual]))

def latest_cbm_files():
    return dict(zip([os.path.join(os.path.join(os.getcwd(), '..', 'cb_models'), file) for file in os.listdir(os.path.join(os.getcwd(), '..', 'cb_models')) if os.path.isfile(os.path.join(os.path.join(os.getcwd(), '..', 'cb_models'), file))], ['pred_' + x.lstrip('cb_model_').rstrip('.cbm') for x in os.listdir(os.path.join(os.getcwd(), '..', 'cb_models'))]))

def latest_cbm_f():
    
    cbm_files =  os.listdir(os.path.join(os.getcwd(), '..', 'cb_models'))
    cbm_dates = [x.lstrip('cb_model_').rstrip('.cbm') for x in cbm_files]
    latest_file = [x for x in cbm_files if max(cbm_dates) in x][0]

    return os.path.join(os.getcwd(), '..', 'cb_models', latest_file)    
   
def model_prep(df2):
    df2[cats] = df2[cats].astype(str)
    df2[nums] = df2[nums].astype('float64')
    return df2

pred_cols = list(latest_cbm_files().values())
pred_col = list(latest_cbm_files().values())[-1]
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
   
engine = create_engine(f'postgresql+psycopg2://postgres:{db_password}@localhost:5432/cars')
db_path = os.path.abspath('../flask_app/data/car_db.db')  # Adjust '../' if more levels are needed
seql_engine = create_engine(f'sqlite:///{db_path}')

main_table = 'car_test'
pred_col = list(latest_cbm_files().values())[-1]

interval = 2

listings_query = text(f'''

    WITH latest_data AS (WITH ranked_data AS (
        SELECT *, 
            ROW_NUMBER() OVER (
                PARTITION BY vin, odometer 
                ORDER BY posting_date DESC
            ) AS row_num
        FROM {main_table}
    )
    SELECT *
    FROM ranked_data
    WHERE row_num = 1)

    SELECT odometer, bodyclass, drivetype, enginecylinders, modelyear, series, trim, posting_date, make, model, price, link, 
            {pred_col}  as predicted_price, 
            (({pred_col} - price)) as residual, state, region
    FROM latest_data
    WHERE posting_date >= CURRENT_DATE - INTERVAL '{interval} days'

    ORDER BY residual DESC
''')
price_change_query = text(f'''SELECT DISTINCT 
        n.state,
        n.odometer as "new_odometer",
        b.odometer as "old_odometer",
        n.make, 
        n.model, 
        n.modelyear, 
        n.{pred_col} as "predicted_price",
        -- Ensure that new_price is always the later price (higher posting_date)
        CASE 
            WHEN n.posting_date > b.posting_date THEN n.price 
            ELSE b.price
        END AS "new_price", 
        -- Ensure that old_price is always the earlier price (older posting_date)
        CASE 
            WHEN n.posting_date > b.posting_date THEN b.price
            ELSE n.price
        END AS "old_price", 
        -- Ensure that new_posting_date is the later posting_date
        CASE 
            WHEN n.posting_date > b.posting_date THEN n.posting_date
            ELSE b.posting_date
        END AS "new_posting_date", 
        -- Ensure that old_posting_date is the earlier posting_date
        CASE 
            WHEN n.posting_date > b.posting_date THEN b.posting_date
            ELSE n.posting_date
        END AS "old_posting_date",
        n.vin, 
        n.link AS "new_link", 
        b.link AS "old_link",
        n.date_scraped AS "new_date_scraped",
        b.date_scraped AS "old_date_scraped",
        -- Adjust price_drop calculation based on the ordering of price and dates
        CASE 
            WHEN n.date_scraped > b.date_scraped THEN n.price - b.price
            WHEN b.date_scraped > n.date_scraped THEN b.price - n.price
            ELSE 0
        END AS price_drop,
        n.series, 
        n.trim,
        n.drivetype,
        n.bodyclass,
        n.enginecylinders
    FROM {main_table} n
    JOIN {main_table} b ON n.vin = b.vin
    WHERE n.price != b.price
    AND n.posting_date > b.posting_date
    AND LENGTH (n.vin) = 17
    ORDER BY "new_posting_date" DESC
    ''')
        
with engine.connect() as conn:
    price_df = pd.read_sql(price_change_query, conn)
    car_test_df = pd.read_sql_query(text(f"select * FROM car_test"), conn)[cats+nums+['vin']]
    current_unique_veh = pd.read_sql_query(text(f"select * FROM unique_vehicles_current"), conn)[cats+nums+['vin']]
    list_df = pd.read_sql(listings_query, conn)


vin_cols = [x for x in car_test_df.columns if x not in
 ['drive', 'link', 'price', 'odometer', 'days_since', 'state', 'region', 'state_income', 'condition', 
  'paint_color', 'title', 'link', 'location', 'drive', 'type', 'title_status', 'transmission', 'fuel', 'region_url', 'geo_placename',
  'reference_date', 'date_scraped', 'vin', 'posting_date', 'car_id'] + pred_cols]

distinct_veh = car_test_df.drop_duplicates(subset=vin_cols)[cats+nums+['vin']]

distinct_veh2 = pd.concat([car_test_df[cats+nums+['vin']], distinct_veh]).drop_duplicates(subset=vin_cols)[cats+nums+['vin']]

for num in nums:
    distinct_veh2[num] = distinct_veh2[num].apply(
        lambda x: str(int(x)) if pd.notnull(x) and x.is_integer() else str(x)
    )

for num in nums:
    if num in price_df.columns:
        price_df[num] = price_df[num].apply(
            lambda x: str(int(x)) if pd.notnull(x) and x.is_integer() else str(x)
        )
    if num in list_df.columns:
        list_df[num] = list_df[num].apply(
            lambda x: str(int(x)) if pd.notnull(x) and x.is_integer() else str(x)
        )        
 
price_df['price_drop'] = price_df['price_drop']*-1

price_df['new_posting_date'] = pd.to_datetime(price_df['new_posting_date']).dt.date
price_df['old_posting_date'] = pd.to_datetime(price_df['old_posting_date']).dt.date

price_df = price_df.drop_duplicates(subset=['state', 'vin', 'new_price'])


with seql_engine.connect() as conn:
    # Execute the SELECT query to count the rows in the table
    result = conn.execute(text("SELECT COUNT(*) FROM unique_vehicles"))
    row = result.fetchone()  # Fetch the first row of the result
    print(f"Number of rows in unique_vehicles: {row[0]}")

    result = conn.execute(text("SELECT COUNT(*) FROM price_changes"))
    row = result.fetchone()  # Fetch the first row of the result
    print(f"Number of rows in price_changes: {row[0]}")

    result = conn.execute(text("SELECT COUNT(*) FROM latest_listings"))
    row = result.fetchone()  # Fetch the first row of the result
    print(f"Number of rows in latest_listings: {row[0]}")        

distinct_veh2.to_sql('unique_vehicles', seql_engine, index=False, if_exists='replace')
list_df.to_sql('latest_listings', seql_engine, index=False, if_exists='replace')
price_df.to_sql('price_changes', seql_engine, index=False, if_exists='replace')

with seql_engine.connect() as conn:
    # Execute the SELECT query to count the rows in the table
    result = conn.execute(text("SELECT COUNT(*) FROM unique_vehicles"))
    row = result.fetchone()  # Fetch the first row of the result
    print(f"Number of rows in unique_vehicles: {row[0]}")

    result = conn.execute(text("SELECT COUNT(*) FROM price_changes"))
    row = result.fetchone()  # Fetch the first row of the result
    print(f"Number of rows in price_changes: {row[0]}")

    result = conn.execute(text("SELECT COUNT(*) FROM latest_listings"))
    row = result.fetchone()  # Fetch the first row of the result
    print(f"Number of rows in latest_listings: {row[0]}")
 