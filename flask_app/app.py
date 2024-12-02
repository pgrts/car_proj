from flask import Flask, request, render_template, jsonify, g
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import io
import base64
import requests
import random
from sqlalchemy import create_engine
import os 
from datetime import date

app = Flask(__name__)

cats = [x.lower() for x in ['ABS',

 'Trim2',
 'ESC',
 'SteeringLocation',
 'BatteryInfo',
 'DaytimeRunningLight',
 'PedestrianAutomaticEmergencyBraking',
 'TransmissionStyle',
 'WheelBaseType',
 'Trim',
 'ChargerLevel',
 'AutomaticPedestrianAlertingSound',
 'TractionControl',
 'AirBagLocFront',
 'Pretensioner',
 'TransmissionSpeeds',
 'AdaptiveDrivingBeam',
 'Model',
 'BlindSpotMon',
 'EntertainmentSystem',
 'BodyCabType',
 'FuelTypeSecondary',
 'LaneDepartureWarning',
 'TPMS',
 'Seats',
 'FuelInjectionType',
 'EDR',
 'LowerBeamHeadlampLightSource',
 'ParkAssist',
 'AirBagLocCurtain',
 'RearAutomaticEmergencyBraking',
 'RearCrossTrafficAlert',
 'SemiautomaticHeadlampBeamSwitching',
 'CIB',
 'AirBagLocSide',
 'BrakeSystemDesc',
 'KeylessIgnition',
 'EngineConfiguration',
 'AirBagLocKnee',
 'RearVisibilitySystem',
 'VehicleType',
 'AdaptiveCruiseControl',
 'AirBagLocSeatCushion',
 'BlindSpotIntervention',
 'ForwardCollisionWarning',
 'SeatRows',
 'BatteryType',
 'LaneKeepSystem',
 'GVWR',
 'ElectrificationLevel',
 'DynamicBrakeSupport',
 'LaneCenteringAssistance',
 'BedType',
 'BrakeSystemType',
 'Series2',
 'CoolingType',
 'Doors',
 'EngineCylinders',
 'CAN_AACN',
 'Turbo',
 'BodyClass',
 'DriveType',
 'ValveTrainDesign',
 'FuelTypePrimary',
 'Make',
 'AutoReverseSystem',
 'EVDriveUnit',
 'Series',
 'SeatBeltsAll',
 'PlantCity',
 'PlantCountry',
 'PlantState',
 'Note',
 'OtherEngineInfo',
 'GVWR_to',
 'EngineModel',
 'DestinationMarket',
 'ActiveSafetySysNote',
 'state',
 'region',
 'condition',
'paint_color']]

nums =  [x.lower() for x in ['ModelYear',
 'WheelSizeRear',
 'BasePrice',
 'WheelSizeFront',
 'CurbWeightLB',
 'WheelBaseShort',
 'WheelBaseLong',
 'BatteryPacks',
 'SAEAutomationLevel',
 'odometer',
 'EngineHP',
 'TopSpeedMPH',
 'TrackWidth',
 'ChargerPowerKW',
 'EngineKW',
 'EngineHP_to',
 'BatteryKWh',
 'BedLengthIN',
 'BatteryV',
 'DisplacementCC',
 'Wheels',
 'Windows',
 'days_since',
 'state_income']]
    
def latest_cbm_f():
    
    cbm_files =  os.listdir(os.path.join(os.getcwd(), '..', 'cb_models'))
    cbm_dates = [x.lstrip('cb_model_').rstrip('.cbm') for x in cbm_files]
    latest_file = [x for x in cbm_files if max(cbm_dates) in x][0]

    return os.path.join(os.getcwd(), '..', 'cb_models', latest_file)    

def model_prep(df2):
    df2[cats] = df2[cats].astype(str)
    df2[nums] = df2[nums].astype('float64')
    return df2
    
# Load the dataframe of vehicle data for finding similar vehicles

today_date = date.today().strftime("%Y-%m-%d")
days_since_reference = (date.today() - date(2021, 1, 1)).days
unique_vehicles = 'unique_vehicles'
price_changes = 'price_changes'
latest_listings = 'latest_listings'

db_path = os.path.abspath('data/car_db.db')  # Adjust '../' if more levels are needed
seql_engine = create_engine(f'sqlite:///{db_path}')

#df_vehicles = model_prep(pd.read_sql(f'{unique_vehicles}', seql_engine)).replace({'None':'nan'})
df_vehicles = None
state_income_map = None

cb72 = CatBoostRegressor()
cb72.load_model('data/cb_model_2024_11_25.cbm')


@app.route('/vehicle_tool', methods=['GET'])
def vehicle_tool():
    global df_vehicles
    global state_income_map

    if df_vehicles is None:
        df_vehicles = model_prep(pd.read_sql(f'{unique_vehicles}', seql_engine)).replace({'None': 'nan'})

    if state_income_map is None:
        state_income_map = df_vehicles[["state", "state_income"]].drop_duplicates().set_index("state").to_dict()["state_income"]

    dropdown_options = {
        "make": sorted(df_vehicles["make"].unique().tolist()),
        "paint_color": sorted(df_vehicles["paint_color"].unique().tolist()),
        "condition": sorted(df_vehicles["condition"].unique().tolist()),
        "state": sorted(df_vehicles["state"].unique().tolist())
    }


    table_html = f'''
    <button id="addRow">Add Row</button>
    <table id="vehicleTable" border="1">
        <tr>
            <th>Clone</th><th>Remove</th><th>Make</th><th>Model</th><th>Year</th><th>Series</th><th>Trim</th><th>Engine Size</th><th>Engine Cylinders</th>
            <th>Fuel Type</th><th>DriveType</th><th>State</th><th>Region</th>
            <th>Condition</th><th>Paint Color</th><th>Days Since 01-01-2021<br><small>({today_date}: {days_since_reference} days)</small></th>
            <th>Odometer</th><th>Predicted Price</th><th>Predict</th>
        </tr>
    </table>
    '''
    # Return the table HTML and dropdown options as a JSON response
    return jsonify({
        'html': table_html,
        'dropdown_options': dropdown_options
    })

@app.route('/get-models', methods=['GET', 'POST'])
def get_models():
    global df_vehicles
    if request.method == 'POST':
        data = request.get_json()  # Ensure you are getting JSON data
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
        
        make = data.get('make')
        print(f'make: {make}')  # Should print the make if received correctly
        
        if not make:
            return jsonify({"error": "Make is required"}), 400

        models = df_vehicles[df_vehicles['make'] == make]['model'].unique().tolist()
        print(f'models: {models}')  # Should print the models

        return jsonify({"models": models})

@app.route('/get_model_year', methods=['POST'])
def get_model_year():
    global df_vehicles
    try:
        data = request.get_json()
        make = data['make']
        model = data['model']
        
        print(f'make: {make}, model: {model}')  # Debugging output
        
        # Filter the DataFrame for the given make and model
        filtered_df = df_vehicles[(df_vehicles['make'] == make) & (df_vehicles['model'] == model)]
        
        # Extract and sort modelyear values
        modelyear = filtered_df['modelyear'].apply(lambda x: int(x) if isinstance(x, (float, int)) and not pd.isna(x) else None).dropna().unique().tolist()
        modelyear.sort()  # Ensure the years are sorted numerically
        
        return jsonify({
            'modelyear': modelyear
        })
    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error
        return jsonify({"error": f"Failed to get model year data: {str(e)}"}), 500

@app.route('/get_model_extras', methods=['POST'])
def get_model_extras():
    global df_vehicles
    try:
        data = request.get_json()  # Ensure you are getting JSON data
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        make = data['make']
        model = data['model']
        year = data['modelyear']
        print(f'make: {make}, model: {model} year: {str(year)}')  # Debugging output
        
        # Filter the DataFrame
        filtered_df = df_vehicles[(df_vehicles['make'] == make) & (df_vehicles['model'] == model) & (df_vehicles['modelyear'] == float(year))]

        print(filtered_df[['series','trim', 'displacementcc', 'fueltypeprimary','drivetype']])
        
        if filtered_df.empty:
            return jsonify({"error": "No matching records found"}), 404

        # Convert values and handle NaN
        enginecylinders = filtered_df['enginecylinders'].unique().tolist()#.apply(lambda x: int(x) if isinstance(x, (float, int)) and not pd.isna(x) else 'nan').unique().tolist()


        # Extract unique values
        series = filtered_df['series'].unique().tolist()#.apply(lambda x: int(x) if isinstance(x, (float, int)) and not pd.isna(x) else 'None').unique().tolist()
        trim = filtered_df['trim'].unique().tolist()#.apply(lambda x: int(x) if isinstance(x, (float, int)) and not pd.isna(x) else 'None').unique().tolist()
        drivetype = filtered_df['drivetype'].unique().tolist()#.apply(lambda x: int(x) if isinstance(x, (float, int)) and not pd.isna(x) else 'None').unique().tolist()
        displacementcc = filtered_df['displacementcc'].apply(lambda x: int(x) if isinstance(x, (float, int)) and not pd.isna(x) else 'None').unique().tolist()
        fueltypeprimary = filtered_df['fueltypeprimary'].unique().tolist()#.apply(lambda x: int(x) if isinstance(x, (float, int)) and not pd.isna(x) else 'None').unique().tolist()
    
        print(
            f'enginecylinders: {enginecylinders}\n'
            f'series: {series}\n'
            f'trim: {trim}\n'
            f'drivetype: {drivetype}\n'
            f'displacementcc: {displacementcc}\n'
            f'fueltypeprimary: {fueltypeprimary}'
        )

        # Return the relevant data as a JSON response
        return jsonify({
            'series': series,
            'trim': trim,
            'drivetype': drivetype,
            'enginecylinders': enginecylinders,
            'displacementcc': displacementcc,
            'fueltypeprimary': fueltypeprimary
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error
        return jsonify({"error": str(e)}), 500

@app.route('/get_states_regions', methods=['POST'])
def get_states_regions():
    global df_vehicles
    try:
        data = request.get_json()  # Ensure you are getting JSON data
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        state = data['state']
        print(f'state: {state}')  # Debugging output
        
        # Simulate region lookup based on state
        regions = (df_vehicles.groupby('state')['region']
            .apply(lambda x: x.unique().tolist())
            .to_dict()
        )
        
        # Return the regions as a JSON response
        return jsonify(regions.get(state, []))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_price', methods=["POST"])
def predict_price():
    global df_vehicles
    global state_income_map

    make = request.form.get('make')
    model = request.form.get('model')
    modelyear = request.form.get('model_year')
    series = request.form.get('series')
    trim = request.form.get('trim')
    displacementcc = request.form.get('displacementcc')
    enginecylinders = request.form.get('enginecylinders')
    drivetype = request.form.get('drivetype')
    fueltypeprimary = request.form.get('fueltypeprimary')
    state = request.form.get('state')
    region = request.form.get('region')
    condition = request.form.get('condition')
    paint_color = request.form.get('paint_color')
    days_since = int(request.form.get('days_since', days_since_reference))  
    odometer = int(request.form.get('odometer', 100000)) 

    print(f"make: {make}, type: {type(make)}")
    print(f"model: {model}, type: {type(model)}")
    print(f"model_year: {modelyear}, type: {type(modelyear)}")
    print(f"series: {series}, type: {type(series)}")
    print(f"trim: {trim}, type: {type(trim)}")
    print(f"displ: {displacementcc}, type: {type(displacementcc)}")    
    print(f"engine_cylinders: {enginecylinders}, type: {type(enginecylinders)}")
    print(f"drive_type: {drivetype}, type: {type(drivetype)}")
    print(f"fuel_type_primary: {fueltypeprimary}, type: {type(fueltypeprimary)}")
    print(f"state: {state}, type: {type(state)}")
    print(f"region: {region}, type: {type(region)}")
    print(f"condition: {condition}, type: {type(condition)}")
    print(f"paint_color: {paint_color}, type: {type(paint_color)}")
    print(f"days_since: {days_since}, type: {type(days_since)}")
    print(f"odometer: {odometer}, type: {type(odometer)}")

    mask = df_vehicles

    mask_length = len(mask)

    while mask_length > 1:
        # Step-by-step condition filtering
        mask = mask[mask["make"] == make]
        print(f"Rows after filtering by make: {len(mask)}")
        if len(mask) == 1:
            break

        mask = mask[mask["model"] == model]
        print(f"Rows after filtering by model: {len(mask)}")
        if len(mask) == 1:
            break

        mask = mask[mask["modelyear"] == float(modelyear)]
        print(f"Rows after filtering by modelyear: {len(mask)}")
        if len(mask) == 1:
            break

        mask = mask[mask["series"] == series]
        print(f"Rows after filtering by series: {len(mask)}")
        if len(mask) == 1:
            break        

        mask = mask[mask["trim"] == trim]
        print(f"Rows after filtering by trim: {len(mask)}")
        if len(mask) == 1:
            break        

        mask = mask[mask["enginecylinders"] == enginecylinders]
        print(f"Rows after filtering by enginecylinders: {len(mask)}")
        if len(mask) == 1:
            break

        if displacementcc == 'None':
            mask = mask[mask["displacementcc"].isnull()]
        else:
            mask = mask[mask["displacementcc"] == float(displacementcc)]
        print(f"Rows after filtering by displacementcc: {len(mask)}")
        if len(mask) == 1:
            break

        mask = mask[mask["drivetype"] == drivetype]
        print(f"Rows after filtering by drivetype: {len(mask)}")
        if len(mask) == 1:
            break

        if fueltypeprimary == None:
            print('Fuel Type None')
        else:
            mask = mask[mask["fueltypeprimary"] == fueltypeprimary]
        print(f"Rows after filtering by fueltypeprimary: {len(mask)}")
        if len(mask) == 1:
            break


    state_income = state_income_map[state]

    if len(mask) == 1:
        mask['state_income'] = state_income
        mask["state"] = state
        mask["region"] = region
        mask["condition"] = condition
        mask["paint_color"] = paint_color
        mask["days_since"] = days_since
        mask["odometer"] = odometer

        print("Final row with new values:")
        print(mask)

    else:
        print("No unique row found.")

    pred = cb72.predict(mask[cats+nums])[0].round().astype(int)

    print(type(pred))
    # Return the prediction
    return jsonify({"predicted_price": int(pred)})

@app.route('/model-tool',  methods=['GET', 'POST'])
def model_tool():
    df_vehicles = model_prep(pd.read_sql(f'{unique_vehicles}', seql_engine))

    # Precompute mappings
    make_to_model = (
        df_vehicles.groupby('make')['model']
        .apply(lambda x: x.unique().tolist())
        .to_dict()
    )

    model_to_series_trim = (
        df_vehicles.groupby(['model', 'series'])['trim']
        .apply(lambda x: x.unique().tolist())
        .reset_index()
        .groupby('model')
        .apply(lambda x: {
            'series': x['series'].unique().tolist(),
            'trim': list(set([item for sublist in x['trim'] for item in sublist]))  # Simplified trim processing
        })
        .to_dict()
    )

    dropdown_options = {
        "make": df_vehicles["make"].unique().tolist(),
        "Paint Color": df_vehicles["paint_color"].unique().tolist(),
        "condition": df_vehicles["condition"].unique().tolist(),
        "state" : df_vehicles['state'].unique().tolist()
    }

    return 

@app.route('/get-displacement-fueltype-engine', methods=['POST'])
def get_displacement_fueltype_engine():
    data = request.get_json()
    model = data.get('model')
    series = data.get('series')
    trim = data.get('trim')

    if not model or not series or not trim:
        return jsonify({"error": "Model, series, and trim are required"}), 400

    # Filter df_vehicles based on selected model, series, and trim
    filtered_df = df_vehicles[(df_vehicles['model'] == model) & 
                              (df_vehicles['series'] == series) & 
                              (df_vehicles['trim'] == trim)]

    # Get unique values for displacement, fueltype, and engine cylinders
    displacementcc = filtered_df['displacementcc'].fillna('None').unique().tolist()
    fueltypeprimary = filtered_df['fueltypeprimary'].fillna('None').unique().tolist()
    enginecylinders = filtered_df['enginecylinders'].fillna('None').unique().tolist()

    print(displacementcc)
    print(fueltypeprimary)
    print(enginecylinders)

    # Ensure non-empty arrays are returned, otherwise empty arrays
    return jsonify({
        "displacementCC": displacementcc if displacementcc else ['None'],
        "fueltypeprimary": fueltypeprimary if fueltypeprimary else ['None'],
        "enginecylinders": enginecylinders if enginecylinders else ['None']
    })

def get_json(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return {}

def vin_decode(vin):
    json_data = get_json(f'https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{vin}?format=json')
    
    if 'Results' in json_data and isinstance(json_data['Results'], list) and json_data['Results']:
        # Attempt to create a DataFrame
        return pd.DataFrame(json_data['Results'])
    else:
        print(f"Error in API response: {json_data.get('Results', {}).get('ErrorText', 'Unknown error')}")
        return None
    
def model_prep(df2):
    df2[cats] = df2[cats].astype(str)
    df2[nums] = df2[nums].astype('float64')
    return df2

def create_assumption(df):
    # Decode VIN to get vehicle features
    if isinstance(df, pd.Series):
        print('SRS')
        df = df.to_frame().T  # Convert Series to a DataFrame
    
    # Set default values for additional columns
    df['odometer'] = 100000
    df['paint_color'] = 'white'
    df['condition'] = 'good'
    df['state'] = 'tx'

    df['state_income'] = 59802
    df['region'] = 'dallas / fort worth'
    df['days_since'] = 1400

    return df

def find_similar_vehicles(row, df2, initial_threshold=0, increment=500, max_threshold=2000, n_veh=3):
    # Normalize column names to lowercase
    # df2.columns = df2.columns.str.lower()
    
    if isinstance(row, pd.DataFrame):
        if row.shape[0] == 1:
            df1 = row  # Keep as DataFrame if it's a single row
        else:
            df1 = row.iloc[0:1]  # If multiple rows, take the first row as a DataFrame
    elif isinstance(row, pd.Series):
        df1 = row.to_frame().T  # Convert Series to DataFrame
    else:
        print('row is not series or dataframe')
        print(row)
        return pd.DataFrame()  # Return empty DataFrame in case of error
    print('df1')
    print(type(df1))  # Should print <class 'pandas.core.frame.DataFrame'>
    print(df1.columns)  # To check if 'enginecylinders' is a column in df1
    print(df1['enginecylinders'].dtype)

    
    print(df1['enginecylinders'])
    print(df2['enginecylinders'].dtype)
    print(df2['enginecylinders'])
    df1['enginecylinders'] = df1['enginecylinders'].astype(float)
    df2['enginecylinders'] = df2['enginecylinders'].astype(float)
    print('df1 myr dtype')
    print(df1['modelyear'].dtype)
    print('df2 myr dtype')
    print(df2['modelyear'].dtype)

    # Ensure df1 column names are also lowercase
    #df1 = df1.rename(str.lower)

    # Check if displacementcc in df1 is null
    if pd.isna(df1['displacementcc'].iloc[0]):
        print('NULL DISPLACEMENT')
        return find_similar_vehicles_no_threshold(df1, df2, n_veh)
    else:
        print('succ')
        return find_similar_vehicles_with_threshold(df1, df2, initial_threshold, increment, max_threshold, n_veh)
    
def find_similar_vehicles_with_threshold(df1, df2, initial_threshold, increment, max_threshold, n_veh):
    print('sim thres called')
    threshold = initial_threshold
    similar_vehicles = pd.DataFrame()

    while len(similar_vehicles) < n_veh and threshold <= max_threshold:
        # Initialize mask to True for all rows
        
        mask = pd.Series(True, index=df2.index)
        
        # Assuming df1 is a single-row DataFrame
        if pd.notnull(df1['vehicletype'].iloc[0]):  # Using .iloc[0] to get the value from the first row
            mask &= df2['vehicletype'] == df1['vehicletype'].iloc[0]  # Compare with the value of df1
            print(mask[mask == True].size)
            
        if pd.notnull(df1['drivetype'].iloc[0]):  # Using .iloc[0] to get the value from the first row
            mask &= df2['drivetype'] == df1['drivetype'].iloc[0]  # Compare with the value of df1
            print(mask[mask == True].size)

        if pd.notnull(df1['gvwr'].iloc[0]):  # Using .iloc[0] to get the value from the first row
            mask &= df2['gvwr'] == df1['gvwr'].iloc[0]  # Compare with the value of df1
            print(mask[mask == True].size)

        if pd.notnull(df1['bodyclass'].iloc[0]):  # Using .iloc[0] to get the value from the first row
            mask &= df2['bodyclass'] == df1['bodyclass'].iloc[0]  # Compare with the value of df1
            print(mask[mask == True].size)

        if pd.notnull(df1['enginecylinders'].iloc[0]):  # Using .iloc[0] to get the value from the first row
            mask &= df2['enginecylinders'] == df1['enginecylinders'].iloc[0]  # Compare with the value of df1
            print(mask[mask == True].size)

        if pd.notnull(df1['modelyear'].iloc[0]):  # Using .iloc[0] to get the value from the first row
            mask &= abs(df2['modelyear'] - df1['modelyear'].iloc[0]) <= threshold / 500  # Compare with the value of df1
            print(mask[mask == True].size)

        if pd.notnull(df1['displacementcc'].iloc[0]):  # Using .iloc[0] to get the value from the first row
            mask &= abs(df2['displacementcc'] - df1['displacementcc'].iloc[0]) < threshold  # Compare with the value of df1
            print(mask[mask == True].size)
                    

        # Exclude the same make and model
        mask &= (df2['make'] + '_' + df2['model']) != (df1['make'].iloc[0] + '_' + df1['model'].iloc[0])
        
        # Apply the mask to filter df2
        similar_vehicles = df2[mask].drop_duplicates(subset=['make', 'model'])

        if len(similar_vehicles) < n_veh:
            threshold += increment

    return model_prep(similar_vehicles.head(n_veh))

def find_similar_vehicles_no_threshold(df1, df2, n_veh, max_threshold=5):
    similar_vehicles = pd.DataFrame()
    threshold = 1  # Start with a threshold of 1 year

    while len(similar_vehicles) < n_veh and threshold <= max_threshold:
        similar_vehicles = df2[
            (df2['vehicletype'] == df1['vehicletype']) &
            (df2['drivetype'] == df1['drivetype']) &
            (df2['gvwr'] == df1['gvwr']) &
            (df2['bodyclass'] == df1['bodyclass']) &
            (df2['enginecylinders'] == df1['enginecylinders']) &
            (abs(df2['modelyear'] - df1['modelyear']) <= threshold) &  # Allowing for modelyear leeway
            (df2['displacementcc'].isna()) &  # Only looking for rows where displacementCC is null
            ((df2['make'] + '_' + df2['model']) != (df1['make'] + '_' + df1['model']))
        ]

        threshold += 1  # Increment the threshold by 1 year for the next iteration

    # If still less than n_veh after 5 years, you can return what you found or handle as needed
    if len(similar_vehicles) < n_veh:
        print(f"Found {len(similar_vehicles)} similar vehicles, which is less than the requested {n_veh}.")
    '''
    # Adding extra columns to the resulting DataFrame
    similar_vehicles['condition'] = df1['condition']
    similar_vehicles['state'] = df1['state']
    similar_vehicles['region'] = df1['region']
    similar_vehicles['state_income'] = df1['state_income']   
    print(similar_vehicles)
    '''
    return model_prep(similar_vehicles.head(n_veh))

def create_odo_preds(row, odo_values, model=cb72, cats=cats, nums=nums):
    preds = []
    for odo in odo_values:
        row['odometer'] = odo
        preds.append(model.predict(row[cats+nums]).round().astype(int))
        
    return preds

def create_label(row):
    mk = row['make']
    mdl = row['model']
    
    # Ensure ModelYear is an integer, check for string 'nan'
    syr = str(int(row['modelyear'])) if row['modelyear'] != 'nan' else ''
    srs = row['series'] if row['series'] != 'nan' else ''
    trm = row['trim'] if row['trim'] != 'nan' else ''
    
    # Create label string
    label_str = syr + ' ' + mk + ' ' + mdl
    
    if srs:  # Only append if not empty
        label_str += ' ' + srs
        
    if trm:  # Only append if not empty
        label_str += ' ' + trm
        
    return label_str.strip()

def format_price(value):
    """Format a number as a price string."""
    return f"${value:,.0f}"

def generate_random_color():
    """Generate a random color that is not red."""
    while True:
        # Generate random RGB values
        color = [random.random() for _ in range(3)]  # Random RGB
        # Ensure the color is not too close to red (R=1, G=0, B=0)
        if not (color[0] > 0.8 and color[1] < 0.2 and color[2] < 0.2):  # Adjust this threshold as necessary
            return color
        
def plot_comparison(row, df, model=cb72, cats=cats, nums=nums, odo_values=np.arange(50000, 300001, 10000)):
    # Prepare the row
    if isinstance(row, pd.DataFrame):
        df1 = model_prep(row).iloc[0]  # Ensure single row
    elif isinstance(row, pd.Series):
        df1 = model_prep(row)
    else:
        print('row is not series or dataframe')
        return None  # Return None if input is invalid

    if isinstance(df, pd.DataFrame):
        df = model_prep(df)
    else:
        print('df is not dataframe')
        return None  # Return None if input is invalid

    label = create_label(df1)
    preds = create_odo_preds(df1, odo_values, model=model, cats=cats, nums=nums)

    # Initialize a DataFrame to hold the results
    results = pd.DataFrame({'odometer': odo_values})
    results[label] = [format_price(x) for x in preds]  # Use the primary vehicle label as the column name and format prices

    plt.figure(figsize=(12, 6))
    plt.plot(odo_values, preds, label=label, color='red', linewidth=3)
    # Plot primary vehicle predictions
    used_colors = []  # To keep track of used colors

    for idx, sim_row in df.iterrows():
        sim_label = create_label(sim_row)
        sim_preds = create_odo_preds(sim_row, odo_values, model=model, cats=cats, nums=nums)

        # Generate a new random color
        color = generate_random_color()
        
        # Ensure the color is unique
        while color in used_colors:
            color = generate_random_color()
        
        used_colors.append(color)  # Add to the used colors list
        plt.plot(odo_values, sim_preds, label=sim_label, color=color, linewidth=1)
        
        # Assign formatted predictions to the results DataFrame for each similar vehicle
        results[sim_label] = [format_price(x) for x in sim_preds]  # Directly assign formatted prices for the similar vehicle

        # Label predictions for similar vehicles every 50,000 miles
        for odo in range(0, 300001, 50000):
            if odo in odo_values:
                sim_pred_value = sim_preds[np.where(odo_values == odo)[0][0]]  # Get the corresponding prediction
                plt.text(odo, sim_pred_value, format_price(sim_pred_value), fontsize=10, color='black')

    # Label predictions for the primary vehicle every 50,000 miles
    for odo in range(0, 300001, 50000):
        if odo in odo_values:
            pred_value = preds[np.where(odo_values == odo)[0][0]]  # Get the corresponding prediction
            plt.text(odo, pred_value, format_price(pred_value), fontsize=10, color='black')

    # Add title and labels
    plt.title(label + ' vs. Competition')
    plt.xlabel('Miles')
    plt.ylabel('Price')

    # Show the legend
    plt.legend()

    results = results[results['odometer'].isin(np.arange(50000, 300001, 50000))]
    results['odometer'] = results['odometer'].apply(lambda x: f"{int(x):,}")
    print(results)
    # Save plot to a string buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return img_base64, results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    action = request.form.get('action')
    if action == 'search_by_vin':
        vin = request.form.get('vin')
        # Handle VIN search logic
    elif action == 'search_by_make_model_year':
        make_model_year = request.form.get('make_model_year')
        # Handle Make/Model/Year search logic
    # Render results with table.html included
    return render_template('index.html')

@app.route('/search_make_model_year', methods=['POST'])
def search_make_model_year():
    global df_vehicles
    if df_vehicles is None:
        print('search make model  year getting df_vehicles')
        df_vehicles = model_prep(pd.read_sql(f'{unique_vehicles}', seql_engine)).replace({'None': 'nan'})

    try:
        data = request.get_json()  # Get the JSON data sent from the frontend
        make_model_year = data.get('make_model_year', '').strip()
        
        if not make_model_year:
            return jsonify({'error': 'Please enter a valid Make, Model, and Year.'}), 400
        
        parts = make_model_year.split(' ')
        year, make, model = None, None, None

        try:
            for i, part in enumerate(parts):
                if part.isnumeric():
                    if 1950 < int(part) < 2025:  # Check if it is a valid year
                        year = int(part)
                elif not make:
                    make = part
                elif not model:
                    model = part
                elif len(parts) > 3 and i >= 2:  # Add remaining parts to model if there are more than 3 parts
                    model += ' ' + str(part)
            print(f"make:{make} model:{model} year:{str(year)}")
        except:
            print('string parser fail')
            print(f"make:{make} model:{model} year:{str(year)}")

        print(df_vehicles[['make','model','modelyear']].dtypes)
        print(type(make))
        print(type(model))
        print(type(year))

        # Filter the vehicles from df_vehicles
        matches = df_vehicles[
            (df_vehicles['make'].str.lower() == make.lower()) &
            (df_vehicles['model'].str.lower() == model.lower()) &
            (df_vehicles['modelyear'] == year)
        ]

        if matches.empty:
            return jsonify({'error': f"No vehicles found for {year} {make} {model}."}), 400

        # Build the HTML form with table rows manually
        table_html = f'<h1>Select a Vehicle for {year} {make.capitalize()} {model.capitalize()}</h1>'
        table_html += '<table id="vinTable">'
        table_html += '''
            <tr>
                <th>Series</th>
                <th>Trim</th>                
                <th>DriveType</th>
                <th>DisplacementCC</th>
                <th>FuelTypePrimary</th>
                <th>Select</th>
            </tr>
        '''

        # Loop through the matches and create table rows with buttons
        for _, row in matches[['series', 'trim', 'drivetype', 'displacementcc', 'vin', 'fueltypeprimary']].drop_duplicates(
            subset=['series', 'trim', 'drivetype', 'displacementcc', 'fueltypeprimary']).iterrows():
            table_html += f'''
                <tr>
                    <td>{row['series']}</td>
                    <td>{row['trim']}</td>                
                    <td>{row['drivetype']}</td>
                    <td>{row['displacementcc']}</td>
                    <td>{row['fueltypeprimary']}</td>
                    <td>
                        <button 
                            type="button" 
                            class="select-button" 
                            data-vin="{row['vin']}"
                        >
                            Select
                        </button>
                    </td>
                </tr>
            '''

        table_html += '</table>'

        # Return the dynamically created table as a response
        return jsonify({'html': table_html})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search_vin', methods=['POST'])
def search_vin():
    global df_vehicles
    if df_vehicles is None:
        df_vehicles = model_prep(pd.read_sql(f'{unique_vehicles}', seql_engine)).replace({'None': 'nan'})
    try:
        vin = None

        # Handle JSON request (likely from the input form)
        if request.is_json:
            vin = request.get_json().get('vin', '').strip()
            print(vin)
        # Handle form submissions (either POST or GET)
        if not vin:
            print('used selected idx')
            vin = request.args.get('selected_idx') or request.form.get('selected_idx')
        print(vin)
        # Ensure VIN is cleaned up and handle missing VIN
        if vin:
            vin = vin.strip()
            print(f"DEBUG: Received VIN: {vin}")
            #return jsonify({"vin": vin})
        else:
            return "VIN is required!", 400
        
        if vin in df_vehicles.vin.unique():
            print('found vin in vehicles')
            df = df_vehicles[df_vehicles['vin'] == vin]
            print(df[['vin', 'model', 'enginecylinders', 'modelyear', 'displacementcc']])
            print(df[['vin', 'model', 'enginecylinders', 'modelyear', 'displacementcc']].dtypes)
            #if not df.empty:
            #df = df.replace('', 'nan')  # Replace empty strings with 'nan'
            #df.columns = df.columns.str.lower()
            #df['enginecylinders'] = pd.to_numeric(df['enginecylinders'], errors='coerce').astype('Int64')
            #df['modelyear'] = pd.to_numeric(df['modelyear'], errors='coerce').astype('Int64')

            #df['displacementcc'] = df['displacementcc'].astype(float)
            #print('cols lowered')
        
            features = create_assumption(df)
            
            return create_plot(features)

        else:    
            print('decoding vin')
            df = vin_decode(vin)

            if df is not None:
                mask = (
                    (df['ErrorCode'].isin(['0', '1', '6'])) &  # ErrorCode should be 0, 1, or 6
                    (~df[['Make', 'Model', 'ModelYear', 'VIN']].isnull().any(axis=1))  # Make sure Make, Model, ModelYear, VIN are not null
                )
                if mask.any():
                    df = df.replace('', 'nan')  # Replace empty strings with 'nan'
                    df.columns = df.columns.str.lower()
                    df['enginecylinders'] = pd.to_numeric(df['enginecylinders'], errors='coerce').astype(float)
                    df['modelyear'] = pd.to_numeric(df['modelyear'], errors='coerce').astype(float)
                    df['displacementcc'] = df['displacementcc'].astype(float)
                    print('cols lowered')
                    print(df[['enginecylinders', 'modelyear', 'displacementcc']])
                    print(df[['enginecylinders', 'modelyear', 'displacementcc']].dtypes)
                    features = create_assumption(df)
                    return create_plot(features, new_vin=True)
                else:
                    return 'error'
            else:
                return 'error'
                    
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_prediction', methods=['POST'])
def update_prediction():
    try:
        # Get the data sent from the frontend (row data)
        row_data = request.get_json()

        # Get the values sent from the row, including the necessary ones for prediction
        odometer = row_data['odometer']
        paint_color = row_data['paint_color']
        state = row_data['state']
        condition = row_data['condition']
        index = row_data['index'] 

        data_cols = [x for x in cats+nums if x not in ['odometer','paint_color','state','condition']]

        vehicle_row = df_vehicles.loc[[index]]
        
        vehicle_row['odometer'] = odometer
        vehicle_row['paint_color'] = paint_color
        vehicle_row['state'] = state
        vehicle_row['condition'] = condition
        vehicle_row['state_income'] = 59802
        vehicle_row['region'] = 'dallas / fort worth'
        vehicle_row['days_since'] = 1400

        try:
        # Make the prediction using the model (cb72 in your case)
            predicted_price = cb72.predict(model_prep(vehicle_row[cats+nums])).round().astype(int)[0]
        except:
            print('prediction failure')

        # Return the new predicted price for this row
        return jsonify({'success': True, 'predicted_price': int(predicted_price)})

    except Exception as e:
        print(f"Error during prediction update: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/create_plot', methods=['POST'])
def create_plot(features, new_vin=False):
    global df_vehicles
    if df_vehicles is None:
        print('df_vehicles being created')
        df_vehicles = model_prep(pd.read_sql(f'{unique_vehicles}', seql_engine)).replace({'None': 'nan'})

    print('creating plot...')
    print('features:')
    print(features)

    similar_vehicles = find_similar_vehicles(features, df_vehicles)
    print('sim veh:')
    print(similar_vehicles)

    if new_vin==True:
        print('this is the row + similar vehicles')
        df = pd.concat([df_vehicles.loc[similar_vehicles.index.tolist()], features])[cats + nums]
        print(df.head())
    else:
        print('this is the row + similar vehicles')
        df = df_vehicles.loc[similar_vehicles.index.tolist() + features.index.tolist(), cats + nums]
        print(df.head())

    # Default values for dropdowns
    default_paint_color = 'white'
    default_state = 'tx'
    default_odometer = 100000
    default_condition = 'good'

    # Fill in default values for dropdowns
    df['paint_color'] = default_paint_color
    df['state'] = default_state
    df['odometer'] = default_odometer
    df['condition'] = default_condition
    df['state_income'] = 59802
    df['region'] = 'dallas / fort worth'
    df['days_since'] = 1400

    # Predict using the default values
    df['predicted_price'] = cb72.predict(model_prep(df)).round().astype(int)

    make_options = df_vehicles.make.unique()
    paint_color_options = [np.nan, 'red', 'grey', 'brown', 'silver', 'black', 'white', 'green',
                           'blue', 'custom', 'orange', 'yellow', 'purple']
    odometer_options = list(range(0, 500001, 25000))
    state_options = df_vehicles['state'].dropna().unique().tolist()
    condition_options = ['excellent', 'like new', 'good', np.nan, 'fair', 'new', 'salvage']

    plot_img, _ = plot_comparison(features, similar_vehicles)

    # Generate dropdowns and table HTML
    table_rows = []
    for i, row in df.iterrows():
        row_html = f"""
        <tr>
            <td>
                <button type="button" class="clone-button" data-row-index="{i}">Clone</button>
            </td>
            <td>
                <select name="odometer" data-row-index="{i}">
                    {"".join(
                        f'<option value="{value}" {"selected" if value == default_odometer else ""}>{value}</option>'
                        for value in odometer_options
                    )}
                </select>
            </td>
            <td>{row['make']}</td>
            <td>{row['model']}</td>
            <td>{row['modelyear']}</td>
            <td>{row['series']}</td>
            <td>{row['trim']}</td>
            <td>{row['enginecylinders']}</td>
            <td>{row['drivetype']}</td>
            <td>
                <select name="paint_color" data-row-index="{i}">
                    {"".join(
                        f'<option value="{color}" {"selected" if color == default_paint_color else ""}>{color}</option>'
                        for color in paint_color_options
                    )}
                </select>
            </td>
            <td>
                <select name="state" data-row-index="{i}">
                    {"".join(
                        f'<option value="{state}" {"selected" if state == default_state else ""}>{state}</option>'
                        for state in state_options
                    )}
                </select>
            </td>
            <td>
                <select name="condition" data-row-index="{i}">
                    {"".join(
                        f'<option value="{condition}" {"selected" if condition == default_condition else ""}>{condition}</option>'
                        for condition in condition_options
                    )}
                </select>
            </td>            
            <td>
                <span class="predicted-price" data-row-index="{i}">{row['predicted_price']}</span>
            </td>
            <td>
                <button type="button" class="update-prediction" id="update-prediction" data-row-index="{i}">Update</button>
            </td>
        </tr>
        """
        table_rows.append(row_html)

    table_body_html = "\n".join(table_rows)

    styled_table_html = f"""
    <style>
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #f4f4f4;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f1f1f1;
        }}
        select {{
            width: 100%;
        }}
    </style>
    <table class="results-table">
        <thead>
            <tr>
                <th>Action</th>
                <th>Odometer</th>
                <th>Make</th>
                <th>Model</th>
                <th>Year</th>
                <th>Series</th>
                <th>Trim</th>
                <th>Engine Cylinders</th>
                <th>DriveType</th>
                <th>Paint Color</th>
                <th>State</th>
                <th>Condition</th>
                <th>Predicted Price</th>
                <th>Update</th>
            </tr>
        </thead>
        <tbody>
            {table_body_html}
        </tbody>
    </table>
    """

    return jsonify({'plot_img': plot_img, 'html_table': styled_table_html})


@app.route('/process_selection', methods=['POST'])
def process_selection():
    selected_idx = request.form.get('selected_idx')
    vin = df_vehicles.loc[selected_idx, 'vin']
    features = create_assumption(df_vehicles.loc[int(selected_idx)])  # Fetch the vehicle row using the index
    similar_vehicles = find_similar_vehicles(features, df_vehicles)
                
    plot_img, _ = plot_comparison(features, similar_vehicles)

    table_html = df_vehicles.loc[similar_vehicles.index+selected_idx, ['odometer', 'price', 'make','model', 'modelyear', 'series', 'trim', 'enginecylinders', 'drivetype']].to_html(
                index=False, header=False, escape=False)
    
    styled_table_html = f"""
            <style>
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border: 1px solid #ddd;
                }}
                th {{
                    background-color: #f4f4f4;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                tr:hover {{
                    background-color: #f1f1f1;
                }}
            </style>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Odometer</th>
                        <th>Price</th>
                        <th>Make</th>
                        <th>Model</th>
                        <th>Year</th>
                        <th>Series</th>
                        <th>Trim</th>
                        <th>Engine Cylinders</th>
                        <th>DriveType</th>
                    </tr>
                </thead>
                <tbody>
                    {table_html}
                </tbody>
            </table>
            """
    #results_html = results.to_html(index=False, classes='data', border=0)
    return jsonify({'plot_img': plot_img, 'html_table':styled_table_html})

@app.route('/api/price_changes', methods=['GET'])
def price_changes():

    try:
        df = pd.read_sql('price_changes', seql_engine)
    except:
        df = pd.DataFrame()
        print('df fail')
    #df['modelyear'] = df['modelyear'].astype(int)
    #df['new_posting_date'] = df['new_posting_date'].astype(str)
    #df['old_posting_date'] = df['old_posting_date'].astype(str)
    df['new_date_scraped'] = df['new_date_scraped'].dt.strftime('%d %b %Y')
    df['old_date_scraped'] = df['old_date_scraped'].dt.strftime('%d %b %Y')
    df = df.fillna({'trim': 'None', 'series' : 'None', 'enginecylinders': 'N/A', 'drivetype' : 'unknown'})
    # Store in Flask.g for reuse
    g.table_data = df.to_dict(orient='records')

    # Return as JSON for API clients
    return jsonify(g.table_data)

@app.route('/api/data', methods=['GET'])
def data():
    
    df = pd.read_sql('latest_listings', seql_engine)
    df['residual_percentage'] = round(((df['residual'] / df['price']) * 100),2)
    
    # Format posting_date to just show day, month, and year
    df['posting_date'] = df['posting_date'].dt.strftime('%d %b %Y')
    df = df.fillna({'trim': 'None', 'series' : 'None', 'enginecylinders': 'N/A', 'drivetype' : 'unknown'})
    # Store in Flask.g for reuse
    g.table_data = df.to_dict(orient='records')

    # Return as JSON for API clients
    return jsonify(g.table_data)


@app.route('/search_make_model', methods=['GET', 'POST'])
def search_make_model():
    make_model_year = request.args.get('make_model_year')  # Use request.args for GET parameters
    parts = make_model_year.split(' ')
    year, make, model = None, None, None

    for part in parts:
        if part.isnumeric():
            year = float(part)
        elif not make:
            make = part
        else:
            model = part

    make_lower = make.lower() if make else None

    matches = df_vehicles[
        (df_vehicles['make'].str.lower() == make_lower) &
        (df_vehicles['model'].str.contains(model, case=False)) &
        (df_vehicles['modelyear'] == year)
    ]
    
    if matches.empty:
        return render_template('error.html', error_message="No vehicles found matching the provided Make, Model, and Year.")
    # Get unique Series, Trim, and DriveType options along with their index
    unique_options = matches[[x.lower() for x in ['series', 'trim', 'drivetype', 'displacementcc', 'fueltypeprimary']]].drop_duplicates()
    return render_template('select_vehicle.html', matches=unique_options, make=make_lower, model=model, year=year)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
