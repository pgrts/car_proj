from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import io
import base64
import requests
import random
from sqlalchemy import create_engine, text, inspect
import os 
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file
db_password = os.getenv('DB_PASSWORD')

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

def get_first(ls):
    if (len(ls) == 1):
        return ls[0]
    else: 
        return get_first(list(set(ls)))
    
def latest_cbm_f():
    
    cbm_files =  os.listdir(os.path.join(os.getcwd(), '..', 'cb_models'))
    cbm_dates = [x.lstrip('cb_model_').rstrip('.cbm') for x in cbm_files]
    latest_file = [x for x in cbm_files if max(cbm_dates) in x]

    return os.path.join(os.getcwd(), '..', 'cb_models', get_first(latest_file))    
 

def model_prep(df2):
    df2[cats] = df2[cats].astype(str)
    df2[nums] = df2[nums].astype('float64')
    return df2
    
# Load the dataframe of vehicle data for finding similar vehicles
engine = create_engine(f'postgresql+psycopg2://postgres:{db_password}@localhost:5432/cars')
df_vehicles = pd.read_sql('car_data', engine)
cb72 = CatBoostRegressor()
cb72.load_model(latest_cbm_f())


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
    
    if json_data['Results']:
        df = pd.DataFrame(json_data['Results'])
    else: 
        df = pd.DataFrame(columns=cats+nums)
    return df

def model_prep(df2):
    df2[cats] = df2[cats].astype(str)
    df2[nums] = df2[nums].astype('float64')
    return df2

def create_assumption(df):
    # Decode VIN to get vehicle features
    if isinstance(df, pd.Series):
        df = df.to_frame().T  # Convert Series to a DataFrame

    # Set default values for additional columns
    df['odometer'] = 100000
    df['paint_color'] = 'white'
    df['condition'] = 'good'
    df['state_income'] = 59802
    df['state'] = 'tx'
    df['region'] = 'dallas / fort worth'
    df['days_since'] = 1400

    return model_prep(df)


def find_similar_vehicles(row, df2, initial_threshold=0, increment=100, max_threshold=1000, n_veh=10):
    # Normalize column names to lowercase
    df2.columns = df2.columns.str.lower()
    
    if isinstance(row, pd.DataFrame):
        row = model_prep(row)
        df1 = row.iloc[0]  # Ensure single row
    elif isinstance(row, pd.Series):
        df1 = row
    else:
        print('row is not series or dataframe')
        print(row)
        return pd.DataFrame()  # Return empty DataFrame in case of error

    # Ensure df1 column names are also lowercase
    df1 = df1.rename(str.lower)

    # Check if displacementcc in df1 is null
    if pd.isna(df1['displacementcc']):
        return find_similar_vehicles_no_threshold(df1, df2, n_veh)
    else:
        return find_similar_vehicles_with_threshold(df1, df2, initial_threshold, increment, max_threshold, n_veh)

def find_similar_vehicles_with_threshold(df1, df2, initial_threshold, increment, max_threshold, n_veh):
    threshold = initial_threshold
    similar_vehicles = pd.DataFrame()

    while len(similar_vehicles) < n_veh and threshold <= max_threshold:
        similar_vehicles = df2[
            (df2['vehicletype'] == df1['vehicletype']) &
            (df2['drivetype'] == df1['drivetype']) &
            (df2['gvwr'] == df1['gvwr']) &
            (df2['bodyclass'] == df1['bodyclass']) &
            (df2['enginecylinders'] == df1['enginecylinders']) &
            (df2['modelyear'] == df1['modelyear']) &
            (abs(df2['displacementcc'] - df1['displacementcc']) < threshold) & 
            ((df2['make'] + '_' + df2['model']) != (df1['make'] + '_' + df1['model']))
        ]
        
        similar_vehicles = similar_vehicles.drop_duplicates(subset=['make', 'model'])

        if len(similar_vehicles) < n_veh:
            threshold += increment

    # Add extra columns
    similar_vehicles['condition'] = df1['condition']
    similar_vehicles['state'] = df1['state']
    similar_vehicles['region'] = df1['region']
    similar_vehicles['state_income'] = df1['state_income']   
    
    return similar_vehicles.reset_index(drop=True)

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

    # Adding extra columns to the resulting DataFrame
    similar_vehicles['condition'] = df1['condition']
    similar_vehicles['state'] = df1['state']
    similar_vehicles['region'] = df1['region']
    similar_vehicles['state_income'] = df1['state_income']   

    return similar_vehicles.reset_index(drop=True)


def create_odo_preds(row, odo_values, model=cb72, cats=cats, nums=nums):
    preds = []
    for odo in odo_values:
        row['odometer'] = odo
        preds.append(model.predict(row[cats+nums]))
        
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
        row = create_assumption(row)
        df1 = row.iloc[0]  # Ensure single row
    elif isinstance(row, pd.Series):
        df1 = row
    else:
        print('row is not series or dataframe')
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
    #print(results['Odometer'])
    results = results[results['odometer'].isin(np.arange(50000, 300001, 50000))]
    results['odometer'] = results['odometer'].apply(lambda x: f"{int(x):,}")
    print(results)
    # Save plot to a string buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return img_base64, results

@app.route('/', methods=['GET', 'POST'])
def index():
    df = pd.read_sql('SELECT modelyear, make, model, price, predicted_price, residual, residual_percentage, link, posting_date, date_scraped FROM "latest_listings"', engine)
    filtered_df = df[(df['residual_percentage'] >= 5) & (df['residual_percentage'] < 5550) & (abs(df['residual']) > 500)].sort_values(by='residual', ascending=True)
    filtered_df['neg_res'] = filtered_df['residual']*-1
    filtered_df['modelyear'] = filtered_df['modelyear'].astype(int)
    filtered_df['price'] = '$' + filtered_df['price'].astype(int).apply(lambda x: "{:,}".format(int(x)))
    filtered_df['predicted_price'] = '$' + filtered_df['predicted_price'].apply(lambda x: "{:,}".format(int(x)))
    
    
    filtered_df['new_resid'] = '$' + filtered_df['neg_res'].apply(lambda x: "{:,}".format(int(x))) + ' (' + filtered_df['residual_percentage'].round(0).astype(int).astype(str) + '%)'

    good_listings = filtered_df.head(25).to_dict(orient='records')
    bad_listings = filtered_df.tail(25).to_dict(orient='records')

    if request.method == 'POST':
        action = request.form.get('action')  # Get the button action

        if action == 'search_by_vin':
            vin = str(request.form.get('vin')).strip()

            if vin:
                
                vehicle_row = create_assumption(df_vehicles[df_vehicles['vin'] == vin])
                if not vehicle_row.empty:

                    features = vehicle_row.iloc[0]
                else:
                    df = vin_decode(vin)
 
                    if df.empty:
                        raise ValueError("Error: No vehicle information found for the provided VIN.")

                    df = df.replace('', 'nan')  # Replace empty strings with 'nan'

                    features = create_assumption(df)

                similar_vehicles = find_similar_vehicles(features, df_vehicles)
                plot_url, results = plot_comparison(features, similar_vehicles)
                results_html = results.to_html(index=False, classes='data', border=0)

                return render_template('result.html', plot_url=plot_url, results=results_html, similar_vehicles=similar_vehicles[[x.lower() for x in ['Make', 'Model', 'ModelYear', 'Series', 'Trim', 'DriveType']]])

            else:
                # Handle the case when no VIN is entered
                error_message = "Please enter a VIN."
                return render_template('index.html', error_message=error_message)

        elif action == 'search_by_make_model_year':
            make_model_year = str(request.form.get('make_model_year')).strip()
            return redirect(url_for('search_make_model', make_model_year=make_model_year))

    return render_template('index.html',  good_listings=good_listings, bad_listings=bad_listings)

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
    unique_options = matches[[x.lower() for x in ['Series', 'Trim', 'DriveType', 'DisplacementCC', 'FuelTypePrimary']]].drop_duplicates()
    return render_template('select_vehicle.html', matches=unique_options, make=make_lower, model=model, year=year)

@app.route('/process_selection', methods=['POST'])
def process_selection():
    selected_idx = request.form.get('selected_idx')
    print(selected_idx)  # Get the index of the selected option
    vehicle_row = create_assumption(df_vehicles.loc[int(selected_idx)])  # Fetch the vehicle row using the index
    print(vehicle_row)
    if not vehicle_row.empty:
        features = vehicle_row.iloc[0]
        similar_vehicles = find_similar_vehicles(features, df_vehicles)
        plot_url, results = plot_comparison(features, similar_vehicles)
        results_html = results.to_html(index=False, classes='data', border=0)

        return render_template('result.html', plot_url=plot_url, results=results_html,
                               similar_vehicles=similar_vehicles[[x.lower() for x in ['Make', 'Model', 'ModelYear', 'Series', 'Trim', 'DriveType']]])
    else:
        error_message = "No vehicle found with the selected options."
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
