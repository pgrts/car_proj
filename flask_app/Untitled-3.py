
import pandas as pd
import numpy as np
import requests

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
        return pd.DataFrame(json_data['Results'])
    else: 
        return None
    
print(vin_decode('19XFC2F59JE206832'))