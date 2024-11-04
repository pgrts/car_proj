from sqlalchemy import inspect
import pandas as pd
import numpy as np

from datetime import date, datetime
import time 

from sqlalchemy import create_engine, text

import json
import requests
from bs4 import BeautifulSoup
import re

import matplotlib.pyplot as plt
from catboost import CatBoostRegressor


def df_to_table(df, final_table_name, engine, exist = 'replace'):
    with engine.connect() as connection:
        df.to_sql(final_table_name, con=connection, if_exists=exist, index=False)
        print(f"Table '{final_table_name}' created successfully using SQLAlchemy.")
        
def scrape_regions(df):
    dfls = []

    for region_url in df.region_url.unique():
        link = region_url + '/search/cta?auto_title_status=1&bundleDuplicates=1&query=vin#search=1~gallery~0~0'
        tdf = df_from_link(link)
        tdf['region_url'] = region_url
        dfls.append(tdf)
        time.sleep(1)
        
    return dfls

def clean_mask_price(df, mini, maxi):
    if df['price'].dtype == 'object':
        df['price'] = df['price'].str.replace(',', '').str.replace('$', '').astype(float)
    return df[(df['price'] > mini) & (df['price'] < maxi)]   


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

def batch_vin(vin_input):
    url = 'https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/'
    post_fields = {'format': 'json', 'data': vin_input}
    r = requests.post(url, data=post_fields)
    vin_return = json.loads(r.text)
    return pd.DataFrame(vin_return['Results'])


def clean_vin_output(df):
    # Replace empty strings with 'nan' (if you want actual NaNs, use np.nan instead of 'nan')
    df = df.replace('', 'nan')

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

    return df2, df_bad

def process_vin_batch(vin_batch, engine, vins_accepted, vins_rejected, datestr):
    # Decode batch and clean
    vin_df = batch_vin(';'.join(vin_batch))
    valid_vin_df, reject_vin_df = clean_vin_output(vin_df)
    
    with engine.connect() as conn:
        # Store valid VINs in database
        if not valid_vin_df.empty:
            valid_vin_df.to_sql(vins_accepted, engine, if_exists='append', index=False)
            print(f"Added {len(valid_vin_df)} records to {vins_accepted}")

        # Store rejected VINs in database with scrape date
        if not reject_vin_df.empty:
            reject_vin_df['date_scraped'] = datestr
            reject_vin_df.to_sql(vins_rejected, engine, if_exists='append', index=False)
            print(f"Added {len(reject_vin_df)} records to {vins_rejected}")
    
    # Clear batch after processing
    vin_batch.clear()
    print("Cleared vin_batch after processing.")

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
        
    posting_body_section = soup.find('section', id='postingbody')
    if posting_body_section:
        h2_tag = posting_body_section.find('h2')
        if h2_tag:
            # Clean text by removing HTML tags and special characters
            posting_body = h2_tag.get_text(strip=True)
            posting_body = re.sub(r'[^a-zA-Z0-9\s]', '', posting_body)  # Remove special characters
            data['postingbody'] = posting_body
        else:
            data['postingbody'] = None
    else:
        data['postingbody'] = None
        
        
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

def clean_listing_output(df2):
    # Column renaming dictionary
    column_rename_dc = {
        'VIN:': 'VIN', 'condition:': 'condition', 'drive:': 'drive', 'fuel:': 'fuel',
        'paint color:': 'paint_color', 'type:': 'type', 'transmission:': 'transmission',
        'title status:': 'title_status', 'odometer:': 'odometer'
    }
    
    # Rename and clean DataFrame
    df = df2.rename(columns=column_rename_dc).reset_index(drop=True)
    df = df.replace('', 'nan')

    if 'odometer' in df and df['odometer'].dtype == 'object':
        df['odometer'] = df['odometer'].str.replace(',', '').astype(float)

    # Truncate VIN to 16 characters if longer, else mark for rejection if less than 16
    df['VIN'] = df['VIN'].str[:16]
    
    # Filter listings with valid VIN and odometer
    valid_df = df[(df['VIN'].notnull()) & (df['VIN'].str.len() == 16) & (df['odometer'].between(25000, 310000))]
    reject_df = df[~df.index.isin(valid_df.index)]  # Entries that don't meet criteria

    return valid_df, reject_df

def posting_date(df):
    #df['posting_date'] = pd.to_datetime(df['posting_date']).dt.tz_localize(None)
    df['posting_date'] = pd.to_datetime(df['posting_date'], utc=True).dt.tz_localize(None)
    reference_date = pd.to_datetime('2021-01-01')
    df['days_since'] = (df['posting_date'] - reference_date).dt.days
    df['reference_date'] = reference_date
    return df


def do_lots_stuff(main_data, datestr, reg_ref = 'region_reference',
    dbname='cars', user='postgres', password='p33Gritz!!', host='localhost', port='5432',
    links_rejected = 'links_rejected', vins_rejected = 'vins_rejected', listings_rejected = 'listings_rejected'):
                  
    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}')
    
    backup_tablename = main_data + datestr
    
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

    listings_accepted = 'listings_accepted' + datestr
    vins_accepted = 'vins_accepted' + datestr    
    links_accepted = 'links_accepted' + datestr
    
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
            backup_df = pd.read_sql(f'SELECT * FROM "{main_data}";', conn)[['link','VIN']]
            print(f"Backup DataFrame shape: {backup_df.shape}")
            
            reg_ref_df = pd.read_sql(reg_ref, conn)
                    
            check_link_query = f"""
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.tables 
                WHERE table_name = '{links_accepted}'
            );
            """
            result = conn.execute(text(check_link_query)).fetchone()
            
            print("Fetching already rejected links...")
            already_rejected_links = pd.read_sql(f'SELECT "link" FROM {links_rejected}', engine)['link'].tolist()
            all_links = list(set(backup_df.link.tolist() + already_rejected_links))
            print('length of all rejected and in data links ' + str(len(all_links)))
            
            print("Fetching already rejected VINS...")
            already_rejected_vins = pd.read_sql(f'SELECT "VIN" FROM {vins_rejected}', engine)['VIN'].tolist()
            all_vins = list(set(backup_df.VIN.tolist() + already_rejected_vins))
            print('length of all rejected and in data vins ' + str(len(all_vins)))
            
            if result[0]:  # If the table exists
                print(f"Table '{links_accepted}' already exists. Skipping creation.")
            
            else:
                scraped_links = pd.concat(scrape_regions(reg_ref_df))
      
                repeat_mask = scraped_links.link.isin(all_links)
                
                # takes bulk links, removes those in accepted database and reject database
                new_links = scraped_links[~repeat_mask].drop_duplicates(subset=['link'])

                # only want price > 4000, < 110000
                need_these_links = clean_mask_price(new_links, 4000, 110000) 
                df_to_table(need_these_links, links_accepted, engine, 'replace')
                print(links_accepted + ' created')           
                            
                # add more rejects that do not match price requirements
                reject_links = new_links[~new_links.link.isin(need_these_links.link)]
                reject_links['date_scraped'] = datestr
                df_to_table(reject_links, links_rejected, engine, 'append')
                print(links_rejected + ' created')
            
            if not table_exists(listings_accepted, conn):
                conn.execute(text(f"""
                    CREATE TABLE {listings_accepted} (
                        "VIN" TEXT, condition TEXT, drive TEXT, fuel TEXT, odometer FLOAT,
                        paint_color TEXT, title_status TEXT, transmission TEXT, type TEXT,
                        posting_date TIMESTAMP, lat TEXT, long TEXT, geo_placename TEXT,
                        geo_region TEXT, postingbody TEXT, title TEXT, link TEXT PRIMARY KEY
                    )
                """))
                conn.commit()
                print(f"Created the listing target table '{listings_accepted}'.")
            else:
                print(f"The listing target table '{listings_accepted}' already exists.")

            # Check and create good VIN table if it doesn't exist
            if not table_exists(vins_accepted, conn):
                conn.execute(text(f"""
                    CREATE TABLE \"{vins_accepted}\" AS TABLE "schema_example" WITH NO DATA;
                """))
                print(f"Created the good VIN table '{vins_accepted}'.")
                conn.commit()
            else:
                print(f"The good VIN table '{vins_accepted}' already exists.")
    
            # all_links contains rejected links so will not process them. adding those already in accepted in case of restart
            #existing_links = pd.read_sql(f"SELECT link FROM {links_accepted};", conn)['link'].tolist()
            listing_links = pd.read_sql(f"SELECT link FROM {listings_accepted};", engine)['link'].tolist()
            rej_listings = pd.read_sql(f"SELECT link FROM {listings_rejected};", engine)['link'].tolist()

            all_links = list(set(all_links + listing_links + rej_listings))

            link_source_df = pd.read_sql(f'SELECT link FROM {links_accepted}', engine)

            remaining_links = link_source_df[~link_source_df['link'].isin(all_links)]['link'].tolist()

            print(f"Found {len(remaining_links)} remaining links to process.")

    except Exception as e:
        print(f"Error fetching links from tables: {e}")
        return  # Exit if there's an issue
    
    #do_more(remaining_links, listings_accepted, listings_rejected, vins_accepted, vins_rejected, all_vins, datestr, engine)         
    vin_batch = [] 
    for link in remaining_links:
        try:
            parsed_df = link_parser(link)
            if parsed_df is not None:
                valid_df, reject_df = clean_listing_output(parsed_df)
                
                with engine.connect() as conn:
                    if not valid_df.empty:
                        good_vin_df = pd.read_sql(f'SELECT "VIN" FROM {vins_accepted}', con=engine)
                        bad_vin_df = pd.read_sql(f'SELECT "VIN" FROM {vins_rejected}', con=engine)
                        all_vins += good_vin_df['VIN'].tolist() + bad_vin_df['VIN'].tolist()
                        vin = valid_df.at[0, 'VIN']
                        
                        if (vin not in vin_batch) and (vin not in all_vins):
                            valid_df.to_sql(listings_accepted, con=conn, if_exists='append', index=False)
                            vin_batch.append(vin)
                            print(f"VIN added to batch: {vin}")
                        else:
                            print(f"VIN already in database: {vin}, adding to {listings_rejected}")
                            valid_df.to_sql(listings_rejected, con=conn, if_exists='append', index=False)

                        # Process batch if full
                        if len(vin_batch) == 50:
                            process_vin_batch(vin_batch, engine, vins_accepted, vins_rejected, datestr)

                    # Insert rejected listings
                    if not reject_df.empty:
                        reject_df['VIN'] = None
                        reject_df['odometer'] = None
                        reject_df['date_scraped'] = datestr
                        reject_df.to_sql(listings_rejected, con=conn, if_exists='append', index=False)
                        print(f"Added reject records to {listings_rejected}")

                print(f"Processed link: {link}")
            else:
                print(f"Link {link} returned blocked or empty data.")
        
        except Exception as e:
            print(f"Error processing {link}: {e}")

        finally:
            time.sleep(0.5)

    # Process any remaining vins in the batch after all links are processed
    if vin_batch:
        process_vin_batch(vin_batch, engine, vins_accepted, vins_rejected, datestr)



def finish_everything(main_data, links_table, listing_table, vin_table, engine):

    gd_list = pd.read_sql(listing_table, engine)
    gd_link = pd.read_sql(links_table, engine)
    gd_vin = pd.read_sql(vin_table, engine)

    reg_ref = pd.read_sql('region_reference', engine)
    
    merged_linkls = pd.merge(pd.merge(gd_list, gd_link, on='link', how='left'),reg_ref, on='region_url')

    mm_dropped = merged_linkls.drop_duplicates(subset=['VIN'])
    vn_no_dupe = gd_vin.drop_duplicates(subset=['VIN'])

    f_df = pd.merge(vn_no_dupe, mm_dropped,  on='VIN', how='left')

    f_df.to_sql(main_data, engine)

    print(main_data + ' created')

def reject_more_values(gd_links, bd_links, gd_listings, bd_listings, gd_vins, bd_vins, engine):
    with engine.connect() as conn:
        
        # Step 1: Move rows from gd_listings to bd_listings if VIN is in bd_vins
        move_to_bd_listings_in_bd_vins = text(f"""
        INSERT INTO {bd_listings}
        SELECT * FROM {gd_listings}
        WHERE "VIN" IN (SELECT "VIN" FROM {bd_vins});
        
        DELETE FROM {gd_listings}
        WHERE "VIN" IN (SELECT "VIN" FROM {bd_vins});
        """)
        
        # Step 2: Move rows from gd_links to bd_links if link is in bd_listings
        move_to_bd_links_in_bd_listings = text(f"""
        INSERT INTO {bd_links}
        SELECT * FROM {gd_links}
        WHERE "link" IN (SELECT "link" FROM {bd_listings});
        
        DELETE FROM {gd_links}
        WHERE "link" IN (SELECT "link" FROM {bd_listings});
        """)
        
        # Step 3: Move rows from gd_listings to bd_listings if VIN is NOT in gd_vins
        move_to_bd_listings_not_in_gd_vins = text(f"""
        INSERT INTO {bd_listings}
        SELECT * FROM {gd_listings}
        WHERE "VIN" NOT IN (SELECT "VIN" FROM {gd_vins});
        
        DELETE FROM {gd_listings}
        WHERE "VIN" NOT IN (SELECT "VIN" FROM {gd_vins});
        """)
        
        # Step 4: Move rows from gd_links to bd_links if link is NOT in gd_listings
        move_to_bd_links_not_in_gd_listings = text(f"""
        INSERT INTO {bd_links}
        SELECT * FROM {gd_links}
        WHERE "link" NOT IN (SELECT "link" FROM {gd_listings});
        
        DELETE FROM {gd_links}
        WHERE "link" NOT IN (SELECT "link" FROM {gd_listings});
        """)
        
        # Execute and commit each operation
        conn.execute(move_to_bd_listings_in_bd_vins)
        conn.commit()
        
        conn.execute(move_to_bd_links_in_bd_listings)
        conn.commit()
        
        conn.execute(move_to_bd_listings_not_in_gd_vins)
        conn.commit()
        
        conn.execute(move_to_bd_links_not_in_gd_listings)
        conn.commit()
        
        print("Rows moved successfully based on VIN and link conditions.")


engine = create_engine('postgresql+psycopg2://postgres:p33Gritz!!@localhost:5432/cars')
dfdf = reject_more_values('links_accepted_2024_11_02', 'links_rejected', 'listings_accepted_2024_11_02', 'listings_rejected', 'vins_accepted_2024_11_02', 'region_reference', 'new_data_2024_11_02', engine)

# TODO: backup _accepted_2024
# TODO: retrain model
# TODO: dockerize

def do_more(remaining_links, listings_accepted, listings_rejected, vins_accepted, vins_rejected, all_vins, datestr, engine):

    vin_batch = [] 

    for link in remaining_links:
        try:
            parsed_df = link_parser(link)
            if parsed_df is not None:
                valid_df, reject_df = clean_listing_output(parsed_df)

                # Insert valid listings into target table
                with engine.connect() as conn:
                    if not valid_df.empty:
                        good_vin_df = pd.read_sql(f'SELECT "VIN" FROM {vins_accepted}', con=engine)
                        bad_vin_df = pd.read_sql(f'SELECT "VIN" FROM {vins_rejected}', con=engine)

                        # Concatenate VIN values from all tables into a single list
                        all_vins = all_vins + good_vin_df['VIN'].tolist() + bad_vin_df['VIN'].tolist()

                        vin = valid_df.at[0, 'VIN']

                        if (vin not in vin_batch) and (vin not in all_vins):
                            valid_df.to_sql(listings_accepted, con=conn, if_exists='append', index=False)
                            vin_batch.append(vin)
                            print(f"VIN added to batch: {vin}")

                        else:
                            print(f"VIN already in database: {vin}, added to {listings_rejected}")
                            valid_df.to_sql(listings_rejected, con=conn, if_exists='append', index=False)

                        # If batch is full, process it
                        if len(vin_batch) == 50:
                            print("Processing full batch:")
                            print(len(vin_batch))

                            # Decode batch and clean
                            vin_df = batch_vin(';'.join(vin_batch))
                            valid_vin_df, reject_vin_df = clean_vin_output(vin_df)

                            # Store valid VINs in database
                            valid_vin_df.to_sql(vins_accepted, engine, if_exists='append', index=False)
                            print(f"Added {len(valid_vin_df)} records to {vins_accepted}")

                            # Store rejected VINs in database with scrape date
                            reject_vin_df['date_scraped'] = datestr
                            reject_vin_df.to_sql(vins_rejected, engine, if_exists='append', index=False)
                            print(f"Added {len(reject_vin_df)} records to {vins_rejected}")

                            # Clear batch after processing
                            vin_batch.clear()
                            print("Cleared vin_batch after processing.")
                            print(len(vin_batch))

                    # Insert reject records
                    if not reject_df.empty:     
                        reject_df['VIN'] = None
                        reject_df['odometer'] = None
                        reject_df['date_scraped'] = datestr

                        reject_df.to_sql(listings_rejected, con=conn, if_exists='append', index=False)
                        print(f"Added reject records to {listings_rejected}")

                print(f"Processed link: {link}")
            else:
                print(f"Link {link} returned blocked or empty data.")

        except Exception as e:
            print(f"Error for {link}: {e}")

        finally:
            time.sleep(1)  # Avoid hitting the server too quickly
