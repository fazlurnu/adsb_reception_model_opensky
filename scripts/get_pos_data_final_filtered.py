from pyopensky.trino import Trino
from sqlalchemy.exc import OperationalError

import pyopensky
import pandas as pd
from datetime import datetime, timedelta

import numpy as np
from scipy.spatial import ConvexHull

import os
import re

# Function to calculate circularity
def calculate_circularity(area, perimeter):
    return (4 * np.pi * area) / (perimeter ** 2)

def is_in_europe(lat, lon):
    return 35.0 <= lat <= 71.0 and -25.0 <= lon <= 45.0

def get_sensor_list(directory):

    # Initialize a list to store the directory names
    dir_list = []

    # Walk through the directory and process all files and directories
    for dirpath, dirnames, filenames in os.walk(directory):
        # Check and save all subdirectories
        for dirname in dirnames:
            full_dir_path = os.path.join(dirpath, dirname)
            if os.path.isdir(full_dir_path):
                dir_list.append(dirname)  # Add directory name to the list

    return dir_list

import re

def get_circular_sensor(directory_path):
    sensor_ids = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            parts = filename.split('/')[-1]  # Get the last part 'sensor123_2024-10-04_data.csv'
            id_date = parts.split('_')  # Split by '_'
            sensor_id = int(id_date[0])  # Extract sensor ID
            sensor_ids.append(sensor_id)

    return sensor_ids



# Initialize Trino instance
trino = Trino()

# Load sensor location data
df_sensor_loc = pd.read_csv('../sensors/sensor_loc.csv')

directory_path = "../sensor_circularity_quick_test/sensor_is_center_and_circular"  # Change this to your actual directory
df_sensor_interest = get_circular_sensor(directory_path)

df_sensor_loc = df_sensor_loc[df_sensor_loc['serial'].isin(df_sensor_interest)]

# Get the list of sensor IDs (with the first ID being -1408237098)
sensor_ids = list(df_sensor_loc['serial'])
print(sensor_ids)

# Define constants
NM_per_degree = 60
max_radius_NM = 300

# Set how many hours required to check the circularity
minute_interval = 15
hours_required = 0.5
df_per_hours = 60/minute_interval
df_required = df_per_hours * hours_required

# Define start and end dates for iteration
begin_hour = 0
duration = 2

start_date = datetime(2022, 1, 1, begin_hour, 0, 0)
end_date = datetime(2022, 12, 31, begin_hour, 0, 0)

# Iterate over each sensor in the sensor_ids list
circular_coverage_sensor = {}

# Receiver counter
receiver_counter = 0

for sensor_id in sensor_ids:
    receiver_counter += 1
    print("-----------Define the bound-----------")
    
    # Get sensor latitude and longitude
    sensor_lat = float(df_sensor_loc[df_sensor_loc['serial'] == sensor_id]['lat'].iloc[0])
    sensor_lon = float(df_sensor_loc[df_sensor_loc['serial'] == sensor_id]['lon'].iloc[0])

    # Set bounds based on radius in nautical miles
    west_bound = sensor_lon - (max_radius_NM / NM_per_degree)
    east_bound = sensor_lon + (max_radius_NM / NM_per_degree)
    north_bound = sensor_lat + (max_radius_NM / NM_per_degree)
    south_bound = sensor_lat - (max_radius_NM / NM_per_degree)

    bounds = (west_bound, south_bound, east_bound, north_bound)  # Define bounds as a tuple
    print(f"Bounds for sensor {sensor_id}: {bounds}")

    # Reset the start date for each sensor
    current_date = start_date

    # Loop over each day
    while current_date < end_date:
        # Get a formatted date string for file naming
        begin_time = current_date.strftime('%Y%m%d')

        # Create an empty list to store data for the day
        all_data = []

        # For breaking code. Sounds cool, eh?
        dataframe_empty = False
        circularity_checked = False
        circularity_broken = False

        print(f"Fetching data for: {begin_time}")

        # Inner loop: Loop every 15 minutes within the day (from 00:00:00 to 23:45:00)
        time_of_day = current_date
        while time_of_day < current_date + timedelta(hours=duration):
            print(f"{receiver_counter}/{len(sensor_ids)}. Current time: {time_of_day.strftime('%H:%M:%S')}")

            # Set the start and stop times for each 1-minute interval
            start_time = time_of_day
            stop_time = start_time + timedelta(minutes=1)  # 1 minute later

                        # Fetch raw data for the current time interval
            try:
                raw_data = trino.rawdata(
                    start=start_time,
                    stop=stop_time,
                    serials=sensor_id,
                    Table=pyopensky.schema.PositionData4,
                    # bounds=bounds,
                    cached=True,
                    extra_columns=(
                        pyopensky.schema.PositionData4.sensors,  # Sensors
                        pyopensky.schema.PositionData4.mintime,  # Minimum time
                        pyopensky.schema.PositionData4.maxtime,  # Maximum time
                        pyopensky.schema.PositionData4.rawmsg,   # Raw message
                        pyopensky.schema.PositionData4.icao24,   # ICAO24 code
                        pyopensky.schema.PositionData4.lat,      # Latitude
                        pyopensky.schema.PositionData4.lon,      # Longitude
                        pyopensky.schema.PositionData4.alt       # Altitude
                    )
                )

                # Convert the raw data to a DataFrame
                df = pd.DataFrame(raw_data)
                if not df.empty:
                    # Process the DataFrame: Extract the 'serial' field from 'sensors'
                    df['serial'] = df['sensors'].apply(lambda sensor_list: next((s['serial'] for s in sensor_list if s['serial'] == sensor_id), None))
                    df['mintime_sensor'] = df['sensors'].apply(lambda sensor_list: next((s['mintime'] for s in sensor_list if s['serial'] == sensor_id), None))
                    df['maxtime_sensor'] = df['sensors'].apply(lambda sensor_list: next((s['maxtime'] for s in sensor_list if s['serial'] == sensor_id), None))

                    # Select the relevant columns and drop rows with missing values
                    df = df[['serial', 'mintime_sensor', 'maxtime_sensor', 'rawmsg', 'icao24', 'lat', 'lon', 'alt']].dropna()
                    
                    # Append the DataFrame to the list for the day
                    all_data.append(df)

                # Increment time_of_day by 15 minutes
                time_of_day += timedelta(minutes=minute_interval)
            except:
                print("Error here")

        # Concatenate all data for the day into a single DataFrame
        target_folder = "../sensor_pos_data/" + str(sensor_id)

        if not os.path.exists(target_folder):
                os.makedirs(target_folder)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)

            # Save the DataFrame for the day into a CSV file
            if (not final_df.empty) and (not circularity_broken):
                filename = f'{target_folder}/{sensor_id}_{begin_time}_{begin_hour}_{duration}_data.csv'
                final_df.to_csv(filename, index=False)

                # Print save status
                print(f"Data saved to {target_folder}/{sensor_id}_{begin_time}_{begin_hour}_{duration}_data.csv")
            else:
                print(f"Data for {sensor_id} at {begin_time} is empty, or not circular, not saved")

        else:
            print(f"No data available for {begin_time}")

        # Increment current_date to the next day
        current_date += timedelta(days=1)

print("Data fetching complete.")
