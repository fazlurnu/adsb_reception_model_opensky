import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os

from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon

import cartopy
import cartopy.crs as ccrs

from sklearn.metrics import root_mean_squared_error

# Haversine formula for distance calculation in nautical miles
def haversine_NM(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 3440.065  # Radius of earth in nautical miles
    return R * c

def get_largest_file_in_folder(sensor_folder):
    largest_file = None
    max_size = 0

    # Iterate over all files in the directory
    for file_name in os.listdir(sensor_folder):
        file_path = os.path.join(sensor_folder, file_name)
        
        # Check if it's a file and not a subdirectory
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > max_size:
                max_size = file_size
                largest_file = file_path
    
    return largest_file

def get_nb_airport(sensor_id, df_airport, sensor_lon = 2, sensor_lat = 50):
    width = 800
    height = 800
    dpi = 96

    sensor_folder = f"../sensor_pos_data/{sensor_id}"
    largest_file = get_largest_file_in_folder(sensor_folder)

    df_pos = pd.read_csv(largest_file)
    df_pos['distance_NM'] = df_pos.apply(lambda row: haversine_NM(row['lat'], row['lon'], sensor_lat, sensor_lon), axis=1)
    df_pos = df_pos[df_pos['distance_NM'] < 300]

    points = df_pos[['lat', 'lon']].values
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_polygon = Polygon(hull_points)

    airport_points = df_airport[['latitude_deg', 'longitude_deg']].values

    # Check how many airports are inside the convex hull
    inside_hull = [hull_polygon.contains(Point(lat, lon)) for lat, lon in airport_points]
    num_airports_inside = np.sum(inside_hull)
    
    # Define the projection used to display the map, centered at the first sensor location:
    proj = ccrs.Orthographic(central_longitude=sensor_lon, central_latitude=sensor_lat)

    return num_airports_inside

df_airport = pd.read_csv('airports.csv')
df_airport = df_airport[(df_airport['name'] != 'SPAM')]
df_airport = df_airport[df_airport['scheduled_service'] == 'yes']
df_airport = df_airport[(df_airport['type'] == 'medium_airport') | (df_airport['type'] == 'large_airport')]

# Step 1: Load and aggregate CSV files
csv_files = glob.glob('../sensors_reception_prob/reception_prob/receptionprob_*.csv')  # Update with your file path pattern

# Initialize a list to store DataFrames
df_list = []

df_sensor = pd.read_csv('../sensors/sensor_loc.csv')

# Step 2: Loop through each file, load, assign CR, and add to the list
for i, file in enumerate(csv_files):
    sensor_id = file.split('/')[-1].split('_')[1].split('.')[0]
    
    df = pd.read_csv(file)
    df = df[df['data_count'] > 2500]
    # Clean the distance and traffic bins to get the average values
    df['distance_avg'] = df['distance_bin'].apply(lambda x: (float(x.split(',')[0][1:]) + float(x.split(',')[1][:-1])) / 2)
    df['traffic_avg'] = df['traffic_bin'].apply(lambda x: (float(x.split(',')[0][1:]) + float(x.split(',')[1][:-1])) / 2)
    
    # Add CR value as a new column
    df['CR'] = df['max_dist_NM'].unique()[0]
    sensor_lat = df_sensor[df_sensor['serial'] == int(sensor_id)]['lat'].iloc[0]
    sensor_lon = df_sensor[df_sensor['serial'] == int(sensor_id)]['lon'].iloc[0]
    
    df['airport'] = get_nb_airport(sensor_id, df_airport, sensor_lon, sensor_lat)
    df['sensor_id'] = sensor_id
    # Append to the list
    df_list.append(df)

# Step 3: Concatenate all DataFrames into a single DataFrame
aggregated_df = pd.concat(df_list, ignore_index=True)

print(aggregated_df)