# %%

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
    print(f'Processing {i+1}/{len(csv_files)} files')
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
    df['sensor_id'] = int(sensor_id)
    df['type'] = df_sensor[df_sensor['serial'] == int(sensor_id)]['type'].iloc[0]
    
    # Append to the list
    df_list.append(df)

# Step 3: Concatenate all DataFrames into a single DataFrame
aggregated_df = pd.concat(df_list, ignore_index=True)

# %% 
# Step 1: Split data by sensor_id
sensor_to_test = aggregated_df['sensor_id'].unique()[4]  # Choose one sensor_id for testing
train_df = aggregated_df[(aggregated_df['sensor_id'] != sensor_to_test) & (aggregated_df['type'] == 'dump1090')]
test_df = aggregated_df[aggregated_df['sensor_id'] == sensor_to_test]

# Step 2: Define features and target
features = ['distance_avg', 'CR', 'traffic_avg', 'airport']  # Add relevant features
target = '2500_reception_probability'

X_train = train_df[features].values
y_train = train_df[target].values

X_test = test_df[features].values
y_test = test_df[target].values

# Step 3: Define model function
def reception_model_variable_a(X, a1_t, a1_a, a1_intercept, 
                               a2_t, a2_a, a2_intercept, 
                               a3_t, a3_a, a3_intercept, 
                               a4_t, a4_a, a4_intercept):
    d, CR, traffic_avg, airport = X.T  # Unpack the columns
    a1 = a1_t * traffic_avg + a1_a * airport + a1_intercept
    a2 = a2_t * traffic_avg + a2_a * airport + a2_intercept
    a3 = a3_t * traffic_avg + a3_a * airport + a3_intercept
    a4 = a4_t * traffic_avg + a4_a * airport + a4_intercept
    ratio = d / CR
    return np.exp(-3 * ratio**2) * (1 + a1 * ratio + a2 * ratio**2 + a3 * ratio**3 + a4 * ratio**4)

# Step 4: Perform multiple iterations of curve fitting with random initial guesses
iterations = 100
best_rmse = np.inf
best_popt = None

for i in range(iterations):
    # Generate random initial guess for parameters
    initial_guess = np.random.uniform(-1, 1, size=12)  # Adjust as needed

    try:
        # Perform curve fitting on the training data
        popt, _ = curve_fit(reception_model_variable_a, X_train, y_train, p0=initial_guess)

        # Use the fitted parameters to predict on the training data
        reception_pred_train = reception_model_variable_a(X_train, *popt)

        # Calculate RMSE for the training data
        rmse_train = np.sqrt(mean_squared_error(y_train, reception_pred_train))

        # If this fit is better, save the parameters and RMSE
        if rmse_train < best_rmse:
            best_rmse = rmse_train
            best_popt = popt

    except RuntimeError as e:
        print(f"Iteration {i+1} failed: {e}")

# Step 5: Evaluate on the test data
reception_pred_test = reception_model_variable_a(X_test, *best_popt)

# Calculate RMSE for the test data
rmse_test = np.sqrt(mean_squared_error(y_test, reception_pred_test))

# Print results
print(f"Best RMSE on training data: {best_rmse}")
print(f"RMSE on test data (sensor {sensor_to_test}): {rmse_test}")
print(f"Best fitted parameters: {best_popt}")