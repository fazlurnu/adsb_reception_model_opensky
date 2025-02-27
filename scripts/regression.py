# %% 
import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from sklearn.metrics import mean_squared_error

# %%

# Haversine formula for distance calculation in nautical miles
def haversine_NM(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 3440.065 * 2 * np.arcsin(np.sqrt(a))

# Get the largest file in a folder
def get_largest_file(sensor_folder):
    files = [os.path.join(sensor_folder, f) for f in os.listdir(sensor_folder) if os.path.isfile(os.path.join(sensor_folder, f))]
    return max(files, key=os.path.getsize) if files else None

# Count airports within the convex hull of sensor positions
def get_nb_airport(sensor_id, df_airport, sensor_lon=2, sensor_lat=50):
    sensor_folder = f"../sensor_pos_data/{sensor_id}"
    largest_file = get_largest_file(sensor_folder)
    if not largest_file:
        return 0

    df_pos = pd.read_csv(largest_file)
    df_pos['distance_NM'] = df_pos.apply(lambda row: haversine_NM(row['lat'], row['lon'], sensor_lat, sensor_lon), axis=1)
    df_pos = df_pos[df_pos['distance_NM'] < 300]

    points = df_pos[['lat', 'lon']].values
    hull_polygon = Polygon(points[ConvexHull(points).vertices])

    return sum(hull_polygon.contains(Point(lat, lon)) for lat, lon in df_airport[['latitude_deg', 'longitude_deg']].values)

# Load airport data
df_airport = pd.read_csv('airports.csv')
df_airport = df_airport[(df_airport['name'] != 'SPAM') & (df_airport['scheduled_service'] == 'yes') & (df_airport['type'].isin(['medium_airport', 'large_airport']))]

# Load sensor data
df_sensor = pd.read_csv('../sensors/sensor_loc.csv')

# Aggregate sensor reception data
csv_files = glob.glob('../sensors_reception_prob/reception_prob/receptionprob_*.csv')
df_list = []

for file in csv_files:
    sensor_id = file.split('/')[-1].split('_')[1].split('.')[0]
    df = pd.read_csv(file)
    df = df[df['data_count'] > 2500]

    df['distance_avg'] = df['distance_bin'].apply(lambda x: (float(x.split(',')[0][1:]) + float(x.split(',')[1][:-1])) / 2)
    df['traffic_avg'] = df['traffic_bin'].apply(lambda x: (float(x.split(',')[0][1:]) + float(x.split(',')[1][:-1])) / 2)

    sensor_data = df_sensor[df_sensor['serial'] == int(sensor_id)]
    df['CR'] = df['max_dist_NM'].unique()[0]
    df['airport'] = get_nb_airport(sensor_id, df_airport, sensor_data['lon'].iloc[0], sensor_data['lat'].iloc[0])
    df['sensor_id'] = int(sensor_id)
    df['type'] = sensor_data['type'].iloc[0]
    df_list.append(df)

aggregated_df = pd.concat(df_list, ignore_index=True)
aggregated_df.to_csv('../sensors_reception_prob/processed_sensor_probability.csv', index = False)
# %% 

if 'aggregated_df' not in globals():
    aggregated_df = pd.read_csv('../sensors_reception_prob/processed_sensor_probability.csv')
    
# Initialize results dataframe
results = []

# Function to fit and evaluate reception model
def fit_and_evaluate_model(sensor_type):
    for sensor_to_test in aggregated_df[aggregated_df['type'] == sensor_type]['sensor_id'].unique():
        train_df = aggregated_df[(aggregated_df['sensor_id'] != sensor_to_test) & (aggregated_df['type'] == sensor_type)]
        test_df = aggregated_df[aggregated_df['sensor_id'] == sensor_to_test]
    
        features = ['distance_avg', 'CR', 'traffic_avg', 'airport']
        target = '2500_reception_probability'
        X_train, y_train = train_df[features].values, train_df[target].values
        X_test, y_test = test_df[features].values, test_df[target].values

        def reception_model(X,
                            a1_t, a1_a, a1_intercept, 
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

        best_rmse, best_popt = np.inf, None
        for _ in range(100):
            try:
                initial_guess = np.random.uniform(-1, 1, size=12)  # Reduce the number of parameters
                popt, _ = curve_fit(reception_model, X_train, y_train, p0=initial_guess)
                rmse = np.sqrt(mean_squared_error(y_train, reception_model(X_train, *popt)))

                if rmse < best_rmse:
                    best_rmse, best_popt = rmse, popt
            except RuntimeError:
                continue

        y_pred_test = reception_model(X_test, *best_popt)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        results.append({
            "Sensor Type": sensor_type,
            "Sensor ID": sensor_to_test,
            "Best RMSE (Train)": best_rmse,
            "RMSE (Test)": rmse_test,
            "Best Parameters": best_popt
        })

# Fit and evaluate for both sensor types
fit_and_evaluate_model('Radarcape')
fit_and_evaluate_model('dump1090')

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print(results_df)

results_df.to_csv('../model/regression_models.csv', index=False)