import argparse

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from math import *

import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from math import factorial, e

from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description="Process reception probability and optionally plot results.")
parser.add_argument("--plot", type=bool, default=False, help="Set to True to generate and show the plot.")
args = parser.parse_args()

# Function to calculate d0 without interference
def get_d0(r, r0, k):
    return 1 - (r/r0)**k

# Function to calculate dA with ATCRBS interference
def get_dA(r, r0, x):
    k = [3.94, 3.37, 3.23, 2.75, 2.66]
    return 1 - (r/r0)**(k[x-1])

# Function to calculate dS with Mode S interference
def get_dS(r, r0prime, k):
    return 1 - (r/r0prime)**k

# Function to calculate probability p_x_mA with ATCRBS
def get_p_x_mA(x, r, rho, N_atcrbs):
    tau_identify = 20.3e-6
    tau_altitude = 20.3e-6
    nb_ac = rho * r
    prf = 1.65
    mA = nb_ac * (prf * N_atcrbs * tau_identify + prf * N_atcrbs * tau_altitude)
    p_x_mA = (mA**x / factorial(x)) * e**(-mA)
    return p_x_mA

# Function to calculate probability p_x_mS with Mode S
def get_p_x_mS(x, r, rho, N_tcas_s, N_mode_s):
    tau_short = 64e-6
    tau_long = 120e-6
    nb_ac = rho * r
    lb_0 = N_tcas_s * nb_ac
    lb_4 = N_mode_s * 1/3 * nb_ac
    lb_11 = N_mode_s * 1/6 * nb_ac
    lb_17 = 6.2 * nb_ac
    mS = lb_0 * tau_short + lb_4 * tau_short + lb_11 * tau_short + lb_17 * tau_long
    p_x_mS = (mS**x / factorial(x)) * e**(-mS)
    return p_x_mS

# Main function to calculate the reception probability
def get_prob_chung(r, rho, params, r0=96.6, r0prime=30, k=6.4314):
    N_atcrbs, N_tcas_s, N_mode_s = params

    d0 = get_d0(r, r0, k)

    dA = 0
    for x in range(1, 6):
        dA += get_dA(r, r0, x)

    dS = get_dS(r, r0prime, k)

    p_0_mA = get_p_x_mA(0, r, rho, N_atcrbs)
    p_0_mS = get_p_x_mS(0, r, rho, N_tcas_s, N_mode_s)
    p_1_mS = get_p_x_mS(1, r, rho, N_tcas_s, N_mode_s)

    first_term_1 = 0
    for x in range(1, 6):
        first_term_1 += get_dA(r, r0, x) * get_p_x_mA(x, r, rho, N_atcrbs)

    term1 = d0 * p_0_mA * p_0_mS
    term2 = first_term_1 * p_0_mS
    term3 = dS * p_1_mS * p_0_mA
    term4 = first_term_1 * p_1_mS * dS
    
    return term1 + term2 + term3 + term4

def final_prob_chung(r, traf, r0, N_atcrbs):
    rho = traf / r0
    k = 6.4314

    M_atcrbs_per_mode_s = 2
    N_mode_s = N_atcrbs * M_atcrbs_per_mode_s
    N_tcas_s = traf
    params = N_atcrbs, N_tcas_s, N_mode_s
        
    return get_prob_chung(r, rho, params, r0, r0prime=r, k=6.4314)

def get_circular_sensor(directory_path):
    sensor_ids = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            parts = filename.split('/')[-1]  # Get the last part 'sensor123_2024-10-04_data.csv'
            id_date = parts.split('_')  # Split by '_'
            sensor_id = int(id_date[0])  # Extract sensor ID
            sensor_ids.append(sensor_id)

    return sensor_ids

def get_largest_file(sensor_folder):
    files = [os.path.join(sensor_folder, f) for f in os.listdir(sensor_folder) if os.path.isfile(os.path.join(sensor_folder, f))]
    return max(files, key=os.path.getsize) if files else None

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

def final_prob_chung(r, traf, r0, N_atcrbs):
    rho = traf / r0
    k = 6.4314

    M_atcrbs_per_mode_s = 2
    N_mode_s = N_atcrbs * M_atcrbs_per_mode_s
    N_tcas_s = traf
    params = N_atcrbs, N_tcas_s, N_mode_s
        
    return get_prob_chung(r, rho, params, r0, r0prime=r, k=6.4314)

df_airport = pd.read_csv('../airports/airports.csv')
df_airport = df_airport[(df_airport['name'] != 'SPAM')]
df_airport = df_airport[df_airport['scheduled_service'] == 'yes']
df_airport = df_airport[(df_airport['type'] == 'medium_airport') | (df_airport['type'] == 'large_airport')]

# Step 1: Load and aggregate CSV files
csv_files = glob.glob('../sensors_reception_prob/reception_prob/receptionprob_*.csv')

# Initialize a list to store DataFrames
df_list = []

df_sensor = pd.read_csv('../sensors/sensor_loc.csv')

# Get processed features
df_processed = pd.read_csv('../sensors_reception_prob/processed_sensor_probability.csv')

# Step 2: Loop through each file, load, assign CR, and add to the list
for i, file in enumerate(csv_files):
    sensor_id = file.split('/')[-1].split('_')[1].split('.')[0]
    
    df = pd.read_csv(file)
    df = df[df['data_count'] > 2500]
    # Clean the distance and traffic bins to get the average values
    df['distance_avg'] = df['distance_bin'].apply(lambda x: (float(x.split(',')[0][1:]) + float(x.split(',')[1][:-1])) / 2)
    df['traffic_avg'] = df['traffic_bin'].apply(lambda x: (float(x.split(',')[0][1:]) + float(x.split(',')[1][:-1])) / 2)
    
    # Add CR value as a new column
    sensor_data = df_sensor[df_sensor['serial'] == int(sensor_id)]
    df['CR'] = df['max_dist_NM'].unique()[0]
    df['airport'] = df_processed[df_processed['sensor_id'] == int(sensor_id)]['airport'].dropna().iloc[0]
    df['sensor_id'] = sensor_id
    # Append to the list
    df_list.append(df)


# Step 3: Concatenate all DataFrames into a single DataFrame
aggregated_df = pd.concat(df_list, ignore_index=True)

aggregated_df['prob_chung'] = aggregated_df.apply(lambda row: final_prob_chung(row['distance_avg'], 
                                                         row['traffic_avg'], 
                                                         row['CR'], 
                                                         row['airport']), axis=1)

aggregated_df.to_csv('../sensors_reception_prob/chungs_reception_probability.csv', index = False)
print('File is saved to: ../sensors_reception_prob/chungs_reception_probability.csv')

rmse_dict = {'sensor': [], 'rmse': []}

for i, file in enumerate(csv_files):
    sensor_to_test = file.split('/')[-1].split('_')[1].split('.')[0]

    test_df = aggregated_df[aggregated_df['sensor_id'] == sensor_to_test]

    # Extract the two columns for RMSE calculation
    true_values = test_df['2500_reception_probability'].values
    estimated_values = test_df['prob_chung'].values

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(true_values, estimated_values)) * 100

    rmse_dict['sensor'].append(sensor_to_test)
    rmse_dict['rmse'].append(rmse)

    if args.plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(test_df['2500_reception_probability'], test_df['prob_chung'])

        plt.xlim(-0.025, 1.025)
        plt.ylim(-0.025, 1.025)

        plt.xlabel('True Reception Probability [-]')
        plt.ylabel('Estimated Reception Probability [-]')
        plt.title(f'Sensor ID: {sensor_to_test}. RMSE: {rmse:.2f}%')

        plt.plot([0, 1], [0, 1], 'r', label='Error = 0')

        plt.gca().set_aspect('equal', adjustable='box')

        plt.legend()
        plt.show()

df_rmse = pd.DataFrame(rmse_dict)
df_rmse.to_csv('../model/rmse_chung.csv', index = False)