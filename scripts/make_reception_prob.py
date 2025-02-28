# %% 

import pandas as pd

import matplotlib.pyplot as plt

import pandas as pd
from datetime import datetime, timedelta

import numpy as np

import glob
import os

import seaborn as sns

# Functions here
def haversine_NM(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of Earth in kilometers. Use 6371 for kilometers, 3956 for miles.
    r = 6371  # in kilometers
    km_to_NM = 1/1.852
    return c * r * km_to_NM # Distance in kilometers

def get_circular_sensor(directory_path):
    sensor_ids = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            parts = filename.split('/')[-1]  # Get the last part 'sensor123_2024-10-04_data.csv'
            id_date = parts.split('_')  # Split by '_'
            sensor_id = int(id_date[0])  # Extract sensor ID
            sensor_ids.append(sensor_id)

    return sensor_ids

# Path to the folder where the .csv files are stored
# sensor_id = 1998349148
df_sensor_loc = pd.read_csv('../sensors/sensor_loc.csv')

directory_path = "sensor_is_center_and_circular"  # Change this to your actual directory
df_sensor_interest = get_circular_sensor(directory_path)

df_sensor_loc = df_sensor_loc[df_sensor_loc['serial'].isin(df_sensor_interest)]

# Get the list of sensor IDs (with the first ID being -1408237098)
sensor_ids = list(df_sensor_loc['serial'])
print(sensor_ids)

count_sensor = 0

maximum_time_update = 5

for sensor_id in sensor_ids:
    count_sensor += 1
    print(f"Making reception prob: {count_sensor}/{len(sensor_ids)}")
    # Get the last reception probability value
    tolerance = 0.025
    minimum_nb_data = 5000

    folder_path = os.path.dirname(f"../sensor_pos_data/{sensor_id}/")

    # Use glob to get a list of all .csv files in the same folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # Concatenate all CSV files into a single DataFrame
    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)


    df_sensor_loc = pd.read_csv('../sensors/sensor_loc.csv')
    df['mintime_sensor_date'] = pd.to_datetime(df['mintime_sensor'], unit='s')

    sensor_lat = df_sensor_loc[df_sensor_loc['serial'] == sensor_id]['lat'].iloc[0]
    sensor_lon = df_sensor_loc[df_sensor_loc['serial'] == sensor_id]['lon'].iloc[0]

    print(sensor_id, sensor_lat, sensor_lon)
    df['distance_NM'] = df.apply(lambda row: haversine_NM(row['lat'], row['lon'], sensor_lat, sensor_lon), axis=1)

    distance_threshold = df['distance_NM'].quantile(0.9999)
    distance_threshold = 300

    df = df[df['distance_NM'] < distance_threshold] ## filter out outliers, theoretically ADS-B receiver max dist is 250
    # df = df.drop_duplicates(subset='rawmsg')

    # df[df['distance_NM'] > 160].to_csv(f'{sensor_id}_more_than_160.csv')
    maximum_distance = df['distance_NM'].max()

    print("Maximum disance: ", maximum_distance)

    # Calculate the time difference between consecutive rows in minutes
    df['time_diff'] = df['mintime_sensor_date'].diff().dt.total_seconds().div(60)
    df = df[~df['time_diff'].isna()]

    # Create a new grouping column: Start a new group if the time difference exceeds 2 minutes
    df['group'] = (df['time_diff'] > 2).cumsum()

    # Group by the new 'group' column and count the unique 'icao24' values per group
    grouped_df = df.groupby('group').agg(
        nb_traffic=('icao24', pd.Series.nunique),  # Count unique icao24 per group
    ).reset_index()

    # Merge the nb_traffic values back onto the original DataFrame based on the group column
    df = pd.merge(df, grouped_df, on='group', how='left')

    # Drop the 'group' and 'time_diff' columns if you don't need them anymore
    df = df.drop(columns=['group', 'time_diff'])

    # Assuming df is your DataFrame
    # Define bins for 'distance_NM' (10 NM intervals) and 'nb_traffic' (10 interval bins)
    max_distance = 300
    max_traffic = 300
    interval_distance = 10
    interval_traffic = 10

    distance_bins = pd.cut(df['distance_NM'], bins=range(0, max_distance + interval_distance, interval_distance))
    traffic_bins = pd.cut(df['nb_traffic'], bins=range(0, max_traffic + interval_traffic, interval_traffic))

    # Create a new DataFrame with binned categories
    df['distance_bin'] = distance_bins
    df['traffic_bin'] = traffic_bins

    average_distance_per_bin = df.groupby('distance_bin')['distance_NM'].mean().reset_index()
    print(average_distance_per_bin)
    df['average_distance'] = average_distance_per_bin['distance_NM']

    # Create a pivot table or crosstab to count how many rows fall into each bin
    binned_data = pd.crosstab(df['distance_bin'], df['traffic_bin'])
    binned_data.to_csv(f'nb_data_{sensor_id}.csv', index=True)

    binary_binned_data = binned_data.applymap(lambda x: 1 if x > minimum_nb_data else 0)

    # Create the heatmap using the binary data
    plt.figure(figsize=(12, 8))
    sns.heatmap(binary_binned_data, annot=False, cmap="Blues", cbar=True, vmax = 1, vmin = 0)

    # Add titles and labels
    plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.xlabel('Traffic Bin', labelpad=20)
    plt.xticks(rotation=90)
    plt.xlabel('Traffic Bin')
    plt.ylabel('Distance Bin')

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'nb_data_{sensor_id}_bin.png')

    plt.figure(figsize=(12, 8))
    sns.heatmap(binned_data, annot=False, cmap="Blues", cbar=True, vmax = 10000)

    # Add titles and labels
    plt.xlabel('Traffic Bin', labelpad=20)
    plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.xticks(rotation=90)
    # plt.title('Heatmap of Distance Bin vs Traffic Bin')
    plt.xlabel('Traffic Bin')
    plt.ylabel('Distance Bin')

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'nb_data_{sensor_id}_heatmap.png')

    plt.figure(figsize=(12, 8))
    sns.heatmap(binned_data, annot=False, cmap="Blues", cbar=True, vmax = 2500, vmin = 0)

    # Add titles and labels
    plt.xlabel('Traffic Bin', labelpad=20)
    plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.xticks(rotation=90)
    # plt.title('Heatmap of Distance Bin vs Traffic Bin')
    plt.xlabel('Traffic Bin')
    plt.ylabel('Distance Bin')

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'nb_data_{sensor_id}_heatmap_2500.png')

    filtered_df = df

    filtered_df = filtered_df.groupby('icao24', group_keys=True).apply(lambda x: x)

    filtered_df['updateinterval'] = filtered_df['mintime_sensor'].diff()
    filtered_df = filtered_df[(filtered_df['updateinterval'] <= maximum_time_update) & (filtered_df['updateinterval'] > 0)]
    # filtered_df = filtered_df.dropna()

    # Ungroup and reset index to flatten the DataFrame
    filtered_df = filtered_df.reset_index(drop=True)

    # %%

    # Use numpy to calculate histogram values (frequencies and bin edges)
    counts, bin_edges = np.histogram(filtered_df['updateinterval'], bins=maximum_time_update * 10)

    # Define the total area (sum of counts) between 0 and 5 seconds
    total_area = counts[(bin_edges[:-1] >= 0) & (bin_edges[:-1] <= maximum_time_update)].sum()

    # Define the area between 0.25 and 0.75 seconds
    area_025_to_075 = counts[(bin_edges[:-1] >= 0.25) & (bin_edges[:-1] <= 0.75)].sum()

    # Calculate the reception probability
    reception_probability = area_025_to_075 / total_area if total_area > 0 else 0

    # Calculate the total number of occurrences to convert y-axis to percentage
    total_counts = counts.sum()

    # Prepare to draw using plt.bar, converting counts to percentage
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], (counts / total_counts) * 100, width=np.diff(bin_edges), align="edge", color='tab:blue', alpha=1.0)
    plt.xlabel('Update Interval [s]')
    plt.ylabel('Frequency [%]')
    plt.title('Update Interval Distribution')
    plt.xticks(np.arange(0, 5.1, 0.5))
    plt.yticks(np.arange(0, 41, 5.0))
    plt.xlim([-0.025, 5.025])

    plt.savefig(f'../figures/update_interval_all_{sensor_id}.png')

    # Create a crosstab to count how many rows fall into each pair of distance_bin and traffic_bin
    bin_counts = pd.crosstab(filtered_df['distance_bin'], filtered_df['traffic_bin'])

    # Function to calculate reception probability for a given DataFrame
    def calculate_reception_probability(df):
        counts, bin_edges = np.histogram(df['updateinterval'], bins=50)
        
        total_area = counts[(bin_edges[:-1] >= 0) & (bin_edges[:-1] <= maximum_time_update)].sum()
        area_025_to_075 = counts[(bin_edges[:-1] >= 0.25) & (bin_edges[:-1] <= 0.75)].sum()
        
        reception_probability = area_025_to_075 / total_area if total_area > 0 else 0
        return reception_probability

    reception_probabilities_df = (
        filtered_df.groupby(['distance_bin', 'traffic_bin'])
        .apply(calculate_reception_probability)
        .reset_index(name='reception_probability')
    )

    reception_probabilities_df.to_csv('reception_prob.csv', index = False)

    print(reception_probabilities_df.columns)

    receptionprob_filename = f'../sensors_reception_prob/reception_prob/receptionprob_{sensor_id}.csv'

    # Convert the crosstab into a long format DataFrame for merging
    bin_counts_long = bin_counts.stack().reset_index()
    bin_counts_long.columns = ['distance_bin', 'traffic_bin', 'data_count']

    # Create an empty list to store the sampled DataFrames
    sampled_results = []
    sampled_2500 = []

    # Iterate over each unique combination of distance_bin and traffic_bin in the combined DataFrame
    for (distance_bin, traffic_bin), group in filtered_df.groupby(['distance_bin', 'traffic_bin']):
        # Check if the number of data points in this group is above the threshold
        if len(group) > minimum_nb_data:
            # If above threshold, take a random sample of 'sample_size' rows
            sampled_group = group.sample(n=minimum_nb_data, random_state=42)
            sampled_2500_ = group.sample(n=2500, random_state=42)
        else:
            # Otherwise, use the entire group
            sampled_group = group
            sampled_2500_ = group
        
        # Append the sampled or full group to the list
        sampled_results.append(sampled_group)
        sampled_2500.append(sampled_2500_)

    # Concatenate the sampled results into a single DataFrame
    sampled_filtered_df = pd.concat(sampled_results)
    sampled_2500 = pd.concat(sampled_2500)

    # Calculate reception probabilities for the sampled DataFrame
    sampled_reception_probabilities_df = (
        sampled_filtered_df.groupby(['distance_bin', 'traffic_bin'])
        .apply(calculate_reception_probability)
        .reset_index(name='5k_reception_probability')
    )

    sampled_reception_probabilities_2500_df = (
        sampled_2500.groupby(['distance_bin', 'traffic_bin'])
        .apply(calculate_reception_probability)
        .reset_index(name='2500_reception_probability')
    )

    # Merge the reception probabilities with the bin counts
    reception_probabilities_df = pd.merge(
        reception_probabilities_df,
        bin_counts_long,
        on=['distance_bin', 'traffic_bin'],
        how='left'
    )

    # Merge the reception probabilities with the bin counts
    reception_probabilities_df = pd.merge(
        reception_probabilities_df,
        sampled_reception_probabilities_df,
        on=['distance_bin', 'traffic_bin'],
        how='left'
    )

    reception_probabilities_df = pd.merge(
        reception_probabilities_df,
        sampled_reception_probabilities_2500_df,
        on=['distance_bin', 'traffic_bin'],
        how='left'
    )

    reception_probabilities_df['max_dist_NM'] = maximum_distance

    # Save the DataFrame to a CSV file
    reception_probabilities_df.to_csv(receptionprob_filename, index=False)