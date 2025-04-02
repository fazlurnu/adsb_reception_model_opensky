# %%

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import re
# %%
df_chung_rmse = pd.read_csv('../model/rmse_chung.csv')
df_model_rmse = pd.read_csv('../model/regression_models.csv')

model_new = df_model_rmse['RMSE (Test) %']
chung_rmse = df_chung_rmse['rmse']

print(df_model_rmse['RMSE (Test) %'].mean(), df_chung_rmse['rmse'].mean())
print(len(df_model_rmse))

# Create a boxplot for the given data
plt.figure(figsize=(8, 8))
plt.boxplot([df_model_rmse['RMSE (Test) %'], df_chung_rmse['rmse']], labels=['Adapted Model', 'Existing Model'])

# Set title and labels
plt.ylabel('RMSE [%]', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

output_dir = "../figures"
os.makedirs(output_dir, exist_ok=True)

# Show the plot
plt.savefig(os.path.join(output_dir, 'rmse_comparison.png'), dpi=300, bbox_inches='tight')
# %% Check the error vs distance in the adapted model

error_df = pd.read_csv('../model/error_per_distance.csv')

bins = np.arange(0, 260, 20)  # 0 to 300 with step 20
labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

# Assign each distance to a bin
error_df['distance_bin'] = pd.cut(error_df['distance'], bins=bins, labels=labels, right=False)

# Group errors by distance bin
grouped_errors = [error_df[error_df['distance_bin'] == label]['error'].dropna().values*100 for label in labels]
data_counts = [len(errors) for errors in grouped_errors]  # Count number of points in each bin

for e in grouped_errors:
    print(e.mean(), np.median(e), e.std())

fig, ax1 = plt.subplots(figsize=(12, 6))

# Boxplot on primary y-axis
ax1.boxplot(grouped_errors, labels=labels, showfliers=False)
ax1.set_xlabel("Distance Interval [NM]", fontsize=14)
ax1.set_ylabel("Reception Probability Error [%]", fontsize=14, color='black')
ax1.set_title("Reception Probability Error Distribution by Distance Range with Data Count", fontsize=16)
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticklabels(labels, rotation=45, fontsize=12)

# Create secondary y-axis for data count
ax2 = ax1.twinx()
ax2.bar(range(1, len(labels) + 1), data_counts, alpha=0.4, color='gray', width=0.6)
ax2.set_ylabel("Number of Data Points", fontsize=14, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

output_dir = "../figures"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, 'error_each_distance.png'), dpi=300, bbox_inches='tight')
# %%
# Create a bar plot for RMSE (Test) %
regression_df = pd.read_csv('../model/regression_models.csv')

# Sort the dataframe so that 'Radarcape' sensors are on the left
regression_df_sorted = regression_df.sort_values(by=["Sensor Type", "RMSE (Test) %"], ascending=[True, False])

# Assign colors: Radarcape (lighter gray), Dump1090 (darker gray)
colors = ['lightgray' if sensor_type == 'Radarcape' else 'gray' for sensor_type in regression_df_sorted["Sensor Type"]]

# Create figure and bar plot
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(regression_df_sorted["Sensor ID"].astype(str), regression_df_sorted["RMSE (Test) %"], color=colors)

# Customize plot
ax.set_xlabel("Sensor ID", fontsize=14)
ax.set_ylabel("RMSE [%]", fontsize=14)
ax.set_title("Cross Validation RMSE [%] for Each Sensor", fontsize=16)
ax.set_xticklabels(regression_df_sorted["Sensor ID"].astype(str), rotation=90, fontsize=12)
ax.tick_params(axis='y', labelsize=12)

# Create legend
legend_labels = {'lightgray': 'Radarcape', 'gray': 'Dump1090'}
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_labels.keys()]
ax.legend(handles, legend_labels.values(), title="Sensor Type", fontsize=12, title_fontsize=14)

# Show plot
plt.savefig(os.path.join(output_dir, 'cross_validation_rmse.png'), dpi=300, bbox_inches='tight')

# %%

sensor_data_path = '../sensors_reception_prob/processed_sensor_probability.csv'
sensor_df = pd.read_csv(sensor_data_path)

# Extract unique CR and airport values for each sensor
unique_sensor_data = sensor_df.groupby("sensor_id")[["CR", "airport", "type"]].first().reset_index()

# Assign colors based on sensor type
color_map = {'Radarcape': 'lightgray', 'dump1090': 'gray'}
colors = [color_map[sensor_type] for sensor_type in unique_sensor_data["type"]]

# Create scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(unique_sensor_data["CR"], unique_sensor_data["airport"], c=colors, edgecolors="black", s=100, marker="o")

# Identify the points corresponding to the specified sensor IDs
highlight_sensors = [-1408235680, -1408235424]
highlight_points = unique_sensor_data[unique_sensor_data["sensor_id"].isin(highlight_sensors)]

# Highlight the specified sensors
plt.scatter(highlight_points["CR"], highlight_points["airport"], c="red", edgecolors="black", s=150, marker="o")

# Annotate points
for _, row in highlight_points.iterrows():
    plt.annotate(str(row["sensor_id"]), (row["CR"], row["airport"]), textcoords="offset points", xytext=(5,5), fontsize=12, color="red")

# Customize plot
plt.xlabel("Maximum Distance [NM]", fontsize=14)
plt.ylabel("Number of Airport [-]", fontsize=14)
plt.title("Sensor's Geographical Features", fontsize=16)

# Add legend without the highlighted sensors
handles = [plt.Line2D([0], [0], marker='o', color=color, markersize=10, linestyle='', markeredgecolor="black") for color in color_map.values()]
plt.legend(handles, color_map.keys(), title="Sensor Type", fontsize=12, title_fontsize=14)

# Show the plot
plt.savefig(os.path.join(output_dir, 'sensor_geo_features.png'), dpi=300, bbox_inches='tight')

# %%
from scipy.spatial.distance import cdist

# Extract only the relevant columns (CR, number of airports, type, and sensor_id)
sensor_features = sensor_df[["sensor_id", "CR", "airport", "type"]].drop_duplicates()

# Separate Radarcape and Dump1090 sensors
radarcape_sensors = sensor_features[sensor_features["type"] == "Radarcape"]
dump1090_sensors = sensor_features[sensor_features["type"] == "dump1090"]

# Compute pairwise distances based on CR and airport values
radarcape_coords = radarcape_sensors[["CR", "airport"]].values
dump1090_coords = dump1090_sensors[["CR", "airport"]].values

distance_matrix = cdist(radarcape_coords, dump1090_coords, metric='euclidean')

# Find indices of the three closest pairs
sorted_indices = np.unravel_index(np.argsort(distance_matrix, axis=None)[:2], distance_matrix.shape)

# Create subplots
fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharey=True)
color_list = ['tab:blue', 'tab:orange', 'tab:green']

idx = 0

radarcape_idx = sorted_indices[0][idx]
dump1090_idx = sorted_indices[1][idx]

closest_radarcape = radarcape_sensors.iloc[radarcape_idx]
closest_dump1090 = dump1090_sensors.iloc[dump1090_idx]

# Map sensor_id to type
sensor_type_map = {
    closest_radarcape["sensor_id"]: "Radarcape",
    closest_dump1090["sensor_id"]: "Dump1090"
}

# Extract reception probability vs. distance for the closest pair
closest_pair_ids = [closest_radarcape["sensor_id"], closest_dump1090["sensor_id"]]
reception_data = sensor_df[sensor_df["sensor_id"].isin(closest_pair_ids)][["sensor_id", "distance_bin", "traffic_bin", "reception_probability"]]

print(closest_pair_ids)

# Count distinct values of A for each (B, C) pair
mask = reception_data.groupby(['distance_bin', 'traffic_bin'])['sensor_id'].transform('nunique') > 1

# Filter the dataframe
df_filtered = reception_data[mask]

color_index = 0

for i in range(0, 110, 50):
    for sensor_id in closest_pair_ids:

        mask_traffic = df_filtered['traffic_bin'] == f'({i}, {i+10}]'
        mask_sensor_id = df_filtered['sensor_id'] == sensor_id

        marker_style = '*' if sensor_type_map[sensor_id] == "Dump1090" else 'o'

        # Create labels dynamically
        label = f'Traffic Midpoint: {i+5} - {sensor_type_map[sensor_id]}'

        ax.plot(df_filtered[mask_traffic & mask_sensor_id]['distance_bin'],
                df_filtered[mask_traffic & mask_sensor_id]['reception_probability'],
                color=color_list[color_index],
                marker=marker_style,
                markersize=8,
                label=label)

    color_index += 1

ax.set_xticklabels(df_filtered['distance_bin'].unique(), rotation=90, fontsize=12)
# ax.set_title(f'Closest Pair {idx+1}')
ax.set_xlabel('Distance Interval [NM]', fontsize = 12)
ax.set_ylabel('Reception Probability [-]', fontsize = 12)

# Add legend
ax.legend(fontsize=10)

# Show the plot
plt.tight_layout()

plt.savefig(os.path.join(output_dir, 'effect_of_distance.png'), dpi=300, bbox_inches='tight')

plt.show()