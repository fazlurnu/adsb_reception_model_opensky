import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import re
import os

# Function to extract the midpoint of the bin range
def extract_midpoint(bin_label):
    match = re.findall(r"\d+", bin_label)  # Extract numbers from the bin label
    if len(match) == 2:
        return (int(match[0]) + int(match[1])) / 2  # Compute the midpoint
    return float('inf')  # Default for unexpected formats

for file in os.listdir('../sensors_reception_prob/nb_data_sensors/'):
    if('.csv' in file):
        sensor_name = file[:-4]
        sensor_name = sensor_name[8:]
        df = pd.read_csv(f'../sensors_reception_prob/nb_data_sensors/{file}')

        # Convert the dataframe to a format suitable for a heatmap
        df_melted = df.melt(id_vars=["distance_bin"], var_name="traffic_bin", value_name="count")

        # Apply midpoint extraction for sorting
        df_melted["distance_mid"] = df_melted["distance_bin"].apply(extract_midpoint)
        df_melted["traffic_mid"] = df_melted["traffic_bin"].apply(extract_midpoint)

        # Sort by extracted midpoints
        df_melted = df_melted.sort_values(by=["distance_mid", "traffic_mid"])

        # Recreate the pivot table with sorted values
        df_pivot_sorted = df_melted.pivot(index="distance_bin", columns="traffic_bin", values="count")
        df_pivot_sorted = df_pivot_sorted.reindex(index=df_melted["distance_bin"].unique(), columns=df_melted["traffic_bin"].unique())
        
        # Plot the corrected sorted heatmap
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(df_pivot_sorted, cmap="Blues", cbar=True, vmax = 10000, cbar_kws={'label' : 'Number of Data'})
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(16)  # Change color bar label font size

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)
        plt.xlabel("Traffic Bin [-]", fontsize = 16)
        plt.ylabel("Distance Bin [NM]", fontsize = 16)
        plt.title(f"Number of Data for Traffic vs Distance Pair. Sensor ID: {sensor_name}", fontsize = 16)

        plt.savefig(f'../figures/nb_data_{sensor_name}.png', dpi = 300, bbox_inches = "tight")
        plt.close()