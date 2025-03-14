# %%

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# %%
df_chung_rmse = pd.read_csv('../model/rmse_chung.csv')
df_model_rmse = pd.read_csv('../model/regression_models.csv')

model_new = df_model_rmse['RMSE (Test) %']
chung_rmse = df_chung_rmse['rmse']

# Create a boxplot for the given data
plt.figure(figsize=(8, 6))
plt.boxplot([df_model_rmse['RMSE (Test) %'], df_chung_rmse['rmse']], labels=['Adapted Model', 'Existing Model'])

# Set title and labels
plt.title('Comparison of RMSE Values', fontsize = 14)
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

fig, ax1 = plt.subplots(figsize=(12, 6))

# Boxplot on primary y-axis
ax1.boxplot(grouped_errors, labels=labels, showfliers=False)
ax1.set_xlabel("Distance Range [NM]", fontsize=14)
ax1.set_ylabel("Reception Probability Absolute Error [%]", fontsize=14, color='black')
ax1.set_title("Reception Probability Absolute Error Distribution by Distance Range with Data Count", fontsize=16)
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticklabels(labels, rotation=45, fontsize=12)

# Create secondary y-axis for data count
ax2 = ax1.twinx()
ax2.bar(range(1, len(labels) + 1), data_counts, alpha=0.4, color='gray', width=0.6)
ax2.set_ylabel("Number of Data Points", fontsize=14, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

output_dir = "../figures"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, 'absolute_error_each_distance.png'), dpi=300, bbox_inches='tight')
# %%
