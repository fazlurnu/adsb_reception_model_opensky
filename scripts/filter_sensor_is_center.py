import os
import shutil

import pandas as pd
import numpy as np

from scipy.spatial import ConvexHull

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

def calculate_circularity(area, perimeter):
    return (4 * np.pi * area) / (perimeter ** 2)

# Function to project a point onto a line segment and calculate the Haversine distance
def haversine_projected_distance(sensor, p1, p2):
    """Calculate the Haversine distance from the sensor to the closest point on the line segment p1-p2."""
    
    def haversine_distance(p1, p2):
        """Haversine distance between two (lat, lon) points."""
        lat1, lon1 = p1
        lat2, lon2 = p2
        return haversine_NM(lat1, lon1, lat2, lon2)
    
    # Convert points to numpy arrays for easier calculation
    sensor = np.array(sensor)
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # Calculate the Euclidean projection of the sensor onto the line segment
    v = p2 - p1
    w = sensor - p1
    
    c1 = np.dot(w, v)
    c2 = np.dot(v, v)
    
    if c1 <= 0:
        # Closest point is p1
        closest_point = p1
    elif c2 <= c1:
        # Closest point is p2
        closest_point = p2
    else:
        # Closest point is the projection on the line segment
        b = c1 / c2
        closest_point = p1 + b * v
    
    # Compute the Haversine distance between the sensor and the closest point on the segment
    closest_point_latlon = (closest_point[0], closest_point[1])
    return haversine_distance((sensor[0], sensor[1]), closest_point_latlon)

# Define the path to the scripts directory and the sensor location data
scripts_path = '.'
sensor_loc_path = 'sensors/sensor_loc.csv'

df_sensor_loc = pd.read_csv('../sensors/sensor_loc.csv')

nb_file = len(os.listdir(scripts_path)) - 6 # five files are not .csv
counter_file = 0

# Iterate over the files in the 'scripts' folder
for filename in os.listdir(scripts_path):
    if filename.endswith('.csv'):
        counter_file += 1
        df = pd.read_csv(f"{scripts_path}/{filename}")

        print(filename)
        parts = filename.split('/')[-1]  # Get the last part 'sensor123_2024-10-04_data.csv'
        sensor_id = int(parts.split('_')[0])  # Split by '_'

        sensor_lat = df_sensor_loc[df_sensor_loc['serial'] == sensor_id]['lat'].iloc[0]
        sensor_lon = df_sensor_loc[df_sensor_loc['serial'] == sensor_id]['lon'].iloc[0]

        df['distance_NM'] = df.apply(lambda row: haversine_NM(row['lat'], row['lon'], sensor_lat, sensor_lon), axis=1)
        df = df[df['distance_NM'] < 300] ## filter out outliers, theoretically ADS-B receiver max dist is 250

        # Extract the lat and lon columns as a 2D array of points
        points = np.column_stack((df['lon'], df['lat']))
        
        hull = ConvexHull(points)

        hull_points = points[hull.vertices]

        # Calculate distances of each point in the convex hull to the sensor's location
        distances = [haversine_NM(sensor_lat, sensor_lon, lat, lon) for lon, lat in hull_points]

        # Convert results into a DataFrame for easier visualization
        hull_distances_df = pd.DataFrame(hull_points, columns=["Longitude", "Latitude"])
        hull_distances_df["distance_NM"] = distances
        hull_distances_df = hull_distances_df[hull_distances_df["distance_NM"] < 300]

        # Check circularity
        hull_area = hull.volume  # Convex hull area
        hull_perimeter = hull.area

        # Calculate the circularity
        circularity = calculate_circularity(hull_area, hull_perimeter)
        
        max_dist_hull = hull_distances_df['distance_NM'].max()
        min_dist_hull = hull_distances_df['distance_NM'].min()

        max_min_ratio = max_dist_hull/min_dist_hull

        # Calculate the perpendicular distances from sensor to each edge of the convex hull
        sensor_coords = (sensor_lat, sensor_lon)
        edge_distances = []
        for i in range(len(hull_points)):
            p1 = (hull_points[i][1], hull_points[i][0])  # (lat, lon) for point 1
            p2 = (hull_points[(i + 1) % len(hull_points)][1], hull_points[(i + 1) % len(hull_points)][0])  # Wrap around for the last point
            edge_distances.append(haversine_projected_distance(sensor_coords, p1, p2))

        # Find the minimum distance to any edge of the convex hull
        min_dist_edge = min(edge_distances)
        max_dist_edge = max(edge_distances)

        max_min_ratio_edge = max_dist_edge/min_dist_edge

        is_circular = circularity > 0.8
        is_center = max_min_ratio < 1.25
        is_center_edge = max_min_ratio_edge < 1.25

        print(f"{circularity:.2f}, {max_min_ratio:.2f}, {max_min_ratio_edge:.2f}")
        print(is_circular, is_center, is_center_edge)

        if(is_circular and is_center and is_center_edge):
            target_folder = '../sensor_circularity_quick_test/sensor_is_center_and_circular'

            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

        else:
            target_folder = '../sensor_circularity_quick_test/sensor_not_center_and_circular_again'
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

        source_path = os.path.join(scripts_path, filename)
        destination_path = os.path.join(target_folder, filename)

        shutil.move(source_path, destination_path)
        print(f"{counter_file}/{nb_file}. Moved {filename} to {target_folder}")