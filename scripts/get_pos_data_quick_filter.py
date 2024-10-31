from pyopensky.trino import Trino
import pyopensky
import pandas as pd
from datetime import datetime, timedelta

import numpy as np
from scipy.spatial import ConvexHull

import os

# Function to calculate circularity
def calculate_circularity(area, perimeter):
    return (4 * np.pi * area) / (perimeter ** 2)

def is_in_europe(lat, lon):
    return 35.0 <= lat <= 71.0 and -25.0 <= lon <= 45.0

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

# Initialize Trino instance
trino = Trino()

# Load sensor location data
df_sensor_loc = pd.read_csv('../sensors/sensor_loc.csv')
df_sensor_loc['in_europe'] = df_sensor_loc.apply(lambda row: is_in_europe(row['lat'], row['lon']), axis=1)

# Get the list of sensor IDs (with the first ID being -1408237098)
sensor_ids = list(df_sensor_loc[df_sensor_loc['in_europe']]['serial'])
# sensor_ids = get_sensor_list()
# sensor_ids.insert(0, -1408237098)

# Define constants
NM_per_degree = 60
max_radius_NM = 300

# Set how many hours required to check the circularity
minute_interval = 15
hours_required = 2
df_per_hours = 60/minute_interval
df_required = df_per_hours * hours_required

# Define start and end dates for iteration
start_date = datetime(2022, 6, 1, 6, 0, 0)
end_date = datetime(2022, 6, 4, 6, 0, 0)  # June 1, 2022, is the last day

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
        while time_of_day < current_date + timedelta(hours=6):
            print(f"{receiver_counter}/{len(sensor_ids)}. Current time: {time_of_day.strftime('%H:%M:%S')}")

            # Set the start and stop times for each 1-minute interval
            start_time = time_of_day
            stop_time = start_time + timedelta(minutes=1)  # 1 minute later

            # Fetch raw data for the current time interval
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

                ## if serial is not there
                if(df.empty):
                    print("Dataframe is empty")
                    dataframe_empty = True
                    break
            else:
                print("Dataframe is empty")
                dataframe_empty = True
                break

            # Increment time_of_day by 15 minutes
            time_of_day += timedelta(minutes=15)

            # Check circularity after sufficient data points
            if (len(all_data) > df_required and not(circularity_checked)):
                circularity_checked = True
                for_convex_hull = pd.concat(all_data, ignore_index=True)
                
                # Extract the lat and lon columns as a 2D array of points
                points = np.column_stack((for_convex_hull['lon'], for_convex_hull['lat']))

                # Compute the convex hull
                hull = ConvexHull(points)

                hull_area = hull.volume  # Convex hull area
                hull_perimeter = np.sum(np.sqrt(np.sum(np.diff(points[hull.vertices, :], axis=0)**2, axis=1)))  # Convex hull perimeter

                # Calculate the circularity
                circularity = calculate_circularity(hull_area, hull_perimeter)

                # Calculate dist between furhest and closest node of CH to sensor
                hull_points = points[hull.vertices]
                distances = [haversine_NM(sensor_lat, sensor_lon, lat, lon) for lon, lat in hull_points]

                # Convert results into a DataFrame for easier dataproc
                hull_distances_df = pd.DataFrame(hull_points, columns=["Longitude", "Latitude"])
                hull_distances_df["distance_NM"] = distances

                max_dist_hull = max(distances)
                min_dist_hull = min(distances)

                ratio_max_min = max_dist_hull/min_dist_hull

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

                ratio_max_min_edge = max_dist_edge/min_dist_edge
                
                is_circular = 0.9<circularity<1.1
                is_center = ratio_max_min < 1.25
                is_center_edge = ratio_max_min_edge < 1.25
                
                print(f"Circularity: {circularity:.2f} for sensor_id: {sensor_id}")

                # If circularity is less than 0.9, move to the next sensor_id
                if (circularity < 0.9) or (circularity > 1.1):
                    print(f"Circularity is less than 0.9 or bigger than 1.1 for sensor_id {sensor_id}.\nSkipping to the next sensor.")
                    circularity_broken = True
                    circular_coverage_sensor[sensor_id] = circularity
                    break
                else:
                    circular_coverage_sensor[sensor_id] = circularity

        # Concatenate all data for the day into a single DataFrame
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)

            # Save the DataFrame for the day into a CSV file
            if (not final_df.empty) and (not circularity_broken):
                final_df.to_csv(f'{sensor_id}_{begin_time}_data.csv', index=False)

                # Print save status
                print(f"Data for {begin_time} saved to {sensor_id}_{begin_time}_data.csv")
            else:
                print(f"Data for {sensor_id} at {begin_time} is empty, or not circular, not saved")

            
        else:
            print(f"No data available for {begin_time}")

        # Increment current_date to the next day
        current_date += timedelta(days=1)

print("Data fetching complete.")
