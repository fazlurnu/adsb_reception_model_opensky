import pandas as pd
import numpy as np

def reception_model(X,
                    a1_t, a1_a, a1_intercept, 
                    a2_t, a2_a, a2_intercept, 
                    a3_t, a3_a, a3_intercept, 
                    a4_t, a4_a, a4_intercept):
    d, max_dist, traffic_avg, airport = X.T  # Unpack the columns
    a1 = a1_t * traffic_avg + a1_a * airport + a1_intercept
    a2 = a2_t * traffic_avg + a2_a * airport + a2_intercept
    a3 = a3_t * traffic_avg + a3_a * airport + a3_intercept
    a4 = a4_t * traffic_avg + a4_a * airport + a4_intercept
    ratio = d / max_dist

    return np.exp(-3 * ratio**2) * (1 + a1 * ratio + a2 * ratio**2 + a3 * ratio**3 + a4 * ratio**4)

def main():
    ## Load data
    df = pd.read_csv('regression_models.csv')

    ## Find the sensor with the lowest RMSE
    lowest_rmse_sensor = df.loc[df["RMSE (Test) %"].idxmin(), "Sensor ID"]

    ## Extract and parse best parameters
    best_params_str = df[df['Sensor ID'] == lowest_rmse_sensor]['Best Parameters'].values[0]
    best_params_array = np.fromstring(best_params_str.strip("[]"), sep=' ')

    ## Define input features
    distance = 10 # in Nautical Mile (NM)
    max_dist = 290 # in Nautical Mile (NM)
    traffic = 50 # surrounding the ADS-B receiver, can be considered as within the max_dist
    nb_airport = 50 # surrounding the ADS-B receiver, can be considered as within the max_dist
    X = np.array([[distance, max_dist, traffic, nb_airport]])  # d, max_dist, traffic_avg, airport

    ## Calculate reception probability
    reception_prob = reception_model(X, *best_params_array)

    print(f"Reception probability for sensor {lowest_rmse_sensor}: {reception_prob}")

if __name__ == "__main__":
    main()