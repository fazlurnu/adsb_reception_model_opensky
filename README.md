# ADS-B Reception Probability Model

Why do we need ADS-B Reception Probability Model? Because this model can be a useful to simulate the safety of air traffic when aircraft will fly and avoid conflicts solely relying on ADS-B. Free flight concept!

This repository uses OpenSky's data to model the reception probability of an ADS-B receiver, under different traffic, range, and airport conditions. This study introduces a new approach for estimating the ADS-B message reception probability by examining variations in update intervals.

This repository includes processed dataset, scripts, and results from the model evaluation. Findings indicate that the model has an improved accuracy compared to previous models. However, some non-linearity is still not captured in the model.

Feel free to contribute!

For further details, refer to the full paper (soon published) available on Journal of Open Aviation Science (JOAS).

## Repository Content
- scripts: contain scripts to download data from OpenSky's database
- sensors: .csv file containing list of sensors
- sensors_reception_prob: contain processed position data transformed into reception probability
- airports: contain airports location around the world
- model: contain the constants for the model

## How-to
To generate the results, first you need to download lots of OpenSky data. The first thing to do is to run `get_pos_data_quick_filter.py`, this code downloads data and only save those that are considered "circular" within the time frame of the download. Next, you can run `get_pos_data_final_filtered.py`, this will download the data for an entire year of 2022. You can set the begin and end time to have less or more data. After that, you can run `make_reception_prob.py` to calculate the reception probability of each receiver. Lastly, you can run `regression.py` to produce the model.

The regression model that I have is not the most ideal, please consider improving it.

## Citation

If you use this repository, please cite the following paper (coming soon)

## License

This project is released under the MIT License.
