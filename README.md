# ADS-B Reception Probability Model

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

## Citation

If you use this repository, please cite the following paper (coming soon)

## License

This project is released under the MIT License.
