"""
Volumetric interpolation using artificial neural networks.

Authors: Nicolas Gachancipa, Leslie Lamarche

Stanford Research Institute (SRI)
Embry-Riddle Aeronautical University
"""

# Imports.
import datetime as dt
import pandas as pd
from support_functions_mod import read_datafile
from support_functions_mod import volumetric_nn


# Inputs: Directory to data file, start and end times.
data_file = ["20200518.001_lp_1min-fitcal.h5"]
start = "2020-05-18T19:15:00"
end = "2020-05-18T19:17:00"

# Convert start times to datetime format.
start = dt.datetime.strptime(start, '%Y-%m-%dT%H:%M:%S')
end = dt.datetime.strptime(end, '%Y-%m-%dT%H:%M:%S')

# Open file and extract the following arrays: time, latitude, longitude, altitude, value, error.
data = read_datafile(data_file, start, end)

# Train the neural network.
pd.set_option("display.max_rows", None, "display.max_columns", None)


# Running the neural network and saving the predicted densities to y_pred (y_pred will only be used when evaluating
# the performance of the network with a synthetic model).
volumetric_nn(data, resolution=(100, 100, 30), cbar_lim=(1e10, 3e11), real_dist=False, fig3D=True)


