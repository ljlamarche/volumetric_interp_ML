"""
Volumetric interpolation using artificial neural networks.

Authors: Nicolas Gachancipa, Leslie Lamarche

Stanford Research Institute (SRI)
Embry-Riddle Aeronautical University
"""

# Imports.
import datetime as dt
import numpy as np
import pandas as pd
from AC_read_datafiles import read_AC_datafile
from support_functions_mod import read_datafile
from support_functions_mod import volumetric_nn


# Inputs: Directory to data file, start and end times.
data_file = [r"20220426.001_lp_5min-fitcal.h5"]
AC_data_file = [r"20220426.001_ac_5min-fitcal.h5"]
AC_Alt_lim = 200_000
start = "2022-04-26T13:32:00"
end = "2022-04-26T13:35:00"

# Convert start times to datetime format.
start = dt.datetime.strptime(start, '%Y-%m-%dT%H:%M:%S')
end = dt.datetime.strptime(end, '%Y-%m-%dT%H:%M:%S')

z = 0
if len(AC_data_file) == z:
    # Open file and extract the following arrays: time, latitude, longitude, altitude, value, error.
    data = read_datafile(data_file, start, end)

    # Train the neural network.
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    # Running the neural network and saving the predicted densities to y_pred (y_pred will only be used when evaluating
    # the performance of the network with a synthetic model).
    volumetric_nn(data, start, end, resolution=(100, 100, 30), cbar_lim=(1e10, 3e11), real_dist=False, fig3D=True)

else:
    # Open file and extract the following arrays: time, latitude, longitude, altitude, value, error.
    data = read_datafile(data_file, start, end)
    data_2 = read_AC_datafile(AC_data_file, start, end, Alt_lim=AC_Alt_lim)

    combined_data = pd.concat([data, data_2])
    # Train the neural network.
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    # Running the neural network and saving the predicted densities to y_pred (y_pred will only be used when evaluating
    # the performance of the network with a synthetic model).
    volumetric_nn(combined_data, start, end, resolution=(100, 100, 30), cbar_lim=(1e10, 3e11), real_dist=False, fig3D=True)


# Train the neural network.
#pd.set_option("display.max_rows", None, "display.max_columns", None)


# Running the neural network and saving the predicted densities to y_pred (y_pred will only be used when evaluating
# the performance of the network with a synthetic model).
#volumetric_nn(data, resolution=(100, 100, 30), cbar_lim=(1e10, 3e11), real_dist=False, fig3D=True)
