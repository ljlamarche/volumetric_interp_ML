"""
Volumetric interpolation using artificial neural networks.

Authors: Nicolas Gachancipa, Leslie Lamarche

Stanford Research Institute (SRI)
Embry-Riddle Aeronautical University
"""

# Imports.
import datetime as dt
import pandas as pd
from support_functions import read_datafile, volumetric_nn
from support_functions_2 import read_datafile_2, volumetric_nn_2

# Inputs: Directory to data file, start and end times.
data_file = r"20220426.001_lp_5min-fitcal.h5"
data_file_AC = r"20220426.001_ac_5min-fitcal.h5"
# Data files should be from the same experiment and will have the same timeframe
start = "2022-04-26T13:30:01"
end = "2022-05-26T13:30:16"

# Convert start times to datetime format.
start = dt.datetime.strptime(start, '%Y-%m-%dT%H:%M:%S')
end = dt.datetime.strptime(end, '%Y-%m-%dT%H:%M:%S')

# Open file and extract the following arrays: time, latitude, longitude, altitude, value, error.
time, lat, lon, alt, val, err, _ = read_datafile(data_file, start, end)
time_2, lat_2, lon_2, alt_2, val_2, err_2, _ = read_datafile_2(data_file_AC, start, end)

# Run NN.
max_n = 1512 # Maximum number of points wou want to use (useful for testing).
data = pd.DataFrame([lat, lon, alt, val[0], err[0]]).T.iloc[:max_n, :]
data_2 = pd.DataFrame([lat_2, lon_2, alt_2, val_2[0], err_2[0]]).T.iloc[:max_n, :]

data.columns = ['Latitude', 'Longitude', 'Altitude', 'Value', 'Error']
data_2.columns = ['Latitude', 'Longitude', 'Altitude', 'Value', 'Error']

# Uncomment to remove values with large errors (Errors larger than the values themselves).
# mask = data['Value'] >= data['Error']
# data = data[mask]

# Train the neural network.
pd.set_option("display.max_rows", None, "display.max_columns", None)
volumetric_nn(data, start, end,resolution=(100, 100, 30), cbar_lim=(1e10, 3e11), real_dist=False, fig3D=True)
volumetric_nn_2(data_2, start, end, resolution=(100, 100, 30), cbar_lim=(1e10, 3e11), real_dist=False, fig3D_2=True)

# Things to try.
# 1. Include errors by weighing each data point (density/(error^2)?), or filter them out.
# 2. Overfitting penalization with neural networks. See: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
