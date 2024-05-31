"""
Volumetric interpolation using artificial neural networks.

Authors: Nicolas Gachancipa, Leslie Lamarche

Stanford Research Institute (SRI)
Embry-Riddle Aeronautical University
"""

# Imports.
import datetime as dt
import pandas as pd
from support_functions import volumetric_nn, read_datafile

# List of potential data to test.
data = [r"20200518.001_lp_1min-fitcal.h5", r"20240511.003_lp_1min-fitcal.h5"]

# Inputs: Directory to data file, start and end times.
data_file = data[1]
start = "2016-09-13T00:00:01"
end = "2016-09-13T00:00:10"

# Convert start times to datetime format.
start = dt.datetime.strptime(start, '%Y-%m-%dT%H:%M:%S')
end = dt.datetime.strptime(end, '%Y-%m-%dT%H:%M:%S')

# Open file and extract the following arrays: time, latitude, longitude, altitude, value, error.
time, lat, lon, alt, val, err, _ = read_datafile(data_file, start, end)

# Run NN.
max_n = 1512 # Maximum number of points wou want to use (useful for testing).
data = pd.DataFrame([lat, lon, alt, val[0], err[0]]).T.iloc[:max_n, :]
data.columns = ['Latitude', 'Longitude', 'Altitude', 'Value', 'Error']

# Uncomment to remove values with large errors (Errors larger than the values themselves).
# mask = data['Value'] >= data['Error']
# data = data[mask]

# Train the neural network.
pd.set_option("display.max_rows", None, "display.max_columns", None)

volumetric_nn(data, resolution=(100, 100, 30), cbar_lim=(1e10, 3e11), real_dist=False, fig3D=True)

# Things to try.
# 1. Include errors by weighing each data point (density/(error^2)?), or filter them out.
# 2. Overfitting penalization with neural networks. See: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit

