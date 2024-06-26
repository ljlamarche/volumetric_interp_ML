"""
Volumetric interpolation using artificial neural networks.

Authors: Nicolas Gachancipa, Leslie Lamarche

Stanford Research Institute (SRI)
Embry-Riddle Aeronautical University
"""

# Imports.
import datetime as dt
import pandas as pd
from new_read_datafiles import read_datafiles
from support_functions import volumetric_nn

# Inputs: Directory to data file, start and end times.
data_file = [r"20210418.001_lp_5min-fitcal.h5", r"20210418.001_ac_5min-fitcal.h5"]
start = "2021-04-18T09:00:01"
end = "2021-04-18T09:01:10"

# Convert start times to datetime format.
start = dt.datetime.strptime(start, '%Y-%m-%dT%H:%M:%S')
end = dt.datetime.strptime(end, '%Y-%m-%dT%H:%M:%S')

# Open file and extract the following arrays: time, latitude, longitude, altitude, value, error.
result_df = read_datafiles(data_file, start, end, chi2lim=(0.1, 10))

# Run NN... incorporated within the 'read_datafiles' function
##max_n = 1512 # Maximum number of points wou want to use (useful for testing).
##data = pd.DataFrame([lat, lon, alt, val[0], err[0]]).T.iloc[:max_n, :]
##data.columns = ['Latitude', 'Longitude', 'Altitude', 'Value', 'Error']

# Uncomment to remove values with large errors (Errors larger than the values themselves).
# mask = data['Value'] >= data['Error']
# data = data[mask]

# Train the neural network.
pd.set_option("display.max_rows", None, "display.max_columns", None)
volumetric_nn(result_df, start, end, resolution=(100, 100, 30), cbar_lim=(1e10, 3e11), real_dist=False, fig3D=True)

# Things to try.
# 1. Include errors by weighing each data point (density/(error^2)?), or filter them out.
# 2. Overfitting penalization with neural networks. See: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
