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

# List of potential data to test.
data_lp_1min = [r"20200518.001_lp_1min-fitcal.h5", r"20240511.003_lp_1min-fitcal.h5", r"20240324.001_lp_1min-fitcal.h5", r"20190218.002_lp_1min-fitcal.h5", r"20170304.003_lp_1min-fitcal.h5"]
data_lp_3min = [r"20230605.002_lp_3min-fitcal.h5", r"20220607.001_lp_3min-fitcal.h5"]
data_lp_5min = [r"20220427.002_lp_5min-fitcal.h5"]
data_lp_20min = [r"20220715.002_lp_20min-fitcal.h5"]
two_files = [r'20200613.001_lp_1min-fitcal.h5', r'20200613.001_lp_3min-fitcal.h5']
help = [r"20171119.001_lp_1min-fitcal.h5"]
c_datas = [[r"gradient_circular_data_ac.h5", r"gradient_circular_data_02.h5"], [r"chapman_circular_data_ac.h5", r"chapman_circular_data_015.h5", r"chapman_circular_data_02.h5", r"chapman_circular_data_025.h5", r"chapman_circular_data_03.h5", r"chapman_circular_data_035.h5"], [r"chapman_gradient_data_ac.h5", r"chapman_gradient_data_015.h5", r"chapman_gradient_data_02.h5", r"chapman_gradient_data_025.h5", r"chapman_gradient_data_03.h5", r"chapman_gradient_data_035.h5"], [r"wave2_data_ac.h5", r"wave2_data_015.h5", r"wave2_data_02.h5", r"wave2_data_025.h5", r"wave2_data_03.h5", r"wave2_data_035.h5"], [r"two_circles_data_ac.h5", r"two_circles_data_015.h5", r"two_circles_data_02.h5", r"two_circles_data_025.h5", r"two_circles_data_03.h5", r"two_circles_data_035.h5"]]


# Inputs: Directory to data file, start and end times.
# data_file = [data_lp_3min[1]]
data_file = c_datas[0]
start = "2017-11-21T19:15:00"
end = "2017-11-21T19:17:00"

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


