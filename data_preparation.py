#
# Data preparation.
#
# Author: Nicolas Gachancipa
#

# Imports.
import datetime as dt
import pymap3d as pm
from support_functions import create_plots, read_datafile, train_nn

# Inputs.
# data_file = r"..\..\input_data\2016\11\27\20161127.002_lp_1min-fitcal.h5"
data_file = r"..\..\input_data\risrn_synthetic_imaging_chapman.h5"
start_time = "2016-11-27T22:45:00"
end_time = "2016-11-27T22:48:00"
altitudes = [300]
color_lim = [1e9, 1e11]

# Code.

# Convert start times to datetime format.
start_time = dt.datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
end_time = dt.datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S')

# Open file and extract the following arrays: time, latitude, longitude, altitude, value, error.
time, lat, lon, alt, val, err, hull_vert = read_datafile(data_file, start_time, end_time)

# Set input coordinates.
hull_lat, hull_lon, hull_alt = pm.ecef2geodetic(hull_vert[:, 0], hull_vert[:, 1], hull_vert[:, 2])

# Compute parameters through a NN.
train_nn(lat, lon, alt, val)

# Create plot.
create_plots(data_file, altitudes, start_time, end_time, hull_lat, hull_lon, color_lim=color_lim)