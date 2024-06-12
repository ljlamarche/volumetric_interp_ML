"""
Volumetric interpolation using artificial neural networks.

Authors: Nicolas Gachancipa, Leslie Lamarche

Stanford Research Institute (SRI)
Embry-Riddle Aeronautical University
"""

# Imports.
import datetime as dt
import pandas as pd
import threading
import cv2
import matplotlib.pyplot as plt
from support_functions import read_datafile, volumetric_nn
from support_functions_2 import read_datafile_2, volumetric_nn_2

# Inputs: Directory to data file, start and end times.
data_file = r"20180504.001_lp_5min-fitcal.h5"
data_file_2 = r"20180504.001_ac_5min-fitcal.h5"
# Data files should be from the same experiment and will have the same timeframe
start = "2018-05-04T18:00:01"
end = "2018-05-04T18:00:16"

# Convert start times to datetime format.
start = dt.datetime.strptime(start, '%Y-%m-%dT%H:%M:%S')
end = dt.datetime.strptime(end, '%Y-%m-%dT%H:%M:%S')


# Open file and extract the following arrays: time, latitude, longitude, altitude, value, error.
# time, lat, lon, alt, val, err, _ = read_datafile(data_file, start, end)
# time_2, lat_2, lon_2, alt_2, val_2, err_2, _ = read_datafile_2(data_file_2, start, end)

# Define threading functions to read datafiles
def read_datafile_1_thread():
    global time, lat, lon, alt, val, err
    time, lat, lon, alt, val, err, _ = read_datafile(data_file, start, end)


def read_datafile_2_thread():
    global time_2, lat_2, lon_2, alt_2, val_2, err_2
    time_2, lat_2, lon_2, alt_2, val_2, err_2, _ = read_datafile_2(data_file_2, start, end)


# Create threads in attempt to help code run faster
thread1 = threading.Thread(target=read_datafile_1_thread)
thread2 = threading.Thread(target=read_datafile_2_thread)

# Start threads
thread1.start()
thread2.start()

# Wait for both threads to complete
thread1.join()
thread2.join()

# Run NN.
max_n = 1512  # Maximum number of points wou want to use (useful for testing).
data = pd.DataFrame([lat, lon, alt, val[0], err[0]]).T.iloc[:max_n, :]
data_2 = pd.DataFrame([lat_2, lon_2, alt_2, val_2[0], err_2[0]]).T.iloc[:max_n, :]

data.columns = ['Latitude', 'Longitude', 'Altitude', 'Value', 'Error']
data_2.columns = ['Latitude', 'Longitude', 'Altitude', 'Value', 'Error']

# Uncomment to remove values with large errors (Errors larger than the values themselves).
# mask = data['Value'] >= data['Error']
# data = data[mask]

# Train the neural network.
volumetric_nn(data, resolution=(100, 100, 30), cbar_lim=(1e10, 3e11), real_dist=False, fig3D=True)
volumetric_nn_2(data_2, resolution=(100, 100, 30), cbar_lim=(1e10, 3e11), real_dist=False, fig3D_2=True)

image1 = cv2.imread('Long_Pulse_3D.jpg')
image2 = cv2.imread('Alternating_code_3D.jpg')

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Create a figure with 2 subplots (side by side)
fig, axes = plt.subplots(1, 2)

fig.suptitle(f'UTC Timeframe: {start} - {end}')
axes[0].imshow(image1)
axes[0].axis('off')  # Hide the axis

axes[1].imshow(image2)
axes[1].axis('off')  # Hide the axis

plt.savefig('Combined.jpg')
plt.show()

# Things to try.
# 1. Include errors by weighing each data point (density/(error^2)?), or filter them out.
# 2. Overfitting penalization with neural networks. See: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
