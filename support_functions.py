#
# Support functions.
#
# Author: Nicolas Gachancipa
#

# Imports.
import cartopy.crs as ccrs
import datetime as dt
import math
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymap3d as pm
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from sklearn import metrics
import tables
import tensorflow as tf
import geopy.distance


def read_datafile(filename, start_time, end_time, chi2lim=(0.1, 10)):
    """
    Read a processed AMISR hdf5 file and return the time, coordinates, values, and errors as arrays.

    Parameters:
        filename: [str]
            filename/path of processed AMISR hdf5 file
        chi2lim: [tuple]
            remove all points outside the given chi2 range (lower limit, upper limit)

    Returns:
        utime: [ndarray (nrecordsx2)]
            start and end time of each record (Unix Time)
        latitude: [ndarray (npoints)]
            geodetic latitude of each point
        longitude: [ndarray (npoints)]
            geodetic longitude of each point
        altitude: [ndarray (npoints)]
            geodetic altitude of each point
        value: [ndarray (nrecordsxnpoints)]
            parameter value of each data point
        error: [ndarray (nrecordsxnpoints)]
            error in parameter values
    """

    # Open the file.
    # df = pd.read_hdf(filename)
    # print(df)
    # quit()

    # Open the file, and extract the relevant data.
    with tables.open_file(filename, 'r') as h5file:
        # Obtain time.
        utime = h5file.get_node('/Time/UnixTime')[:]

        # Obtain altitude, longitude, latitude, chi2, and fit code for the given times.
        alt = h5file.get_node('/Geomag/Altitude')[:]
        lat = h5file.get_node('/Geomag/Latitude')[:]
        lon = h5file.get_node('/Geomag/Longitude')[:]

        # Filter time to only include times within the given start and end times.
        idx = np.argwhere((utime[:, 0] >= (start_time - dt.datetime.utcfromtimestamp(0)).total_seconds()) & (
                utime[:, 1] <= (end_time - dt.datetime.utcfromtimestamp(0)).total_seconds())).flatten()
        if len(idx) == 0:
            idx = [0]
        utime = utime[idx, :]
        utime = utime.mean(axis=1)

        # Obtain the density values, errors, chi2 values, and fitcode (only for the selected times).
        val = h5file.get_node('/FittedParams/Ne')[idx, :, :]
        err = h5file.get_node('/FittedParams/dNe')[idx, :, :]
        c2 = h5file.get_node('/FittedParams/FitInfo/chi2')[idx, :, :]
        fc = h5file.get_node('/FittedParams/FitInfo/fitcode')[idx, :, :]

    # Flatten the arrays.
    altitude = alt.flatten()
    latitude = lat.flatten()
    longitude = lon.flatten()

    # Reshape arrays.
    value = val.reshape(val.shape[0], -1)
    error = err.reshape(err.shape[0], -1)
    chi2 = c2.reshape(c2.shape[0], -1)
    fitcode = fc.reshape(fc.shape[0], -1)

    # This accounts for an error in some of the hdf5 files where chi2 is overestimated by 369.
    if np.nanmedian(chi2) > 100.:
        chi2 = chi2 - 369.

    # data_check: 2D boolian array for removing "bad" data.
    # Each column correpsonds to a different "check" condition.
    # TRUE for "GOOD" point; FALSE for "BAD" point.
    # A "good" record that shouldn't be removed should be TRUE for EVERY check condition.
    data_check = np.array([chi2 > chi2lim[0], chi2 < chi2lim[1], np.isin(fitcode, [1, 2, 3, 4])])

    # If ANY elements of data_check are FALSE, flag index as bad data. Remove the bad data from the array.
    bad_data = np.squeeze(np.any(data_check is False, axis=0, keepdims=True))
    value[bad_data] = np.nan
    error[bad_data] = np.nan

    # Remove the points where coordinate arrays are NaN.
    value = value[:, np.isfinite(altitude)]
    error = error[:, np.isfinite(altitude)]
    latitude = latitude[np.isfinite(altitude)]
    longitude = longitude[np.isfinite(altitude)]
    altitude = altitude[np.isfinite(altitude)]

    # Convert coordinates to hull_vert.
    x, y, z = pm.geodetic2ecef(latitude, longitude, altitude)
    r_cart = np.array([x, y, z]).T
    chull = ConvexHull(r_cart)
    hull_vert = r_cart[chull.vertices]

    # Return.
    return utime, latitude, longitude, altitude, value, error, hull_vert


def train_nn(lat, lon, alt, val):
    """

    Returns:

    """

    # Filter by altitude.
    alt /= 1e3
    df = pd.DataFrame(np.array([lat, lon, alt])).T
    df.columns = ['Latitude', 'Longitude', 'Altitude']
    df = pd.concat([df, pd.DataFrame(val).T], axis=1)
    df = df[df['Altitude'] <= 325]
    df = df[df['Altitude'] >= 275]

    # Test the first two rows.
    n = 2
    df = df.iloc[:n]

    # Normalize the data.
    lat, lon, alt, val = df['Latitude'], df['Longitude'], df['Altitude'], df[0]
    df['Latitude'] = (lat - min(lat)) / (max(lat) - min(lat))
    df['Longitude'] = (lon - min(lon)) / (max(lon) - min(lon))
    df['Altitude'] = (alt - min(alt)) / (max(alt) - min(alt))
    df[0] = (val - min(val)) / (max(val) - min(val))

    # Define neural network parameters.
    X = df[['Latitude', 'Longitude']]
    y = df[[0]]
    # print(X)
    # print(y)
    # print(X.shape)
    # print(y.shape)

    # Train the neural network.
    network = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[2])])
    network.compile(optimizer='sgd', loss='mse')
    network.fit(X, y, epochs=1000)

    # Prediction test.
    y_predict = network.predict([[0, 0],
                                 [0, 0.25],
                                 [0, 0.5],
                                 [0, 0.75],
                                 [0, 1],
                                 [0.25, 1],
                                 [0.5, 1],
                                 [0.75, 1],
                                 [1, 1]])
    print('Original:', y)
    print('Prediction: ', y_predict)


class stopAtLossValue(tf.keras.callbacks.Callback):

    def on_batch_end(self, batch, logs={}):
        THR = 0.001
        if logs.get('loss') <= THR:
            self.model.stop_training = True


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 1
    epochs_drop = 1
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def nn_model(df, rows=10, columns=10, r2=False, cbar_lim=None, distances=False):
    """

    Args:
        df:
        rows:
        columns:
        r2:
        cbar_lim:

    Returns:

    """
    # Drop nans and sort dataframe by energy value.
    df = df.dropna()
    df = df.sort_values(by='Value')

    # Convert latitudes/longitudes to distances (if distances == True).
    if distances:

        # RISR location.
        df = df.sort_values(by='Latitude')
        risr_coords = (74.72, -94.9)

        # Compute the distances between every point's latitude and the min lat.
        new_lats, new_lons = [], []
        for lat, lon in zip(df['Latitude'], df['Longitude']):
            lat_coord = (lat, risr_coords[1])
            lon_coord = (risr_coords[0], lon)
            lat_dist = geopy.distance.vincenty(risr_coords, lat_coord).km
            lon_dist = geopy.distance.vincenty(risr_coords, lon_coord).km
            if lat < risr_coords[0]:
                lat_dist *= -1
            if lon < risr_coords[1]:
                lon_dist *= -1
            new_lats.append(lat_dist)
            new_lons.append(lon_dist)

        # Replace in the df.
        df['Latitude'] = new_lats
        df['Longitude'] = new_lons

    # Limits (10^9 or 10^10, 10^13 or 10^12)
    df = df[df['Value'] >= 10 ** 10]
    df = df[df['Value'] <= 10 ** 12]

    # Remove outliers (outside 2 stdev).
    mean = df['Value'].describe()['mean']
    stdev = df['Value'].describe()['std']
    df = df[df['Value'] >= (mean - 2 * stdev)]
    df = df[df['Value'] <= (mean + 2 * stdev)]
    org_df = df.copy()

    # Drop the energy values above 2 standard deviations.
    # mean = df['Value'].describe()['mean']
    # stdev = df['Value'].describe()['std']
    # df = df[df['Value'] <= mean + 1.5 * stdev]

    # Sort by value.
    df['Value'] = np.log10(df['Value'])
    df = df.dropna()

    # Example (density value)
    # [1e10, 1e11, 1e12]
    # [10, 11, 12]
    # [0, 0.5, 1]

    # Remove outliers by error.
    # q1 = df['Error'].quantile(0.1)
    # q3 = df['Error'].quantile(0.9)
    # df = df[df['Error'] >= q1]
    # df = df[df['Error'] <= q3]
    # y_org = pd.DataFrame(df['Value']).copy()
    # X_org = df.drop(['Value', 'Error'], axis=1).copy()

    # df['Error'].plot.hist(bins=100)

    # df = df[df['Value'] <= 5e11]
    # plt.show()
    # y_org = pd.DataFrame(df['Value']).copy()
    # X_org = df.drop(['Value'], axis=1).copy()

    # Save original df.
    df_org = df.copy()

    # Normalize the data.
    for column in df:
        df[column] = (df[column] - min(df[column])) / (max(df[column]) - min(df[column]))

    # Define neural network parameters.
    y = df['Value']  # density values [0 and 1]
    X = df.drop(['Value'], axis=1)  # lat, lon, alt
    # print(X.shape, y.shape)

    # X = [[0, 0, 0],
    #      [1, 1, 1]] - size: [2, 3]
    # y = [[1],
    #      [2]] - size [2, 1]

    # Input layer = 3 neurons (lat, lon, alt)

    # Multiply X by a matrix (A) of shape [3, 64].
    # X of shape [2, 3]*[3, 64] = [2, 64]
    # Activation function = [10000, -10000, 0, 0.25, 1, 2.5] - apply sigmoid: [1, 0, 0.5, ...]

    # hidden layer 1 = 64 neurons
    # X of shape [2, 64]*[64, 32] = [2, 32]

    # hidden layer 2 = 32 neurons
    # hidden layer 3 = 16 neurons
    # hidden layer 4 = 8 neurons
    # hidden layer 5 = 4 neurons
    # X of shape [2, 4]*[4, 1] = [2, 1]

    # Output layer = 1 neurons
    # Output shape = [2, 1]
    # Apply a final activation function: Output value between 0 and 1 (sigmoid).

    # Train the neural network.
    network = tf.keras.Sequential([tf.keras.layers.Dense(units=64, input_shape=[X.shape[1]], activation='tanh'),
                                   tf.keras.layers.Dense(units=32, activation='tanh'),
                                   tf.keras.layers.Dense(units=16, activation='tanh'),
                                   tf.keras.layers.Dense(units=8, activation='tanh'),
                                   tf.keras.layers.Dense(units=4, activation='tanh'),
                                   tf.keras.layers.Dense(units=1, activation='sigmoid')])
    # lrate = k.callbacks.LearningRateScheduler(step_decay)
    network.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

    # 1st iteration (epoch 1).
    # y_hat = [0, 0] and actual y = [1, 2] - MSE loss function: ((0 - 1)^2 + (0 - 2)^2)/2 = 2.5
    # Update the network - Update the coefficients using an optimization technique.
    # Save weights.h5

    # 2nd iteration (epoch 2).
    # y_hat = [0.5, 1] and actual y = [1, 2] - MSE loss function: ((0.5 - 1)^2 + (1 - 2)^2)/2 = 1.25/2 = 0.625
    # 0.625 < 2.5, then save weights.h5

    # 3rd
    # loss = 0.8
    # 0.8 > 0.625, don't save weights.h5

    checkpoint = tf.keras.callbacks.ModelCheckpoint('weights.h5', verbose=1, monitor='loss', save_best_only=True,
                                                    mode='auto')
    # Training happens.
    # network.fit(X, y, epochs=100, callbacks=[stopAtLossValue(), checkpoint])
    # At this point, you have a network and weights.

    # Load the best weights that have been saved to the h5 file.
    network.load_weights('weights.h5')

    # 1. 1500 data points from the radar.
    # 2. Train the network. - Best weights.
    # 3. Load the best weights.
    # 4. Predict for millions of datapoints. Create a grid of a by b by c. a = # of latitude stations
    # b = # of lon
    # c = # of alt
    # a = 30, b = 30, c = 30 = 27,000

    # Use the network to predict values.
    X_hat = [[0.5, 0.5, 0.5],
             [0.6, 0.6, 0.6]]
    y_hat = network.predict(X_hat)

    # print(list(y_hat.T[0]))

    # Get the minimum and maximum original values.
    y_org = pd.DataFrame(df_org['Value'])
    X_org = df_org.drop(['Value'], axis=1)
    X_lim = [[min(X_org[c]), max(X_org[c])] for c in X_org]
    y_lim = [min(y_org.iloc[:, 0]), max(y_org.iloc[:, 0])]
    aspect_ratio = abs(X_lim[1][0] - X_lim[1][1]) / abs(X_lim[0][0] - X_lim[0][1])

    # Prediction test.
    predict = [[r / (rows - 1), c / (columns - 1), a / (rows - 1)] for a in range(rows) for c in range(columns) for
               r in range(rows)]
    y_predict = network.predict(predict)

    # Plot.
    array = np.zeros((rows, columns, rows))
    for e, i in enumerate(predict):
        row = int(round(i[0] * (rows - 1), 0))
        column = int(round(i[1] * (columns - 1), 0))
        level = int(round(i[2] * (rows - 1), 0))

        # Denormalize the data.
        # min = 1e10, max = 1e12
        # min = 0, max = 1
        # y = 0.5 for 1e11
        array[row, column, level] = (10 ** (y_predict[e][0] * (y_lim[1] - y_lim[0]) + y_lim[0]))
    if cbar_lim is None:
        max_val = np.max(org_df['Value'])
        min_val = np.min(org_df['Value'])
        cbar_lim = [min_val, max_val]
    layer = array[:, :, 0]

    # Create a subplot
    fig3D = True
    save_imgs = False
    fig = plt.figure()

    if fig3D:
        ax = fig.add_subplot(121)
        axSlider = plt.axes([0.15, 0.1, 0.25, 0.03])
    else:
        if distances:
            ax = fig.add_subplot()
        else:
            ax = fig.add_subplot(projection=ccrs.PlateCarree())
        axSlider = plt.axes([0.3, 0.1, 0.42, 0.03])
    plt.subplots_adjust(bottom=0.25)

    if distances:
        im = ax.imshow(layer, cmap='jet', origin='lower', extent=[X_lim[1][0], X_lim[1][1], X_lim[0][0], X_lim[0][1]],
                       vmin=cbar_lim[0], vmax=cbar_lim[1], aspect=aspect_ratio)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        slider = Slider(axSlider, 'Altitude (m)', X_lim[2][0], X_lim[2][1], X_lim[2][0],
                        valstep=((X_lim[2][1] - X_lim[2][0]) / (rows - 1)))
    else:
        im = ax.imshow(layer, cmap='jet', origin='lower', extent=[X_lim[1][0], X_lim[1][1], X_lim[0][0], X_lim[0][1]],
                       vmin=cbar_lim[0], vmax=cbar_lim[1], aspect=aspect_ratio, transform=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True)
        ax.coastlines(resolution='50m', color='black', linewidth=0.2)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        slider = Slider(axSlider, 'Altitude (m)', X_lim[2][0], X_lim[2][1], X_lim[2][0],
                        valstep=((X_lim[2][1] - X_lim[2][0]) / (rows - 1)))

    # Add points to their closest layer.
    stations = list(np.linspace(X_lim[2][0], X_lim[2][1], rows))
    indices = []
    for a in list(org_df['Altitude']):
        closest_alt = min(stations, key=lambda x: abs(x - a))
        indices.append(stations.index(closest_alt))
    org_df['Layer'] = indices

    # Plot colorbar.
    # im.set_clim(min_val, max_val)
    fig.colorbar(im, ax=ax, location='right', orientation='vertical', )
    cmap = im.set_clim(cbar_lim[0], cbar_lim[1])

    # Plot points.
    points = org_df[org_df['Layer'] == 0]
    ax.scatter(list(points['Longitude']), list(points['Latitude']), color='white', s=40)
    ax.scatter(list(points['Longitude']), list(points['Latitude']), cmap=cmap, s=20)

    # Create function to be called when slider value is changed
    def update(alt_value):
        value = (alt_value - X_lim[2][0]) / (X_lim[2][1] - X_lim[2][0])
        ax.clear()
        points = org_df[org_df['Layer'] == stations.index(alt_value)]
        l, r, b, u = min(points['Longitude']), max(points['Longitude']), min(points['Latitude']), \
                     max(points['Latitude'])
        if distances:
            ax.imshow(array[:, :, int(round(value * (rows - 1), 0))], cmap='jet', origin='lower',
                      extent=[X_lim[1][0], X_lim[1][1], X_lim[0][0], X_lim[0][1]],
                      vmin=cbar_lim[0], vmax=cbar_lim[1], aspect=aspect_ratio)
            ax.plot([l, r], [b, b], 'k')
            ax.plot([l, r], [u, u], 'k')
            ax.plot([l, l], [b, u], 'k')
            ax.plot([r, r], [b, u], 'k')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        else:
            ax.imshow(array[:, :, int(round(value * (rows - 1), 0))], cmap='jet', origin='lower',
                      extent=[X_lim[1][0], X_lim[1][1], X_lim[0][0], X_lim[0][1]], vmin=cbar_lim[0], vmax=cbar_lim[1],
                      aspect=aspect_ratio, transform=ccrs.PlateCarree())
            ax.gridlines(draw_labels=True)
            ax.coastlines(resolution='50m', color='black', linewidth=0.2)
        ax.scatter(list(points['Longitude']), list(points['Latitude']), color='white', s=40)
        ax.scatter(list(points['Longitude']), list(points['Latitude']), c=list(points['Value']), cmap='jet', s=20,
                   vmin=cbar_lim[0], vmax=cbar_lim[1])

        if save_imgs:
            ax.set_title('Altitude: {} km.'.format(round(alt_value / 1000, 1)))
            plt.savefig('fig_{}.jpg'.format(e))

    if save_imgs:
        for e, value in enumerate(
                range(int(X_lim[2][0]), int(X_lim[2][1]), int((X_lim[2][1] - X_lim[2][0]) / (rows - 1)))):
            update(value)

    # Call update function when slider value is changed
    slider.on_changed(update)

    # Plot 3D figure.
    if fig3D:
        # Create new fig.
        xx, yy = np.meshgrid(np.linspace(X_lim[0][0], X_lim[0][1], rows),
                             np.linspace(X_lim[1][0], X_lim[1][1], columns))
        axis = fig.add_subplot(122, projection='3d')

        # Set z limits.
        axis.set_zlim(X_lim[2][0] / 1000, X_lim[2][1] / 1000)

        # Plot.
        number_of_layers = rows
        if rows < number_of_layers:
            number_of_layers = rows
        for v in range(0, array.shape[-1], int(array.shape[-1] / number_of_layers)):
            print('Plotting 3D layers: {} out of {}.'.format(v + 1, number_of_layers))
            layer = array[:, :, v].transpose()
            axis.contourf(xx, yy, layer, 100, zdir='z',
                          offset=((v / number_of_layers) * (X_lim[2][1] - X_lim[2][0]) + X_lim[2][0]) / 1000,
                          vmin=cbar_lim[0], vmax=cbar_lim[1], cmap='jet')

        # Set labels, and display plot.
        plt.xlabel('Longitude (°)')
        plt.ylabel('Latitude (°)')
        axis.set_zlabel('Altitude (km)')

    # Compute r2.
    if r2:
        y_hat = network.predict(X)
        y_pred = list(y_hat.T[0])
        y_true = list(y)
        r2 = metrics.r2_score(y_true, y_pred)
        # print(y_true)
        # print(y_pred)
        print('R2 Coefficient: {}.'.format(r2))

    # Display plot.
    plt.show()


# Inputs.
data_file = r"..\..\input_data\2016\11\27\20161127.002_lp_1min-fitcal.h5"
start_time = "2016-11-27T22:45:00"
end_time = "2016-11-27T22:48:00"
# data_file = r"..\..\input_data\risrn_synthetic_imaging_chapman.h5"
# start_time = "2016-09-13T00:00:01"
# end_time = "2016-09-13T00:00:10"
altitudes = [300]
color_lim = [1e9, 1e11]

# Convert start times to datetime format.
start_time = dt.datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
end_time = dt.datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S')

# Open file and extract the following arrays: time, latitude, longitude, altitude, value, error.
time, lat, lon, alt, val, err, hull_vert = read_datafile(data_file, start_time, end_time)

# Run NN.
n = 1512
data = pd.DataFrame([lat, lon, alt, val[0], err[0]]).T.iloc[:n, :]  # .sample(frac=1)
data.columns = ['Latitude', 'Longitude', 'Altitude', 'Value', 'Error']

# Normalize errors with altitude.
min_alt, max_alt = min(data['Altitude']), max(data['Altitude'])
# data['Error'] = ((data['Altitude'] - min_alt) / (max_alt - min_alt)) * data['Error']

# Remove values with large errors.
# Potentially, we could divide the density values by the errors squared.
# mask = data['Value'] >= data['Error']
# data = data[mask]
# data['Value'] = data['Value'] / data['Error']

# alts = list(data['Altitude'])
# steps = np.linspace(min(alts), max(alts), 10)
data = data.drop(['Error'], axis=1)
pd.set_option("display.max_rows", None, "display.max_columns", None)
nn_model(data, rows=100, columns=100, r2=True, cbar_lim=[1e10, 3e11], distances=True)

# Things to discuss.
# 1. Which errors should be filtered out? If an error is larger than the value?
# 2. I am applying limits (10^10, 10^12).
# 3. Standard deviation filter. Is it normally distributed?
# 4. High altitudes (Errors are too big). Should we make the error analysis altitude dependent?
# 5. Not including values with large errors leads to non-sense predictions.
# 6. Over fitting vs. epochs
# 7. Chapman - Worked!
# 8. Interpolation per beam is difficult to do. Instead, I am plotting the points at the closest layer.
#    More layers == Higher accuracy?
# 9. Map projection. Is it okay?

# Steps.
# 1. Include errors (by weighing each data point), or filter them out. (density/(error^2))
# 2. Over fitting penalization with neural networks.

# Combining chapman and real data points.
# Apply standard dev after applying log.
# Altitude dependent filter.
# Review statistics - over fitting.
# Map - 100km ticks
# Integration with Gemini.

# from amisrsynthata.Ionosphere import Ionosphere
#
# config_filg = '/path/to/config_file.ini'
# iono = Ionosphere(config_file)
#
# glat = 65.0
# glon = 100.0
# alt = 300000.
# ne = iono.density(glat, glon, alt)
