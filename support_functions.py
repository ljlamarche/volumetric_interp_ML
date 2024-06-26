"""
Volumetric interpolation using artificial neural networks.

Authors: Nicolas Gachancipa, Leslie Lamarche

Stanford Research Institute (SRI)
Embry-Riddle Aeronautical University
"""

# Imports.
import cartopy.crs as ccrs
import datetime as dt
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymap3d as pm
from scipy.spatial import ConvexHull
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tables
import tensorflow as tf
import geopy.distance



def read_datafile(filename, start_time, end_time, chi2lim=(0.1, 10)):
    """
    Read a processed AMISR hdf5 file and return the time, coordinates, values, and errors as arrays.

    Parameters:
        filename: [str]
            filename/path of processed AMISR hdf5 file
        start_time: [datetime]
            All the data before the start time is filtered out. Format: '%Y-%m-%dT%H:%M:%S'
        end_time: [datetime]
            All the data after the end time is filtered out. Format: '%Y-%m-%dT%H:%M:%S'
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

    # Open the file, and extract the relevant data.
    with tables.open_file(filename, 'r') as h5file:
        # Obtain time.
        utime = h5file.get_node('/Time/UnixTime')[:]

        # Obtain altitude, longitude, latitude, chi2, and fit code for the given times.
        altitude = h5file.get_node('/Geomag/Altitude')[:]
        latitude = h5file.get_node('/Geomag/Latitude')[:]
        longitude = h5file.get_node('/Geomag/Longitude')[:]

        # Filter time to only include times within the given start and end times.
        idx = np.argwhere((utime[:, 0] >= (start_time - dt.datetime.utcfromtimestamp(0)).total_seconds()) & (
                utime[:, 1] <= (end_time - dt.datetime.utcfromtimestamp(0)).total_seconds())).flatten()
        if len(idx) == 0:
            idx = [0]
        utime = utime[idx, :]
        utime = utime.mean(axis=1)

        # Obtain the density values, errors, chi2 values, and fitcode (only for the selected times).
        value = h5file.get_node('/FittedParams/Ne')[idx, :, :]
        error = h5file.get_node('/FittedParams/dNe')[idx, :, :]
        chi2 = h5file.get_node('/FittedParams/FitInfo/chi2')[idx, :, :]
        fc = h5file.get_node('/FittedParams/FitInfo/fitcode')[idx, :, :]

    # Flatten the arrays.
    altitude = altitude.flatten()
    latitude = latitude.flatten()
    longitude = longitude.flatten()

    # Reshape arrays.
    value = value.reshape(value.shape[0], -1)
    error = error.reshape(error.shape[0], -1)
    chi2 = chi2.reshape(chi2.shape[0], -1)
    fit_code = fc.reshape(fc.shape[0], -1)

    # This accounts for an error in some of the hdf5 files where chi2 is overestimated by 369.
    if np.nanmedian(chi2) > 100.:
        chi2 = chi2 - 369.

    # data_check: 2D boolean array for removing "bad" data.
    # Each column corresponds to a different "check" condition.
    # TRUE for "GOOD" point; FALSE for "BAD" point.
    # A "good" record that shouldn't be removed should be TRUE for EVERY check condition.
    data_check = np.array([chi2 > chi2lim[0], chi2 < chi2lim[1], np.isin(fit_code, [1, 2, 3, 4])])

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
    c_hull = ConvexHull(r_cart)
    hull_vert = r_cart[c_hull.vertices]

    # Return.
    return utime, latitude, longitude, altitude, value, error, hull_vert


class StopAtLossValue(tf.keras.callbacks.Callback):
    """
    Callback to stop training a neural network when a certain loss is achieved.
    Change the set_point_loss value to define the desired loss value.
    """

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        set_point_loss = 0.001
        if logs.get('loss') <= set_point_loss:
            self.model.stop_training = True


def volumetric_nn(df, start, end, resolution=(10, 10, 10), cbar_lim=None, real_dist=False, density_range=(1e10, 1e12),
                  fig3D=True, save_imgs=False):
    """
    Parameters:
        df: [dataframe]
            Pandas daraframe with 4 columns (not including the index column). Such columns must be: 'Latitude',
            'Longitude', 'Altitude' and 'Value'. The 'Vavlue' column must contain the electron density measures.
        resolution: [tuple or list]
            Tuple or list containing three integers. The resolution is the size of the 3D grid (x, y, z), corresponding
            to (longitude, latitude, altitude). The higher the resolution, the longer it takes for the code to run.
        cbar_lim: [tuple]
            Elctron density color bar limits (e.g. (1e10, 1e11)).
        real_dist: [boolean]
            Set to True if you want the output to show the distances in kilometers from the radar (rather than in
            longitude and latitude degrees).
        density_range: [tuple]
            Tuple containing the lower and upper limits of allowed electron density values. Any radar measurement
            outside this range is removed before training the neural network. Default: (10e10, 10e12).
        fig3D [boolean]:
            True for 3D plot, False for 2D plot only.
        save_imgs:
            True if you want to save the images in a png format to the local directory.
    Returns:
        2D or 3D plot.
    """

    #################
    # DATA PROCESSING
    #################

    # Drop nans and sort dataframe by energy value.
    df = df.dropna()
    df = df.sort_values(by='Value') 

    # Convert latitudes/longitudes to real distances in km (if real_dist == True).
    if real_dist:

        # Define the location of RISR.
        risr_coords = (74.72, -94.9)

        # Compute the real distances (in km) between every measurement's latitude and the radar's latitude.
        # Do the same for longitude.
        new_lats, new_lons = [], []
        for lat, lon in zip(df['Latitude'], df['Longitude']):
            lat_coord = (lat, risr_coords[1])
            lon_coord = (risr_coords[0], lon)
            lat_dist = geopy.distance.geodesic(risr_coords, lat_coord).km
            lon_dist = geopy.distance.geodesic(risr_coords, lon_coord).km
            if lat < risr_coords[0]:
                lat_dist *= -1
            if lon < risr_coords[1]:
                lon_dist *= -1
            new_lats.append(lat_dist)
            new_lons.append(lon_dist)

        # Replace in the dataframe. From now on, the 'Latitude' and 'Longitude' columns contain
        df['Latitude'] = new_lats
        df['Longitude'] = new_lons

    # Filter out electron densities outside the given range.
    df = df[df['Value'] >= density_range[0]]
    df = df[df['Value'] <= density_range[1]]

    # Find the log10 of the electron density values. This conversion allows the neural network to be trained more
    # efficiently. Save the converted values to a new column.
    df['Log Value'] = np.log10(df['Value'])
    df['Log Error'] = np.log10(df['Error'])

    # Drop the data instances that have nan values in any column. Sometimes the log conversion leads to nan errors.
    df = df.dropna()

    # Defining the weight of each point.
    y_actual = list(df['Log Value'])
    y_error = list(df['Log Error'])

    weights = []
    for i in range(len(y_actual)):
        # Finding how each error relates to the actual value (formula was Nick's suggestion).
        weight = y_actual[i] / y_error[i]**2
        weights.append(weight)
    weights = np.array(weights)
    
    # Dropping the Error column.
    df = df.drop(['Error'], axis=1)

    # Normalize the data (in a new dataframe). Normalized data allows the neural network to be trained faster and more
    # effectively. Normalization is a rescaling of the data from the original range so that all values are within the
    # range of 0 and 1.
    # https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
    df_train = df.copy()
    for column in df_train:
        df_train[column] = (df_train[column] - min(df_train[column])) / (max(df_train[column]) - min(df_train[column]))


    ##########
    # TRAINING
    ##########

    # Define neural network inputs (x and y).
    # x: Normalized logarithmic density values (response variable).
    # y: Normalized latitude, longitude, and altitude (predictor variables).
    y = df_train['Log Value']
    x = df_train.drop(['Value', 'Log Value', 'Log Error'], axis=1)

    # Get the minimum and maximum values of each column (lat, lon, alt, logdensities). Save the min/max values to
    # x_lim (lat, lot, alt) and y_lim (densities in log form).
    y_org = pd.DataFrame(df['Log Value'])
    x_org = df.drop(['Value', 'Log Value'], axis=1)
    x_lim = [[min(x_org[c]), max(x_org[c])] for c in x_org]
    y_lim = [min(y_org.iloc[:, 0]), max(y_org.iloc[:, 0])]

    # Compute the lat/lon aspect ratio (useful for plotting).
    aspect_ratio = abs(x_lim[1][0] - x_lim[1][1]) / abs(x_lim[0][0] - x_lim[0][1])

    # Extract the resolution of the grid (x, y, z) = (lon, lat, alt). Given by the user.
    rows, columns, heights = resolution

    # Predict a value for each element in the 3D grid (every lon, lat, alt combination). Save the results to a variable
    # called y_preditct.
    predict = [[r / (rows - 1), c / (columns - 1), h / (heights - 1)] for r in range(rows) for c in range(columns) for
                h in range(heights)]
    predict = np.array(predict)

    # A list of predicted densities.
    predictions = []

    # Training the Artifical Neural Network 15 times (can be changed).
    for i in range(15):
        ####################
        ## NEURAL NETWORK ##
        ####################
        network = tf.keras.Sequential([tf.keras.layers.Dense(units=64, input_shape=[x.shape[1]], activation='tanh'),
                                   tf.keras.layers.Dense(units=32, activation='tanh'),
                                   tf.keras.layers.Dense(units=16, activation='tanh'),
                                   tf.keras.layers.Dense(units=8, activation='tanh'),
                                   tf.keras.layers.Dense(units=4, activation='tanh'),
                                   tf.keras.layers.Dense(units=1, activation='sigmoid')])
        
        # Compile the network: Define the optimizer and the loss function. See the following link for reference:
        # https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c
        network.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

        # Define a checkpoint. This allows the network to save the "best weights" to an h5 file.
        checkpoint = tf.keras.callbacks.ModelCheckpoint('weights.keras', verbose=1, monitor='loss', save_best_only=True, mode='auto')
    
        # Early stopping to prevent overtraining; stopping training if the loss doesn't improve after 17 epochs.
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',mode='auto',verbose=0,patience=17)
        # Train the network.
        # Comment out this line if you already have a weights.h5 file and you want to skip the training process.
        network.fit(x, y, epochs=300, sample_weight=weights, callbacks=[StopAtLossValue(), checkpoint, early_stopping])
        
        # Adding the predicted densities to the list.
        predictions.append(network.predict(predict))

    # Finding the sum of all of the predictions.
    sum = 0
    for i in range(len(predictions)):
        sum = np.add(sum, predictions[i])

    # Finding the average of the predictions.        
    y_predict = sum / len(predictions)
    

    # ##########
    # # PLOTTING
    # ##########

    # Create a 3D array with the results (ressembling a 3D desity map). Convert the results (which are in a log scale)
    # to real density values.
    array = np.zeros((rows, columns, heights))
    for e, i in enumerate(predict):
        # Obtain the x, y, z (lon, lat, alt) location.
        row = int(round(i[0] * (rows - 1), 0))
        column = int(round(i[1] * (columns - 1), 0))
        level = int(round(i[2] * (heights - 1), 0))

        # Convert predicted log density value to absolute scale, and save the value to the 3D array.
        array[row, column, level] = (10 ** (y_predict[e][0] * (y_lim[1] - y_lim[0]) + y_lim[0]))

    # Define the color bar limits (if not provided by the user).
    if cbar_lim is None:
        max_val = np.max(df['Value'])
        min_val = np.min(df['Value'])
        cbar_lim = [min_val, max_val]

    # Create a plot, and corresponding subplots.
    fig = plt.figure()
    if fig3D:
        # ax = fig.add_subplot(1, 1, 1)
        ax = fig.add_subplot(121)
        ax_slider = plt.axes([0.15, 0.1, 0.25, 0.03])
    else:
        if real_dist:
            ax = fig.add_subplot()
        else:
            ax = fig.add_subplot(projection=ccrs.PlateCarree())
        ax_slider = plt.axes([0.3, 0.1, 0.42, 0.03])
    plt.subplots_adjust(bottom=0.25)

    # Extract the first layer (lowest altitude).
    layer = array[:, :, 0]

    # Create grid plot and slider. Plot the first layer (lower altitude).
    im = ax.imshow(layer, cmap='jet', origin='lower', extent=[x_lim[1][0], x_lim[1][1], x_lim[0][0], x_lim[0][1]],
                   vmin=cbar_lim[0], vmax=cbar_lim[1], aspect=aspect_ratio)
    slider = Slider(ax_slider, 'Altitude (m)', x_lim[2][0], x_lim[2][1], x_lim[2][0],
                    valstep=((x_lim[2][1] - x_lim[2][0]) / (heights - 1)))

    # Set x and y axis labels.
    if real_dist:
        ax.set_xlabel('Km from radar - East (+)')
        ax.set_ylabel('Km from radar - North (+)')
    else:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    # Add the original data points to their closest layer: Add a new column to the original dataframe that contains
    # the corresponding layer index of each data point.
    stations = list(np.linspace(x_lim[2][0], x_lim[2][1], heights))
    indices = []
    for a in list(df['Altitude']):
        closest_alt = min(stations, key=lambda x: abs(x - a))
        indices.append(stations.index(closest_alt))
    df['Layer'] = indices

    # Plot colorbar, using the predefined color bar limits.
    fig.colorbar(im, ax=ax, location='right', orientation='vertical', )
    cmap = im.set_clim(cbar_lim[0], cbar_lim[1])

    # Plot original data points of the first layer (layer = 0).
    points = df[df['Layer'] == 0]
    ax.scatter(list(points['Longitude']), list(points['Latitude']), color='white', s=40)
    ax.scatter(list(points['Longitude']), list(points['Latitude']), cmap=cmap, s=20)

    # Create function to be called when the slider value (altitude) is changed.
    def update(alt_value, save=False):
        val = (alt_value - x_lim[2][0]) / (x_lim[2][1] - x_lim[2][0])
        ax.clear()
        pts = df[df['Layer'] == stations.index(alt_value)]
        l, r, b, u = min(pts['Longitude']), max(pts['Longitude']), min(pts['Latitude']), max(pts['Latitude'])
        ax.imshow(array[:, :, int(round(val * (heights - 1), 0))], cmap='jet', origin='lower',
                  extent=[x_lim[1][0], x_lim[1][1], x_lim[0][0], x_lim[0][1]], vmin=cbar_lim[0], vmax=cbar_lim[1],
                  aspect=aspect_ratio)
        if real_dist:
            ax.set_xlabel('Km from radar - East (+)')
            ax.set_ylabel('Km from radar - North (+)')
        else:
            # print('here')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        ax.plot([l, r], [b, b], 'k')
        ax.plot([l, r], [u, u], 'k')
        ax.plot([l, l], [b, u], 'k')
        ax.plot([r, r], [b, u], 'k')
        ax.scatter(list(pts['Longitude']), list(pts['Latitude']), color='white', s=40)
        ax.scatter(list(pts['Longitude']), list(pts['Latitude']), c=list(pts['Value']), cmap='jet', s=20,
                   vmin=cbar_lim[0], vmax=cbar_lim[1])

        # Save the layer to a png file.
        if save:
            ax.set_title('Altitude: {} km.'.format(round(alt_value / 1000, 1)))
            plt.savefig('fig_{}.jpg'.format(int(alt_value / 1000)))

    # Save the images in png files (if selected by the user).
    if save_imgs:
        for value in range(int(x_lim[2][0]), int(x_lim[2][1]), int((x_lim[2][1] - x_lim[2][0]) / (heights - 1))):
            update(value, save=True)

    # Call update function when slider value is changed.
    slider.on_changed(update)

    # Plot 3D figure.
    if fig3D:

        # Create new meshgrid. (xx = lon, yy = lat).
        xx, yy = np.meshgrid(np.linspace(x_lim[1][0], x_lim[1][1], rows),
                             np.linspace(x_lim[0][0], x_lim[0][1], columns))
        axis = fig.add_subplot(122, projection='3d')

        # Set z limits.
        axis.set_zlim(x_lim[2][0] / 1000, x_lim[2][1] / 1000)

        # 3D Plot.
        number_of_layers = heights
        if heights < number_of_layers:
            number_of_layers = heights
        for v in range(0, array.shape[-1], int(array.shape[-1] / number_of_layers)):
            print('Plotting 3D layers: {} out of {}.'.format(v + 1, number_of_layers))
            layer = array[:, :, v]
            axis.contourf(xx, yy, layer, 100, zdir='z',
                          offset=((v / number_of_layers) * (x_lim[2][1] - x_lim[2][0]) + x_lim[2][0]) / 1000,
                          vmin=cbar_lim[0], vmax=cbar_lim[1], cmap='jet')

        # Set labels.
        if real_dist:
            plt.xlabel('Km from radar - North (+)')
            plt.ylabel('Km from radar - East (+)')
        else:
            plt.xlabel('Latitude (°)')
            plt.ylabel('Longitude (°)')
        axis.set_zlabel('Altitude (km)')
        plt.suptitle(f'Date: {start:%Y-%m-%d}  Timeframe (UTC): {start:%H:%M:%S} - {end:%H:%M:%S}')
        plt.show()

    # Compute and display r2 coefficient.
    y_hat = network.predict(x)
    y_pred = list(y_hat.T[0])
    y_true = list(y)
    r2 = metrics.r2_score(y_true, y_pred)
    print(f"R2 Coefficient: {r2}.")
    print(f"Mean absolute error: {mean_absolute_error(y_true, y_pred)}")
    print(f"Mean squared error: {mean_squared_error(y_true, y_pred)}")
   
