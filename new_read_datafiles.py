# Imports.
import cartopy.crs as ccrs
import datetime as dt
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymap3d as pm
from scipy.spatial import ConvexHull
import tables
import tensorflow as tf
import geopy.distance




def read_datafiles(filenames, start_time, end_time, chi2lim=(0.1, 10)):
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
    df = pd.DataFrame()
    for filename in filenames:
        # Open the file, and extract the relevant data.
        data_curr = pd.DataFrame()
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

        # Run NN.
        max_n = 1512 # Maximum number of points wou want to use (useful for testing).
        data_curr = pd.DataFrame([latitude, longitude, altitude, value[0], error[0]]).T.iloc[:max_n, :]
        data_curr.columns = ['Latitude', 'Longitude', 'Altitude', 'Value', 'Error']  
        df = pd.concat([df, data_curr])
    # Return.
    return df
