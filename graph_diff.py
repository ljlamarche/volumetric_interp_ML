from support_functions_mod import read_datafile
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
from matplotlib.widgets import Slider
from graph_support_functions import create_all_data, index_data, filter_data


def get_data():
    # Lists of potential data to test.
    # data_synthetic = [[r"chapman_data_01.h5", r"chapman_data_015.h5", r"chapman_data_02.h5", r"chapman_data_025.h5", r"chapman_data_03.h5"], [r"circular_data_01.h5", r"circular_data_015.h5", r"circular_data_02.h5", r"circular_data_025.h5"], [r"gradient_data_01.h5", r"gradient_data_015.h5", r"gradient_data_02.h5", r"gradient_data_025.h5"], [r"tubular_data_01.h5", r"tubular_data_015.h5", r"tubular_data_02.h5", r"tubular_data_025.h5"], [r"wave_data_01.h5", r"wave_data_015.h5", r"wave_data_02.h5", r"wave_data_025.h5"]]
    # config_files = ['chapman_config.yaml', 'circular_config.yaml', 'gradient_config.yaml', 'tubular_config.yaml', 'wave_config.yaml']
    # datas = [[r"chapman_data_015.h5", r"chapman_data_02.h5", r"chapman_data_025.h5", r"chapman_data_03.h5", r"chapman_data_035.h5", r"chapman_data_ac.h5"], [r"circular_data_015.h5", r"circular_data_02.h5", r"circular_data_025.h5", r"circular_data_03.h5", r"circular_data_035.h5", r"circular_data_ac.h5"], [r"gradient_data_015.h5", r"gradient_data_02.h5", r"gradient_data_025.h5", r"gradient_data_03.h5", r"gradient_data_035.h5", r"gradient_data_ac.h5"], [r"tubular_data_015.h5", r"tubular_data_02.h5", r"tubular_data_025.h5", r"tubular_data_03.h5", r"tubular_data_035.h5", r"tubular_data_ac.h5"], [r"wave_data_015.h5", r"wave_data_02.h5", r"wave_data_025.h5", r"wave_data_03.h5", r"wave_data_035.h5", r"wave_data_ac.h5"]]
    c_datas = [[r"gradient_circular_data_ac.h5", r"gradient_circular_data_025.h5"], [r"chapman_circular_data_ac.h5", r"chapman_circular_data_025.h5"], [r"chapman_gradient_data_ac.h5", r"chapman_gradient_data_025.h5"], [r"wave2_data_ac.h5", r"wave2_data_025.h5"], [r"two_circles_data_ac.h5", r"two_circles_data_025.h5"]]
    config_files = ["gradient_circular_config.yaml", "chapman_circular_config.yaml", "chapman_gradient_config.yaml", "wave2_config.yaml", "two_circles_config.yaml"]

    # n: Determines which datafile is tested.
    n = 0
    
    # Inputs: Directory to data file, start and end times.
    # data_file = data_synthetic[n]
    data_file = c_datas[n]
    config_file = config_files[n]

    start = "2016-09-13T00:00:01"
    end = "2016-09-13T00:00:10"
    
    # Convert start times to datetime format.
    start = dt.datetime.strptime(start, '%Y-%m-%dT%H:%M:%S')
    end = dt.datetime.strptime(end, '%Y-%m-%dT%H:%M:%S')

    # Finding multiple average errors to find the mean average error and standard deviation. 
    mae_list = []
    for i in range(3):
        df, stations = create_all_data(start, end, data_file, 0, config_file)
        mae_list.append(np.mean(abs(np.array(df['Value']))))

    # Mean average error and standard deviation calculation.
    mae = np.mean(mae_list)
    stdd = np.std(mae_list)
        

    # Removing some data points so the plot is easier to rotate.
    rows_remove = df.sample(n=127500, random_state = 1).index
    df = df.drop(rows_remove)

    # Creating the reference date to find the limits for plotting.
    ref_data = read_datafile(data_file, start, end)

    # Indexing the data (assigning each row to a layer).
    df, ref_data = index_data([df, ref_data], stations)

    # Filtering data so that only points within the known region are shown.
    df_filtered = filter_data(df, ref_data)

    
    return df_filtered, mae, stdd

def main():
    # Getting the data.
    data, mae, stdd = get_data()

    # Scaling for plotting
    data['Altitude'] = np.array(data['Altitude'])/1000
    data = data.sort_values(by='Layer')
    
    # Plotting

    # Creating the figure.
    Cen3D = plt.figure()
    ax = Cen3D.add_subplot(111, projection='3d')

    # Creating a scatter plot.
    sc = ax.scatter(data['Longitude'],data['Latitude'],data['Altitude'],cmap='bwr',c=data['Value'], vmin=-2.5e11, vmax=2.5e11)
    
    # Adding the labels.
    ax.set_xlabel('Longitude', labelpad=5)
    ax.set_ylabel('Latitude', labelpad=5)
    ax.set_zlabel('Altitude (km)', labelpad=7)

    # Adding the colorbar legend.
    cbar = plt.colorbar(sc, pad=0.2, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Value')

    # Creating a slider axis and slider
    ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Altitude', 0, 800, valinit=0)

    # Update function
    def update(val):
        current_alt = slider.val
        threshold = 50  # Altitude threshold for filtering
        mask = np.abs(data['Altitude'] - current_alt) < threshold
        sc._offsets3d = (data['Longitude'][mask], data['Latitude'][mask], data['Altitude'][mask])
        sc.set_array(data['Value'][mask])
        Cen3D.canvas.draw_idle()

    # Connecting the slider to the update function
    slider.on_changed(update)

    plt.show()
    print("MAE (Cone):", mae)
    print("STD (Cone):", stdd)
 
    


 

    




if __name__ == main():
    main()
