import csv
import os
import numpy as np
import pandas as pd
from varname import nameof


def extract_data(path):
    """
    Parameters:
    1. path - Absolute path of the location where you saved the csv files

    """

    columns = ['Station', 'Year', 'Month', 'Day', 'Daily Rainfall Total (mm)',
               'Highest 30 min Rainfall (mm)', 'Highest 60 min Rainfall (mm)',
               'Highest 120 min Rainfall (mm)', 'Mean Temperature (°C)',
               'Maximum Temperature (°C)', 'Minimum Temperature (°C)',
               'Mean Wind Speed (km/h)', 'Max Wind Speed (km/h)']

    os.chdir(path)

    for folder in os.listdir("data"):
        globals()[folder] = pd.DataFrame()

        for file in os.listdir("data/{}".format(folder)):
            with open("{}\\{}\\{}".format(path, folder, file)) as file_:
                csv_reader = csv.reader(file_, delimiter=",")
                temp = pd.DataFrame([_ for _ in csv_reader])
                temp.drop(0, axis=0, inplace=True)
                temp.columns = columns

            globals()[folder] = globals()[folder].append(temp)

    return globals()[folder]


def process_features(datasets):
    """
    Parameters:
    1. datasets - an array of your datasets.
       eg: [yishun, changi, tuassouth]

    What this function does:
    1. Initially, the variables are in the first row of the dataframe. The column names is an np.arange().
       Hence, we set the variables (eg: Year, Mean Wind Speed, Minimum Temperature,...) as the index.
    2. We replace '—' and '-' with NumPy's NaN.
    3. Initially, the numeric features (eg: Temperature, Wind Speed,...) were strings.
       Hence, we change them into floats.

    """

    numeric_features = ['Year', 'Month', 'Day', 'Daily Rainfall Total (mm)',
                        'Highest 30 min Rainfall (mm)', 'Highest 60 min Rainfall (mm)',
                        'Highest 120 min Rainfall (mm)', 'Mean Temperature (°C)',
                        'Maximum Temperature (°C)', 'Minimum Temperature (°C)',
                        'Mean Wind Speed (km/h)', 'Max Wind Speed (km/h)']

    for dataset in datasets:
        dataset = dataset.reset_index().drop("index", axis=1)
        dataset.replace(['—', '-'], np.nan, inplace=True)
        dataset[numeric_features] = dataset[numeric_features].apply(pd.to_numeric)
        dataset.to_csv("{}.csv".format(nameof(dataset)))
    
    return


