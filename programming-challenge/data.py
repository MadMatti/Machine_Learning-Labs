import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

TRAIN_FILE = "programming-challenge/resources/TrainOnMe-4.csv"
TEST_FILE = "programming-challenge/resources/TestOnMe-4.csv"

def load_file(file_name):
    """Load a file into a pandas dataframe"""
    return pd.read_csv(file_name)

def cleaning(df):
    '''Drop duplicates and null values'''
    df.drop_duplicates()
    df = df.dropna()

    '''Clean x4 column and convert it to float value'''
    valid_v = pd.to_numeric(df['x4'], errors='coerce').notnull()
    df_c = df[valid_v].copy()
    df_c['x4'] = pd.to_numeric(df_c['x4'])

    '''Clean x12 column'''
    valid_i = df_c.x12.isin(['True', 'False'])
    df_c = df_c[valid_i]

    '''Remove outliers'''
    Q1 = np.percentile(df_c.select_dtypes(include=["float"]), 25, interpolation = 'midpoint')
    Q3 = np.percentile(df_c.select_dtypes(include=["float"]), 75, interpolation = 'midpoint')
    IQR = Q3 - Q1
    upper = np.where(df_c.select_dtypes(include=["float"]) >= (Q3 + 1.5 * IQR))
    lower = np.where(df_c.select_dtypes(include=["float"]) <= (Q1 - 1.5 * IQR))
    inliers = ~ (np.isin(df_c.select_dtypes(include=["float"]), upper) & ~ np.isin(df_c.select_dtypes(include=["float"]), lower)).any(axis=1)
    df_c = df_c[inliers]
    
    return df_c

def analysis(df):
    '''Encode y column and calculate correlation'''
    df_encoded = df.copy()
    df_encoded.y = df.y.astype("category").cat.codes
    print(df_encoded.corr())
    "We notice data are quite uncorrelated except for x2 and x6"


if __name__ == "__main__":
    df = load_file(TRAIN_FILE)

    # print(df.head())
    # print(df.shape)
    # print(df.isnull().sum())
    df_clean = cleaning(df)
    # print(dfc.shape)
    # print(dfc.describe(include='all'))
    # print(dfc.dtypes.value_counts())
    # print(dfc['x1'].describe())
    # print(dfc['x12'].unique())
    # print(dfc.describe())
    # print(dfc.shape)
    # print(df_cc.describe())
    # print(df_cc.shape)


