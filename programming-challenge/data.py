import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_file(file_name):
    """Load a file into a pandas dataframe"""
    return pd.read_csv(file_name, index_col=0)

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
    df = load_file("programming-challenge/resources/TrainOnMe-4.csv")
    print(df.head())




