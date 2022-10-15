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
    numeric_col = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x8', 'x9', 'x10', 'x11', 'x13']
    cateorical_cal = ['x7', 'x12']
    df_c.boxplot(numeric_col)
    plt.show()
    
    return df_c

def analysis(df):
    '''Encode y column and calculate correlation'''
    df_encoded = df.copy()
    df_encoded.y = df.y.astype("category").cat.codes
    print(df_encoded.corr())
    "We notice data are quite uncorrelated except for x2 and x6"

if __name__ == "__main__":
    df = load_file("programming-challenge/resources/TrainOnMe-4.csv")
    cleaning(df)
    print(df.head())




