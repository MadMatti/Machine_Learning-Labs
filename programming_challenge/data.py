import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


def load_file(file_name):
    """Load a file into a pandas dataframe"""
    return pd.read_csv(file_name, index_col=0)

def cleaning(df):
    '''Drop duplicates and null values'''
    df.drop_duplicates()
    df = df.dropna()
    "Drop non numerical columns"
    df = df.drop('x7', axis=1)
    df = df.drop('x12', axis=1)

    '''Clean x4 column and convert it to float value'''
    "x4 column contains float number, but it is encoded as object values"
    valid_v = pd.to_numeric(df['x4'], errors='coerce').notnull()
    df_c = df[valid_v].copy()
    df_c['x4'] = pd.to_numeric(df_c['x4'])

    '''Remove outliers'''
    "I notice x2 and x4 may have outliers, so I remove them"
    zscore = np.abs(stats.zscore(df_c.select_dtypes(include=["float"])))
    is_inlier = ~ (zscore > 4).any(axis=1)
    df_c = df_c[is_inlier]

    return df_c

def analysis(df):
    '''Encode y column and calculate correlation'''
    df_encoded = df.copy()
    df_encoded.y = df.y.astype("category").cat.codes
    print(df_encoded.corr())
    "I notice data are quite uncorrelated except for x2 and x6"

