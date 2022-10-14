import numpy as np
import pandas as pd

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
    
    return df_c
    

if __name__ == "__main__":
    df = load_file(TRAIN_FILE)

    print(df.head())
    print(df.shape)
    print(df.isnull().sum())
    dfc = cleaning(df)
    print(dfc.shape)
    print(dfc.describe(include='all'))
    print(dfc.dtypes.value_counts())
    print(dfc['x1'].describe())
    # print(dfc['x4'].values)
    print(dfc['x12'].unique())