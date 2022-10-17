from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

from os import linesep

import numpy as np
import pandas as pd
R = 42

def split(df):
    X = df.drop('y', axis=1)
    Y = df.y
    X_train, Y_train = shuffle(X, Y, random_state=R)
    return X_train, Y_train

def transform(XY):
    X, Y = XY
    num_features = X.select_dtypes(include=['float64']).columns
    cat_features = X.select_dtypes(include=['object', 'bool']).columns
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value="missing")),
                                              ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('pca', PCA(n_components=8))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_features), 
                                    ('cat', categorical_transformer, cat_features)])
    
    return preprocessor



def test_classifiers(preprocessor, XY_t):
    X_t, Y_t = XY_t

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=R)

    '''Find the best parameters for the random forest classifier'''
    '''Random Forest'''
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('standardscaler', StandardScaler()),
                           ('forest', RandomForestClassifier(random_state=R))])

    params = { 
    'forest__bootstrap': [True, False],
    'forest__max_depth': [10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, None],
    'forest__max_features': ['sqrt', 'log2'],
    'forest__min_samples_leaf': [1, 2, 3, 4],
    'forest__min_samples_split': [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'forest__n_estimators': [200, 600, 800, 850, 875, 900, 950, 975, 1000, 1100, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
    }

    forest_search = RandomizedSearchCV(pipeline, param_distributions=params, n_iter=240, verbose=1, n_jobs=-1, cv=cv)
    forest_search.fit(X_t, Y_t)
    print(forest_search.best_params_)
    print(forest_search.best_score_)

    return forest_search

def test_model(model, X, Y, file):
    cv_test = StratifiedKFold(n_splits=5, random_state=(R + 1), shuffle=True)

    np.average(cross_val_score(model.best_estimator_, X, Y, cv=cv_test))

    y_pred = cross_val_predict(model.best_estimator_, X, Y, cv=cv_test, n_jobs=-1)
    print(classification_report(Y, y_pred))

    estimator = model.best_estimator_
    estimator.fit(X, Y)

    df_eval = pd.read_csv(file, index_col=0)

    predictions = estimator.predict(df_eval)

    OUT_FILE = 'programming_challenge/resources/labels.txt'

    with open(OUT_FILE, 'w') as f:
        for prediction in predictions:
            f.write(prediction + linesep)




