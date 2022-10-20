from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier

from skopt import BayesSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis




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
                                          ('pca', PCA(n_components=None))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_features), 
                                    ('cat', categorical_transformer, cat_features)])
    
    return preprocessor



def test_classifiers(preprocessor, XY_t):
    X_t, Y_t = XY_t

    cv = StratifiedKFold(n_splits=5, shuffle=True)

    '''Find the best parameters for the random forest classifier'''
    '''Random Forest'''
    forest = Pipeline(steps=[('preprocessor', preprocessor),
                            ('standardscaler', StandardScaler()),
                           ('forest', RandomForestClassifier(random_state=900, n_estimators=975, min_samples_split=5, 
                                            min_samples_leaf=1, max_features='sqrt', max_depth=40, bootstrap=False))])
    print("Random Forest")
    print(np.average(cross_val_score(forest, X_t, Y_t, cv=cv)))
    

    # params_forest = { 
    # 'forest__bootstrap': [True, False],
    # 'forest__max_depth': [10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, None],
    # 'forest__max_features': ['sqrt', 'log2'],
    # 'forest__min_samples_leaf': [1, 2, 3, 4],
    # 'forest__min_samples_split': [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    # 'forest__n_estimators': [200, 600, 800, 850, 875, 900, 950, 975, 1000, 1100, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
    # }

    # forest_search = RandomizedSearchCV(forest, param_distributions=params_forest, n_iter=240, verbose=1, n_jobs=-1, cv=cv)
    # forest_search.fit(X_t, Y_t)
    # print(forest_search.best_params_)
    # print(forest_search.best_score_)

    '''SVM'''
    svc = Pipeline(steps=[('preprocessor', preprocessor),
                            ('standardscaler', StandardScaler()),
                           ('svc', SVC(random_state=R, kernel='rbf', C=1.0, gamma='auto', probability=True))])

    print("SVM")
    print(np.average(cross_val_score(svc, X_t, Y_t, cv=cv)))

    '''Gradient Boosting'''
    gradient = Pipeline(steps=[('preprocessor', preprocessor),
                            ('standardscaler', StandardScaler()),
                            ('gb', GradientBoostingClassifier(random_state=R, n_estimators=75, max_depth=5, learning_rate=0.2))])
    print("Gradient Boosting")
    print(np.average(cross_val_score(gradient, X_t, Y_t, cv=cv)))

    '''Extremely Random Forest'''
    extreme_forest = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()),
                                    ('extreme', ExtraTreesClassifier(random_state=R, n_estimators=950, min_samples_split=2, 
                                                                        min_samples_leaf=1, max_features='log2', max_depth=20, bootstrap=False))])
    print("Extremely Random Forest")
    print(np.average(cross_val_score(extreme_forest, X_t, Y_t, cv=cv)))
    # params_extreme = { 
    #     'extreme__bootstrap': [True, False],
    #     'extreme__max_depth': [10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, None],
    #     'extreme__max_features': ['sqrt', 'log2'],
    #     'extreme__min_samples_leaf': [1, 2, 3, 4],
    #     'extreme__min_samples_split': [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    #     'extreme__n_estimators': [200, 600, 800, 850, 875, 900, 950, 975, 1000, 1100, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
    #     'extreme__random_state': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, None],
    # }

    # extreme_search = RandomizedSearchCV(extreme_forest, param_distributions=params_extreme, n_iter=240, verbose=1, n_jobs=-1, cv=cv)
    # extreme_search.fit(X_t, Y_t)
    # print(extreme_search.best_params_)
    # print(extreme_search.best_score_)


    '''Bayes'''
    bayes = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()), ('bayes', GaussianNB())])
    print("Bayes")
    print(np.average(cross_val_score(bayes, X_t, Y_t, cv=cv)))

    '''KNN'''
    knn = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()), 
                            ('knn', KNeighborsClassifier(weights='uniform', n_neighbors=10, leaf_size=40, algorithm='ball_tree'))])
    print("knn")
    print(np.average(cross_val_score(knn, X_t, Y_t, cv=cv)))

    '''Adaboost'''
    adaboost = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()), 
                                ('adaboost', AdaBoostClassifier(random_state=R, n_estimators=400, learning_rate=0.1, algorithm='SAMME.R'))])
    print("adaboost")
    print(np.average(cross_val_score(adaboost, X_t, Y_t, cv=cv)))


    '''Ensemble'''
    forest_e = RandomForestClassifier(random_state=R, n_estimators=1600, min_samples_split=11, 
                                        min_samples_leaf=1, max_features='log2', max_depth=100, bootstrap=True)
    svc_e = SVC(random_state=R, kernel='rbf', C=1.0, gamma='auto', probability=True)
    gradient_e = GradientBoostingClassifier(random_state=R, n_estimators=75, max_depth=5, learning_rate=0.2)
    extreme_forest_e = ExtraTreesClassifier(random_state=R, n_estimators=950, min_samples_split=2, 
                                            min_samples_leaf=1, max_features='log2', max_depth=20, bootstrap=False)
    extreme_forest_e2 = ExtraTreesClassifier(random_state=900, n_estimators=975, min_samples_split=5, 
                                            min_samples_leaf=1, max_features='sqrt', max_depth=40, bootstrap=False)

    estimators = [('forest', forest_e), ('svc', svc_e), ('gradient', gradient_e), ('extreme', extreme_forest_e2)]

    ensemble = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()), 
                                ('ens', VotingClassifier(estimators=estimators, voting='soft'))])

    print("Ensemble")
    print(np.average(cross_val_score(ensemble, X_t, Y_t, cv=cv)))

    '''Ensemble 2'''
    classifiers = [('qda', QuadraticDiscriminantAnalysis()), ('lda', LinearDiscriminantAnalysis()), ('rf', forest_e)] 
    model = StackingClassifier(classifiers, final_estimator=LinearDiscriminantAnalysis(), cv=cv)
    print("Ensemble 2")
    print(np.average(cross_val_score(model, X_t, Y_t, cv=cv)))

    '''Ensemble 3'''
    extreme_forest_f1 = ExtraTreesClassifier(random_state=R, n_estimators=950, min_samples_split=2, 
                                            min_samples_leaf=1, max_features='log2', max_depth=20, bootstrap=False)
    extreme_forest_f2 = ExtraTreesClassifier(random_state=900, n_estimators=975, min_samples_split=5, 
                                            min_samples_leaf=1, max_features='sqrt', max_depth=40, bootstrap=False)
    extreme_forest_f3 = ExtraTreesClassifier(random_state=1000, n_estimators=875, min_samples_split=10, min_samples_leaf=1, 
                                            max_features='sqrt', max_depth=40, bootstrap=False)
    extreme_forest_f4 = ExtraTreesClassifier(random_state=950, n_estimators=1000, min_samples_split=20, min_samples_leaf=1, 
                                            max_features='log2', max_depth=50, bootstrap=False)
    classifiers_f = [('rf1', extreme_forest_f1), ('rf2', extreme_forest_f2), ('rf3', extreme_forest_f3), ('rf4', extreme_forest_f4)]
    ensemble3 = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()),
                        ('ens', VotingClassifier(estimators=classifiers_f, voting='soft'))])
    print("Ensemble 3")
    print(np.average(cross_val_score(ensemble3, X_t, Y_t, cv=cv)))


    '''Test classifiers over different splits value'''
    return extreme_forest
    i = 0
    extreme_acc = []
    ensemble_acc = []
    ensemble3_acc = []
    while i < 100:
        X_t, Y_t = shuffle(X_t, Y_t)
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        print(i)

        # extreme = Pipeline(steps=[('preprocessor', preprocessor),
        #                     ('standardscaler', StandardScaler()),
        #                     ('gb', ExtraTreesClassifier(random_state=R, n_estimators=950, min_samples_split=2, 
        #                                                 min_samples_leaf=1, max_features='log2', max_depth=20, bootstrap=False))])
        extreme = Pipeline(steps=[('preprocessor', preprocessor),
                            ('standardscaler', StandardScaler()),
                            ('gb', ExtraTreesClassifier(random_state=900, n_estimators=975, min_samples_split=5, 
                                                        min_samples_leaf=1, max_features='sqrt', max_depth=40, bootstrap=False))])
        acc_e = np.average(cross_val_score(extreme, X_t, Y_t, cv=cv))
        extreme_acc.append(acc_e)
        print("Extreme")
        print(acc_e)

        ensemble = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()), 
                                    ('ens', VotingClassifier(estimators=estimators, voting='soft'))])
        acc_en = np.average(cross_val_score(ensemble, X_t, Y_t, cv=cv))
        ensemble_acc.append(acc_en)
        print("Ensemble")
        print(acc_en)

        ensemble3 = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()),
                                    ('ens', VotingClassifier(estimators=classifiers_f, voting='soft'))])
        acc_en3 = np.average(cross_val_score(ensemble3, X_t, Y_t, cv=cv))
        ensemble3_acc.append(acc_en3)
        print("Ensemble 3")
        print(acc_en3)

        i+=1
    
    print("Extreme, Ensemble", "Ensemble 3")
    print(np.mean(extreme_acc), np.mean(ensemble_acc), np.mean(ensemble3_acc))

    return extreme

        


def test_model(model, X, Y, file):
    cv_test = StratifiedKFold(n_splits=5, random_state=(R + 1), shuffle=True)

    np.average(cross_val_score(model, X, Y, cv=cv_test))

    y_pred = cross_val_predict(model, X, Y, cv=cv_test, n_jobs=-1)
    print(classification_report(Y, y_pred))

    estimator = model
    estimator.fit(X, Y)

    df_eval = pd.read_csv(file, index_col=0)
    predictions = estimator.predict(df_eval)

    OUT_FILE = 'programming_challenge/resources/labels.txt'

    with open(OUT_FILE, 'w') as f:
        for prediction in predictions:
            f.write(prediction + linesep)




