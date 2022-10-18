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
                                          ('pca', PCA(n_components=8))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_features), 
                                    ('cat', categorical_transformer, cat_features)])
    
    return preprocessor



def test_classifiers(preprocessor, XY_t):
    X_t, Y_t = XY_t

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=R)

    '''Find the best parameters for the random forest classifier'''
    '''Random Forest'''
    forest = Pipeline(steps=[('preprocessor', preprocessor),
                            ('standardscaler', StandardScaler()),
                           ('forest', RandomForestClassifier(random_state=R, n_estimators=900, min_samples_split=10, 
                           min_samples_leaf=2, max_features='sqrt', max_depth=45, bootstrap=True))])
    forest.fit(X_t, Y_t)
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

    # max_depth = list(range(1,101, 2))
    # min_samples_l = list(range(1,21, 2))
    # min_samples_s = list(range(2,51, 2))
    # estim = list(range(100, 5000, 20))

    # params_b = {
    # 'forest__bootstrap': [True, False],
    # 'forest__max_depth': max_depth,
    # 'forest__max_features': ['sqrt', 'log2'],
    # 'forest__min_samples_leaf': min_samples_l,
    # 'forest__min_samples_split': min_samples_s,
    # 'forest__n_estimators': estim,
    # }

    # forest_search_b = BayesSearchCV(pipeline, search_spaces=params_b, n_iter=240, verbose=1, n_jobs=-1, cv=cv)
    # forest_search_b.fit(X_t, Y_t)
    # print(forest_search_b.best_params_)
    # print(forest_search_b.best_score_)

    '''SVM'''
    svc = Pipeline(steps=[('preprocessor', preprocessor),
                            ('standardscaler', StandardScaler()),
                           ('svc', SVC(random_state=R, kernel='rbf', C=1.0, gamma='auto', probability=True))])

    svc.fit(X_t, Y_t)
    print("SVM")
    print(np.average(cross_val_score(svc, X_t, Y_t, cv=cv)))

    '''Gradient Boosting'''
    gradient = Pipeline(steps=[('preprocessor', preprocessor),
                            ('standardscaler', StandardScaler()),
                            ('gb', GradientBoostingClassifier(random_state=R, n_estimators=50, max_depth=5, learning_rate=0.075))])
    gradient.fit(X_t, Y_t)
    print("Gradient Boosting")
    print(np.average(cross_val_score(gradient, X_t, Y_t, cv=cv)))

    '''Extremely Random Forest'''
    extreme_forest = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()),
                                    ('extreme', ExtraTreesClassifier(random_state=R, n_estimators=875, min_samples_split=6, 
                                                                        min_samples_leaf=1, max_features='log2', max_depth=55, bootstrap=False))])
    extreme_forest.fit(X_t, Y_t)
    print("Extremely Random Forest")
    print(np.average(cross_val_score(extreme_forest, X_t, Y_t, cv=cv)))
    # params_extreme = { 
    #     'extreme__bootstrap': [True, False],
    #     'extreme__max_depth': [10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, None],
    #     'extreme__max_features': ['sqrt', 'log2'],
    #     'extreme__min_samples_leaf': [1, 2, 3, 4],
    #     'extreme__min_samples_split': [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    #     'extreme__n_estimators': [200, 600, 800, 850, 875, 900, 950, 975, 1000, 1100, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
    # }

    # extreme_search = RandomizedSearchCV(extreme_forest, param_distributions=params_extreme, n_iter=240, verbose=1, n_jobs=-1, cv=cv)
    # extreme_search.fit(X_t, Y_t)
    # print(extreme_search.best_params_)
    # print(extreme_search.best_score_)

    # '''Bagging'''
    # bagging = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()),
    #                             ('bagging', BaggingClassifier(random_state=R))])
    # params_bagging = {
    #     'bagging__bootstrap': [True, False],
    #     'bagging__max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     'bagging__n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    #     'bagging__max_samples': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # }

    # bagging_search = RandomizedSearchCV(bagging, param_distributions=params_bagging, n_iter=240, verbose=1, n_jobs=-1, cv=cv)
    # bagging_search.fit(X_t, Y_t)
    # print(bagging_search.best_params_)
    # print(bagging_search.best_score_)

    '''Bayes'''
    bayes = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()), ('bayes', GaussianNB())])
    bayes.fit(X_t, Y_t)
    print("Bayes")
    print(np.average(cross_val_score(bayes, X_t, Y_t, cv=cv)))


    # params_gradient = {
    #     "gb__n_estimators" : [50, 75, 100, 150, 200, 250, 300, 400, 500],
    #     "gb__max_depth" : [5, 7, 10, 15, 20, None],
    #     "gb__learning_rate" : [0.1, 0.2, 0.01, 0.05, 0.075],
    # }

    # gradient_search = RandomizedSearchCV(gradient, param_distributions=params_gradient, n_iter=120, verbose=1, n_jobs=-1, cv=cv)
    # gradient_search.fit(X_t, Y_t)
    # print(gradient_search.best_params_)
    # print(gradient_search.best_score_)

    '''KNN'''
    knn = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()), 
                            ('knn', KNeighborsClassifier(weights='uniform', n_neighbors=10, leaf_size=40, algorithm='ball_tree'))])
    knn.fit(X_t, Y_t)
    print("knn")
    print(np.average(cross_val_score(knn, X_t, Y_t, cv=cv)))

    '''Adaboost'''
    adaboost = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()), 
                                ('adaboost', AdaBoostClassifier(random_state=R, n_estimators=400, learning_rate=0.1, algorithm='SAMME.R'))])
    adaboost.fit(X_t, Y_t)
    print("adaboost")
    print(np.average(cross_val_score(adaboost, X_t, Y_t, cv=cv)))


    '''Ensemble'''
    forest_e = RandomForestClassifier(random_state=R, n_estimators=900, min_samples_split=10, 
                           min_samples_leaf=2, max_features='sqrt', max_depth=45, bootstrap=True)
    svc_e = SVC(random_state=R, kernel='rbf', C=1.0, gamma='auto', probability=True)
    gradient_e = GradientBoostingClassifier(random_state=R, n_estimators=50, max_depth=5, learning_rate=0.075)
    bayes_e = GaussianNB()
    extreme_forest_e = ExtraTreesClassifier(random_state=R, n_estimators=875, min_samples_split=6, min_samples_leaf=1, max_features='log2', max_depth=55, bootstrap=False)
    bagging_e = BaggingClassifier(random_state=R, n_estimators=90, max_samples=9, max_features=7, bootstrap=False)

    estimators = [('forest', forest_e), ('svc', svc_e), ('gradient', gradient_e), ('bayes', bayes_e), ('extreme', extreme_forest_e), ('bagging', bagging_e)]

    ensemble = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()), 
                                ('ens', VotingClassifier(estimators=estimators, voting='soft'))])

    ensemble.fit(X_t, Y_t)
    print("Ensemble")
    print(np.average(cross_val_score(ensemble, X_t, Y_t, cv=cv)))

    '''Ensemble 2'''
    classifiers = [('qda', QuadraticDiscriminantAnalysis()), ('lda', LinearDiscriminantAnalysis()), ('rf', forest_e)] 
    model = StackingClassifier(classifiers, final_estimator=LinearDiscriminantAnalysis(), cv=cv)
    model.fit(X_t, Y_t)
    print("Ensemble 2")
    print(np.average(cross_val_score(model, X_t, Y_t, cv=cv)))


    '''Test classifiers over different splits value'''
    i = 0
    while i < 300:
        if (i <=50):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=R)
            print(i)
        if (i <= 100 and i > 50):
            cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=R)
            print(i)
        if (i <= 150 and i > 100):
            cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=R)
            print(i)
        if (i <= 200 and i > 150):
            cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=R)
            print(i)
        if (i <= 250 and i > 200):
            cv = StratifiedKFold(n_splits=9, shuffle=True, random_state=R)
            print(i)
        if (i <= 300 and i > 250):
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=R)
            print(i)
        
        forest = Pipeline(steps=[('preprocessor', preprocessor),
                            ('standardscaler', StandardScaler()),
                           ('forest', RandomForestClassifier(random_state=R, n_estimators=900, min_samples_split=10, 
                           min_samples_leaf=2, max_features='sqrt', max_depth=45, bootstrap=True))])
        forest_acc = []
        forest.fit(X_t, Y_t)
        forest_acc.append(np.average(cross_val_score(forest, X_t, Y_t, cv=cv)))

        svc = Pipeline(steps=[('preprocessor', preprocessor),
                            ('standardscaler', StandardScaler()),
                           ('svc', SVC(random_state=R, kernel='rbf', C=1.0, gamma='auto', probability=True))])
        svc_acc = []
        svc.fit(X_t, Y_t)
        svc_acc.append(np.average(cross_val_score(svc, X_t, Y_t, cv=cv)))

        extreme = Pipeline(steps=[('preprocessor', preprocessor),
                            ('standardscaler', StandardScaler()),
                            ('gb', ExtraTreesClassifier(random_state=R, n_estimators=875, min_samples_split=6, min_samples_leaf=1, max_features='log2', max_depth=55, bootstrap=False))])
        extreme_acc = []
        extreme.fit(X_t, Y_t)
        extreme_acc.append(np.average(cross_val_score(gradient, X_t, Y_t, cv=cv)))

        ensemble = Pipeline(steps=[('preprocessor', preprocessor), ('standardscaler', StandardScaler()), 
                                ('ens', VotingClassifier(estimators=estimators, voting='soft'))])
        ensemble_acc = []
        ensemble.fit(X_t, Y_t)
        ensemble_acc.append(np.average(cross_val_score(ensemble, X_t, Y_t, cv=cv)))

        i+=1
    
    print(np.mean(forest_acc), np.mean(svc_acc), np.mean(extreme_acc), np.mean(ensemble_acc))

        


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




