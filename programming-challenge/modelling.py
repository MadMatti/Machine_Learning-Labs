from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

import numpy as np
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
                                              ('encoder', OrdinalEncoder())])
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('pca', PCA(n_components=8))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_features), ('cat', categorical_transformer, cat_features)])
    
    return preprocessor

classifiers = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000, random_state=R, max_depth=1000),
    "Gradient Boosting": GradientBoostingClassifier(random_state=R),
    "Extreme random forest": ExtraTreesClassifier(random_state=R),
    "AdaBoost": AdaBoostClassifier(),
    "Bagging": BaggingClassifier(random_state=R),
    "SVM": SVC(),
    "Ridge Classifier": RidgeClassifierCV(),
    "Naive Bayes": GaussianNB(),
}

def test_classifiers(preprocessor, XY_t):
    X_t, Y_t = XY_t
    best_cls = None
    name_best_cls = ""
    best_accuracy = 0

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=R)

    for cls_name, cls in classifiers.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', cls)])
        cv_accuracies = np.average(cross_val_score(pipeline, X_t, Y_t, cv=cv))

        print(f"{cls_name} accuracy: {cv_accuracies}")

        if cv_accuracies > best_accuracy:
            best_accuracy = cv_accuracies
            best_cls = pipeline
            name_best_cls = cls_name

    print(f"Best classifier: {name_best_cls} with accuracy: {best_accuracy}")


