# IEE 520: Fall 2019
# Ensembles
# Klim Drobnyh (klim.drobnyh@asu.edu)

# For compatibility with Python 2
from __future__ import print_function

# To load pandas
import pandas as pd

# To load numpy
import numpy as np

# To import the classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

# To measure accuracy
from sklearn import metrics
from sklearn import model_selection

# To import the scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer

# To display a decision tree
from sklearn.tree import plot_tree

# To support plots
import matplotlib.pyplot as plt

# Set to true to plot confusin matrices
PLOT_CM = False
# You need to install pandas_ml in order to use confusin matrices
# conda install -c conda-forge pandas_ml
from pandas_ml import ConfusionMatrix


class DummyScaler:
    
    def fit(self, data):
        pass
    
    def transform(self, data):
        return data

def create_scaler_dummy():
    return DummyScaler()
    
def create_scaler_standard():
    return StandardScaler()

def create_scaler_minmax():
    return MinMaxScaler()

def create_scaler_binarizer():
    return Binarizer()

# You can choose a scaler (just one should be uncommented):
create_scaler = create_scaler_dummy
# create_scaler = create_scaler_standard
# create_scaler = create_scaler_minmax
# create_scaler = create_scaler_binarizer


def create_model_naive_bayes():
    model = GaussianNB()
    return model

def create_model_mlpclassifier():
    # You can find the full list of parameters here:
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    model = MLPClassifier(hidden_layer_sizes=(10,), 
                          random_state=seed)
    return model

def create_model_svc():
    # You can find the full list of parameters here:
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    model = SVC(random_state=seed, 
                probability=True)
    return model

def create_model_decision_tree():
    # You can find the full list of parameters here:
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    model = DecisionTreeClassifier(min_samples_split=5, 
                                   random_state=seed, 
                                   presort=True)
    return model

def create_model_random_forest():
    # You can find the full list of parameters here:
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    model = RandomForestClassifier(n_estimators=100, 
                                   min_samples_split=5, 
                                   random_state=seed, 
                                   n_jobs=-1)
    return model

def create_model_adaboost():
    # You can find the full list of parameters here:
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=5),
                               learning_rate=0.1,
                               n_estimators=500,
                               random_state=seed)
    return model

def create_model_bagging():
    # You can find the full list of parameters here:
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
    model = BaggingClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=5),
                              bootstrap=False,
                              max_samples=0.6,
                              bootstrap_features=False,
                              max_features=0.6,
                              n_estimators=100,
                              random_state=seed, 
                              n_jobs=-1)
    return model

# You can choose a classifier (just one should be uncommented):
# Naive Bayes:
# create_model = create_model_naive_bayes
# Multi-Layer Perceptron:
# create_model = create_model_mlpclassifier
# Support Vector Classifier:
# create_model = create_model_svc
# Decision Tree:
create_model = create_model_decision_tree

# Ensembles:
# create_model = create_model_random_forest
# create_model = create_model_adaboost
# create_model = create_model_bagging


seed = 520
np.set_printoptions(precision=3)


print('Load the data')

# Here we will use Pen-Based Recognition of Handwritten Digits Data Set.

# This is a quite old dataset (1998), 
# it contains features derived from pen trajectories arising from handwritten digits (0–9) from 44 subjects.

import requests
import os

def download_file(url):
    filename = os.path.basename(url)
    if not os.path.exists(filename):
        response = requests.get(url)
        open(filename, 'wb').write(response.content)
    return filename

from sklearn.datasets import load_svmlight_file
X_train, y_train = load_svmlight_file(download_file('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits'), dtype=np.int32)
y_train = y_train.astype(np.int32)
X_train = X_train.toarray()
X_test, y_test = load_svmlight_file(download_file('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t'), dtype=np.int32)
y_test = y_test.astype(np.int32)
X_test = X_test.toarray()

print('Features:')
print(X_train)


print('Targets:')
print(y_train)


print('Train the model and predict')
scaler = create_scaler()
model = create_model()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train_scaled, y_train)
y_train_hat = model.predict(X_train_scaled)
y_test_hat = model.predict(X_test_scaled)


print('Model evaluation (train):')
print('Accuracy:')
print(metrics.accuracy_score(y_train, y_train_hat))

if PLOT_CM:
    cm = ConfusionMatrix(y_train, y_train_hat)

    ax = cm.plot(backend='seaborn', annot=True, fmt='g')
    ax.set_title('Confusion Matrix (Train)')
    plt.show()


print('Cross-validation')
np.random.seed(seed)
y_cv_hat = np.zeros(y_train.shape)

kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)

# Cross-validation
for cv_train, cv_test in kfold.split(X_train, y_train):
    # Train classifier on training data, predict test data
    
    # Scaling train and test data
    # Train scaler on training set only
    scaler.fit(X_train[cv_train])
    X_cv_train = scaler.transform(X_train[cv_train])
    X_cv_test = scaler.transform(X_train[cv_test])
    
    model = create_model()
    model.fit(X_cv_train, y_train[cv_train])
    y_cv_hat[cv_test] = model.predict(X_cv_test)


print('Model evaluation (CV):')
print('Accuracy:')
print(metrics.accuracy_score(y_train, y_cv_hat))

if PLOT_CM:
    cm = ConfusionMatrix(y_train, y_cv_hat)
    ax = cm.plot(backend='seaborn', annot=True, fmt='g')
    ax.set_title('Confusion Matrix (CV)')
    plt.show()


print('Model evaluation (test):')
print('Accuracy:')
print(metrics.accuracy_score(y_test, y_test_hat))

if PLOT_CM:
    cm = ConfusionMatrix(y_test, y_test_hat)
    ax = cm.plot(backend='seaborn', annot=True, fmt='g')
    ax.set_title('Confusion Matrix (Test)')
    plt.show()


print('Grid Search for Hyperparameters')

# Here we should use specific classifier, because of the hyperparameters
model = model_selection.GridSearchCV(DecisionTreeClassifier(random_state=520),
                                     cv=5,
                                     param_grid={
                                         "max_depth": list(range(1, 40, 2)),
                                         "min_samples_split": list(range(2, 5, 2))
                                     })


model.fit(X_train_scaled, y_train)
print('Optimal parameters:', model.best_params_)
y_test_hat = model.predict(X_test_scaled)

print('Model evaluation (Optimal Hyperparameters):')
print('Accuracy:')
print(metrics.accuracy_score(y_test, y_test_hat))

if PLOT_CM:
    cm = ConfusionMatrix(y_test, y_test_hat)
    ax = cm.plot(backend='seaborn', annot=True, fmt='g')
    ax.set_title('Confusion Matrix (Optimal Hyperparameters)')
    plt.show()

# Not optimal, should search for hyperparameters
model = create_model_random_forest()
model.fit(X_train_scaled, y_train)
y_test_hat = model.predict(X_test_scaled)
print('Model evaluation (Random Forest):')
print('Accuracy:')
print(metrics.accuracy_score(y_test, y_test_hat))
if PLOT_CM:
    cm = ConfusionMatrix(y_test, y_test_hat)
    ax = cm.plot(backend='seaborn', annot=True, fmt='g')
    ax.set_title('Confusion Matrix (Random Forest)')
    plt.show()

# Not optimal, should search for hyperparameters
model = create_model_adaboost()
model.fit(X_train_scaled, y_train)
y_test_hat = model.predict(X_test_scaled)
print('Model evaluation (AdaBoost):')
print('Accuracy:')
print(metrics.accuracy_score(y_test, y_test_hat))
if PLOT_CM:
    cm = ConfusionMatrix(y_test, y_test_hat)
    ax = cm.plot(backend='seaborn', annot=True, fmt='g')
    ax.set_title('Confusion Matrix (AdaBoost)')
    plt.show()

# Not optimal, should search for hyperparameters
model = create_model_bagging()
model.fit(X_train_scaled, y_train)
y_test_hat = model.predict(X_test_scaled)
print('Model evaluation (Bagging):')
print('Accuracy:')
print(metrics.accuracy_score(y_test, y_test_hat))
if PLOT_CM:
    cm = ConfusionMatrix(y_test, y_test_hat)
    ax = cm.plot(backend='seaborn', annot=True, fmt='g')
    ax.set_title('Confusion Matrix (Bagging)')
    plt.show()