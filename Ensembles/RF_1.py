#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:01:13 2019

@author: gauravsharma
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import metrics, tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# To support plots
from ipywidgets import interact
import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from io import StringIO
import pydotplus
from matplotlib.pyplot import imread

import numpy as np
import pandas as pd

# To increase quality of figures
plt.rcParams["figure.figsize"] = (20, 20)

warnings.filterwarnings("ignore") #ignoring python warnings

train_data = pd.read_csv('./train.csv')

#converting -999 to NAN
#train_data.x14[train_data.x14==-999]=np.NaN
#train_data.x15[train_data.x15==-999]=np.NaN

train_data = train_data.drop(["InstanceID","x14","x15"], axis=1)

#defining x and y
x = train_data.iloc[:,0:15]
y = train_data.iloc[:,-1]
print("Sample x data\n", x.head(n=5))
print("Sample y data\n", y.head(n=5))

X_train,X_test,Y_train,Y_test = train_test_split(x, y, test_size=0.30, random_state=0)

#X_train.to_csv(r'/Users/gauravsharma/Downloads/Final_project/train_train.csv')

#model_tree = DecisionTreeClassifier(max_depth=3, 
#                               min_samples_split=116,
#                                       random_state=520)
#                                       # class_weight='balanced')

model_RF = GridSearchCV(RandomForestClassifier(class_weight= 'balanced',random_state=598, max_features='sqrt'),
                            n_jobs=-1,
                            iid=False,
                            param_grid={
                                'n_estimators': (100, 500, 1000),
                                'max_depth': (11, 20, 30),
                                'min_samples_split': (2, 10, 20)
                            })
                                     
dt = model_RF.fit(X_train, Y_train)

print('The parameters for RF found by Grid search:')
print(model_RF.best_params_)


dt.best_estimator_.fit(X_train, Y_train)

y_pred = dt.best_estimator_.predict(X_test)
print("Accuracy Score RF with removed columns", accuracy_score(y_pred, Y_test))

# =============================================================================
# features = train_data.columns[0:len(train_data.columns)-1]
# 
# def show_tree(tr, features, path):
#     f = StringIO()
#     tree.export_graphviz(tr, out_file=f, feature_names = features)
#     pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
#     img = imread(path)
#     plt.rcParams["figure.figsize"] = (20,20)
#     plt.imshow(img)
#     
# show_tree(dt, features, 'dec_tree_02.png')
# =============================================================================

print('Classification report:')
#metrics.classification_report(Y_test, y_pred)
print(metrics.classification_report(Y_test, y_pred))
print("Balanced Accuracy Rate of RF = ", metrics.balanced_accuracy_score(Y_test, y_pred))
print("Balanced Error Rate = ", 1-balanced_accuracy_score(Y_test, y_pred))


"""
Checking validation score and model on the overall data
"""
model_RF_best = RandomForestClassifier(class_weight= 'balanced',random_state=598, max_features='sqrt',
                            n_jobs=-1,
                            iid=False,
                            n_estimators= 500,
                            max_depth= 11,
                            min_samples_split= 20)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_RF_best, x, y, cv=5)
print("Cross validation score for the RF using best parameters : ", scores)

dt_overall = model_RF_best.fit(x,y)
y_pred = model_RF_best.predict(x)
print("Accuracy Score DT with removed columns on overall data", accuracy_score(y_pred, y))

print('Classification report for model on overall data:')
#metrics.classification_report(Y_test, y_pred)
print(metrics.classification_report(y, y_pred))
print("Balanced Accuracy Rate of RF on overall data= ", metrics.balanced_accuracy_score(y, y_pred))
print("Balanced Error Rate of RF on overall data= ", 1-balanced_accuracy_score(y, y_pred))