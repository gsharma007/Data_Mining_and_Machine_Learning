#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:16:33 2019

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

model_tree = GridSearchCV(DecisionTreeClassifier(class_weight = 'balanced', random_state=520),
                          cv=5,
                          param_grid={
                              "max_depth": list(range(1, 100, 2)),
                              "min_samples_split": list(range(2, 200, 2))
                          })
                                     
dt = model_tree.fit(X_train, Y_train)

print('The parameters found by CV search:')
print(model_tree.best_params_)

dt.best_estimator_.fit(X_train, Y_train)

y_pred = dt.best_estimator_.predict(X_test)
print("Accuracy Score DT with removed columns", accuracy_score(y_pred, Y_test))

features = train_data.columns[0:len(train_data.columns)-1]

def show_tree(tr, features, path):
    f = StringIO()
    tree.export_graphviz(tr.best_estimator_, out_file=f, feature_names = features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img = imread(path)
    plt.rcParams["figure.figsize"] = (20,20)
    plt.imshow(img)
    
show_tree(dt, features, 'dec_tree_02.png')

print('Classification report:')
#metrics.classification_report(Y_test, y_pred)
print(metrics.classification_report(Y_test, y_pred))
print("Balanced Accuracy Rate From SKLearn= ", metrics.balanced_accuracy_score(Y_test, y_pred))
print("Balanced Error Rate = ", 1-balanced_accuracy_score(Y_test, y_pred))