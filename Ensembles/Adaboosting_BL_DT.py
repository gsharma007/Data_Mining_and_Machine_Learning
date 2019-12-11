#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 14:38:56 2019

@author: gauravsharma
"""

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
#from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
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

model_AB = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3, min_samples_split=116),
    n_estimators=2000
)
                                     
ab = model_AB.fit(X_train, Y_train)

y_pred = ab.predict(X_test)
print("Accuracy Score AB with removed columns", accuracy_score(y_pred, Y_test))

print('Classification report of AB:')
print(metrics.classification_report(Y_test, y_pred))
print("Balanced Accuracy Rate of AB = ", metrics.balanced_accuracy_score(Y_test, y_pred))
print("Balanced Error Rate of AB= ", 1-balanced_accuracy_score(Y_test, y_pred))

"""
Checking validation score and model on the overall data
"""

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_AB, x, y, cv=5)
print("Cross validation score for the AB using best parameters : ", scores)

ab_overall = model_AB.fit(x,y)
y_pred = ab_overall.predict(x)
print("Accuracy Score AB with removed columns on overall data", accuracy_score(y_pred, y))

print('Classification report for model AB on overall data:')
#metrics.classification_report(Y_test, y_pred)
print(metrics.classification_report(y, y_pred))
print("Balanced Accuracy Rate of AB on overall data= ", metrics.balanced_accuracy_score(y, y_pred))
print("Balanced Error Rate of AB on overall data= ", 1-balanced_accuracy_score(y, y_pred))