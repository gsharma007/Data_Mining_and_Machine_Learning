#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:27:34 2019

@author: gauravsharma
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from sklearn import metrics
#from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC

svc=SVC(probability=True, kernel='linear')

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

model_AB = AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)
                                     
ab = model_AB.fit(X_train, Y_train)

y_pred = ab.predict(X_test)
print("Accuracy Score AB with SVC", accuracy_score(y_pred, Y_test))

print('Classification report of AB with SVC:')
print(metrics.classification_report(Y_test, y_pred))
print("Balanced Accuracy Rate of AB with SVC= ", metrics.balanced_accuracy_score(Y_test, y_pred))
print("Balanced Error Rate of AB with SVC= ", 1-balanced_accuracy_score(Y_test, y_pred))

"""
Checking validation score of model AB with SVC on the overall data
"""

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_AB, x, y, cv=5)
print("Cross validation score for the AB using SVC : ", scores)

ab_overall = model_AB.fit(x,y)
y_pred = ab_overall.predict(x)
print("Accuracy Score AB with SVC on overall data", accuracy_score(y_pred, y))

print('Classification report for model AB with SVC on overall data:')
#metrics.classification_report(Y_test, y_pred)
print(metrics.classification_report(y, y_pred))
print("Balanced Accuracy Rate of AB with SVC on overall data= ", metrics.balanced_accuracy_score(y, y_pred))
print("Balanced Error Rate of AB with SVC on overall data= ", 1-balanced_accuracy_score(y, y_pred))