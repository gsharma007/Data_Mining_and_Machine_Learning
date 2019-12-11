#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:40:19 2019

@author: gauravsharma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools
import warnings
from sklearn.preprocessing import StandardScaler

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

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2, random_state = 123)


"""
Fitting linear model
"""

from sklearn.svm import LinearSVC
#Maximum Iteration is set to 10000 to avoid Convergence Warning 
SVM = LinearSVC(random_state = 123, max_iter=10000)
#SVM = LinearSVC(random_state = 1234)
SVM_Linear_fit = SVM.fit(x_train,y_train)

print("Train data accuracy in Linear kernel SVM", SVM.score(x_train, y_train))
print("Test data accuracy in Linear kernel SVM", SVM.score(x_test, y_test))

y_pred_Linear = SVM_Linear_fit.predict(x_test)

print("Accuracy Score Linear SVC", accuracy_score(y_pred_Linear, y_test))
print('Classification report of Linear SVC:')
print(metrics.classification_report(y_test, y_pred_Linear))
print("Balanced Accuracy Rate of Linear SVC= ", metrics.balanced_accuracy_score(y_test, y_pred_Linear))
print("Balanced Error Rate of Linear SVC= ", 1-balanced_accuracy_score(y_test, y_pred_Linear))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(SVM_Linear_fit, x_train, y_train, cv=5)
print("Cross validation score for the Linear SVC : ", scores)

"""
Fitting polynomial model
"""
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc_grid = GridSearchCV(svc, {'gamma':[0.00001,0.001,0.01,0.1,1,10,100], 'C':[0.01,0.1,1,10,100,1000]}, return_train_score=True)
svc_grid.fit(x_train, y_train)
SVM_Poly_fit = svc_grid.best_estimator_
print("Best estimator is:", svc_grid.best_params_)

y_pred_Poly = SVM_Poly_fit.predict(x_test)

print("Train data accuracy in polynomial kernel SVM", SVM_Poly_fit.score(x_train, y_train))
print("Test data accuracy in polynomial kernel SVM",SVM_Poly_fit.score(x_test, y_test))

print("Accuracy Score Poly SVCC", accuracy_score(y_pred_Poly, y_test))
print('Classification report of Poly SVC:')
print(metrics.classification_report(y_test, y_pred_Poly))
print("Balanced Accuracy Rate of Poly SVC= ", metrics.balanced_accuracy_score(y_test, y_pred_Poly))
print("Balanced Error Rate of Poly SVC= ", 1-balanced_accuracy_score(y_test, y_pred_Poly))

scores = cross_val_score(SVM_Poly_fit, x_train, y_train, cv=5)
print("Cross validation score for the Poly SVC : ", scores)

"""
Fitting Linear model on overall data
"""

SVM = LinearSVC(random_state = 123, max_iter=10000)
#SVM = LinearSVC(random_state = 1234)
SVM_Linear_fit_O = SVM.fit(x,y)

print("Overall data accuracy in Linear kernel SVM", SVM.score(x, y))

y_pred_OO_L = SVM_Linear_fit_O.predict(x)

print("Accuracy Score Overall Linear SVC", accuracy_score(y_pred_OO_L, y))
print('Classification report of Overall Linear SVC:')
print(metrics.classification_report(y, y_pred_OO_L))
print("Balanced Accuracy Rate of Overall Linear SVC= ", metrics.balanced_accuracy_score(y, y_pred_OO_L))
print("Balanced Error Rate of Overall Linear SVC= ", 1-balanced_accuracy_score(y, y_pred_OO_L))

scores = cross_val_score(SVM_Linear_fit_O, x, y, cv=5)
print("Cross validation score for the Overall Linear SVC : ", scores)

"""
Fitting Polynomial model on overall data
"""

svc = SVC(kernel='rbf')
svc_grid = GridSearchCV(svc, {'gamma':[0.00001,0.001,0.01,0.1,1,10,100], 'C':[0.01,0.1,1,10,100,1000]}, return_train_score=True)
svc_grid.fit(x, y)
SVM_OO_P_fit = svc_grid.best_estimator_
print("Best estimator for overall data is:", svc_grid.best_params_)

y_pred_OO_P = SVM_OO_P_fit.predict(x)

print("Overall data accuracy in polynomial kernel SVM", SVM_Poly_fit.score(x, y))

print("Accuracy Score Overall Poly SVCC", accuracy_score(y_pred_OO_P, y))
print('Classification report of Overall Poly SVC:')
print(metrics.classification_report(y_pred_OO_P, y))
print("Balanced Accuracy Rate of Overall Poly SVC= ", metrics.balanced_accuracy_score(y_pred_OO_P, y))
print("Balanced Error Rate of Overall Poly SVC= ", 1-balanced_accuracy_score(y_pred_OO_P, y))

scores = cross_val_score(SVM_OO_P_fit, x, y, cv=5)
print("Cross validation score for the Overall Poly SVC : ", scores)