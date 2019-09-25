#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 23:30:06 2019

@author: gauravsharma
"""
import numpy as np
import pandas as pd
from pandas_ml import ConfusionMatrix

#from sklearn.model_selection import train_test_split 
#from sklearn import preprocessing
#from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import Binarizer

bank_data = pd.read_csv('/Users/gauravsharma/Documents/bank-additional-full.csv')

# Checking missing values
bank_data.isnull().values.any()

bank_data["y"] = bank_data["y"].astype('category')
bank_data["y"] = bank_data["y"].cat.codes

target = bank_data["y"]

#X1 = {"job": {"housemaid": 0, "services": 1, "admin.": 2, "blue-collar": 3, "technician": 4, 
    # "retired": 5, "management": 6, "unemployed":7, "self-employed": 8, "unknown": 9, "entrepreneur": 10, 
     # "student": 11}}
#bank_data["job"].replace(X1, inplace= True)
     
SCALE = False

# You can choose a different scaler
# SCALER_CLASS = StandardScaler
# SCALER_CLASS = MinMaxScaler
# SCALER_CLASS = Binarizer

bank_data["job"] = bank_data["job"].astype('category')
bank_data["job"] = bank_data["job"].cat.codes

bank_data["marital"] = bank_data["marital"].astype('category')
bank_data["marital"] = bank_data["marital"].cat.codes

bank_data["education"] = bank_data["education"].astype('category')
bank_data["education"] = bank_data["education"].cat.codes

bank_data["default"] = bank_data["default"].astype('category')
bank_data["default"] = bank_data["default"].cat.codes

bank_data["housing"] = bank_data["housing"].astype('category')
bank_data["housing"] = bank_data["housing"].cat.codes

bank_data["loan"] = bank_data["loan"].astype('category')
bank_data["loan"] = bank_data["loan"].cat.codes

bank_data["contact"] = bank_data["contact"].astype('category')
bank_data["contact"] = bank_data["contact"].cat.codes

bank_data["month"] = bank_data["month"].astype('category')
bank_data["month"] = bank_data["month"].cat.codes

bank_data["day_of_week"] = bank_data["day_of_week"].astype('category')
bank_data["day_of_week"] = bank_data["day_of_week"].cat.codes

bank_data["poutcome"] = bank_data["poutcome"].astype('category')
bank_data["poutcome"] = bank_data["poutcome"].cat.codes

X = bank_data.drop("y", axis=1)

model = GaussianNB()
model.fit(X, target)
yhat = model.predict(X)

print('Accuracy:')
print(metrics.accuracy_score(target, yhat))
print('Classification report:')
print(metrics.classification_report(target, yhat))

print('Confusion matrix:')
cm = ConfusionMatrix(target, yhat)
print(cm)
print('Stats:')
cm.print_stats()
ax = cm.plot(backend='seaborn', annot=True, fmt='g')
ax.set_title('Confusion Matrix')
plt.show()
plt.clf()

fpr, tpr, threshold = metrics.roc_curve(target, yhat)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



seed = 3421
np.random.seed(seed)
actuals = []
probs = []
hats = []

X = X.values

kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in kfold.split(X, target):
    Xtrain = X[train]        
    Xtest = X[test]
    model.fit(Xtrain, target[train])
    foldhats = model.predict(Xtest)
    foldprobs = model.predict_proba(Xtest)[:,1]
    actuals = np.append(actuals, target[test])
    probs = np.append(probs, foldprobs)    
    hats = np.append(hats, foldhats)

print('Accuracy:')
print(metrics.accuracy_score(actuals, hats))
print('Classification report:')
print(metrics.classification_report(actuals, hats))  
        
cm_cv = ConfusionMatrix(actuals, hats)
print(cm_cv)
cm_cv.print_stats()
ax = cm_cv.plot(backend='seaborn', annot=True, fmt='g')
ax.set_title('Test Confusion Matrix for CV')
plt.show()
plt.clf()


fpr, tpr, threshold = metrics.roc_curve(actuals, probs)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic_CV')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
