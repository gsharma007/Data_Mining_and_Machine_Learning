#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:38:44 2019

@author: gauravsharma
"""

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
#from sklearn.naive_bayes import GaussianNB 
from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import Perceptron

#from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
from pandas_ml import ConfusionMatrix
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler


mydata = datasets.load_breast_cancer()
mydata.DESCR
X = mydata.data[:,7:9]
y = mydata.target

seedMLP = 2357
np.set_printoptions(precision = 3)

scaler = StandardScaler()
scaler.fit(X)
Xstd = scaler.transform(X)


#modelnow = DecisionTreeClassifier(random_state = 0)
modelnow = MLPClassifier(hidden_layer_sizes=(10,20), max_iter = 1000)
modelnow.fit(Xstd,y)

yhat = modelnow.predict(Xstd)
print("Training Accuracy")
print(metrics.accuracy_score(y,yhat))

seed=3421
actuals=[]
probs=[]
hats=[]

kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in kfold.split(X, y):
    #print('train: %s, test: %s' % (train, test))
    # Train classifier on training data, predict test data
    scaler.fit(X[train]) #learn scaling parameters on training data
    Xtrain= scaler.transform(X[train])
    Xtest = scaler.transform(X[test]) #Apply transform to test data
    modelnow.fit(Xtrain, y[train])
    foldhats = modelnow.predict(Xtest)
    foldprobs = modelnow.predict_proba(Xtest)[:,1] # Class probability estimates for ROC curve
    actuals = np.append(actuals, y[test]) #Combine targets, then probs, and then predictions from each fold
    probs = np.append(probs, foldprobs)
    hats = np.append(hats, foldhats)

print ("Crossvalidation Error")    
print ("CVerror = ", metrics.accuracy_score(actuals,hats))
print (metrics.classification_report(actuals, hats))
cm = ConfusionMatrix(actuals,hats)
print (cm)
cm.print_stats() 


if len(mydata.target_names) == 2: #ROC curve code here only for 2 classes
    fpr, tpr, threshold = metrics.roc_curve(actuals, probs)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


modelnow.fit(Xstd,y)
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(modelnow, out_file= None)
grpah = graphviz.Source(dot_data)
grpah.render("TreeModelExample1")
grpah.view()