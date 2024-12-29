#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:07:54 2024

@author: mosesodeiaddai
"""

#This project involves the implementation of predictive analytics with logistic regression in predicting prices of houses in California.  

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#getting data
data = pd.read_csv("/Users/mosesodeiaddai/Downloads/california_housing_test_1.csv")
data['bdtohousehold'] = data["total_bedrooms"]/data["households"]


#extracting features
X = data[["housing_median_age","median_income","bdtohousehold"]]
y = data["median_house_value"]

#splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#scaling data
scaler = StandardScaler()
Xtrainsc = scaler.fit_transform(X_train)
Xtestsc = scaler.fit_transform(X_test)

#training logistic regression model
logistic = LogisticRegression(max_iter=2000)
logistic.fit(Xtrainsc, y_train)

#making predictions based on test set
ypred = logistic.predict(Xtestsc)
print(ypred)

#evaluating model
precision = precision_score(y_test, ypred, average='macro',zero_division=0)
recall = recall_score(y_test, ypred, average='macro',zero_division=0)      
f1_macro = f1_score(y_test, ypred, average='macro',zero_division=0)    

print(f"Precision (Macro): {precision:.2f}")
print(f"Recall (Macro): {recall:.2f}")
print(f"F1 Score (Macro): {f1_macro:.2f}")