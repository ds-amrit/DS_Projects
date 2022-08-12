# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:47:53 2022

@author: amrit
"""
# Importing General Libraries:
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pprint import pprint as pp
import csv
from pathlib import Path
import seaborn as sns
from itertools import product
import string

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.wordnet import WordNetLemmatizer

# Importing libraries to resample the class distribution
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline 
from imblearn.under_sampling import RandomUnderSampler

# Importing libraries for Machine Learning models and evaluation metrics
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB





# CREDIT CARD FRAUD DETECTION SUPERVISED LEARNING DATASET

dftrain = pd.read_csv("D:\Projects\Fraud Detection ML\DATA\chapter_1\chapter_1\creditcard_sampledata_3.csv",index_col=0) 

dftrain.info()

# Checking the frequency of Class variable
occ = dftrain['Class'].value_counts()
print(occ)
# The number of fraud cases are 50 and the non fradulent transactions are 5000.
# This is a highly imbalanced class distribution and must be treated before model building.
# Checking the ratio of fraud cases:
ratio_fraud = occ/len(dftrain.index)    
print(f'Ratio of fraud cases = {ratio_fraud[1]},\n Ratio of non fraud cases = {ratio_fraud[0]}')

def prep_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Convert the DataFrame into two variable
    X: data columns (V1 - V28)
    y: lable column
    """
    X = df.iloc[:, 1:30].values
    y = df.Class.values
    return X, y

def plot_data(X: np.ndarray, y: np.ndarray):

    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    plt.xlim(-30,10)
    plt.ylim(-30,15)
    return plt.show()

def plot_report(y_test,y_pred):
    print('Classification report:\n', classification_report(y_test, y_pred))
    conf_mat = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:\n', conf_mat)

# Create X and y from the prep_data function 
X, y = prep_data(dftrain)

# Plot our data by running our plot data function on X and y
plot_data(X, y)

sns.pairplot(dftrain,hue='Class')

# Splitting the dataset into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)



# Resampling target class distribution using SMOTE (Synthetic Minority Oversampling Technique)

# Defining the smote method
over = BorderlineSMOTE(sampling_strategy=0.25)
#under = RandomUnderSampler(sampling_strategy=0.05)

Xtr_re,ytr_re = over.fit_resample(X_train,y_train)
#Xtr_re,ytr_re = under.fit_resample(X_train,y_train)

#Plotting the resampled data
plot_data(Xtr_re,ytr_re)

# Checking the distribution of classes in training data
print(pd.value_counts(pd.Series(y_train)))
print(pd.value_counts(pd.Series(y_test)))

# Distribution of values in transformed training data
print(pd.value_counts(pd.Series(ytr_re)))



# USING MACHINE LEARNING ALGORITHMS TO DETECT FRAUD.

# Checking the prediction of logistic regression model with raw data
model1 = LogisticRegression(solver='liblinear')

model1.fit(X_train,y_train)

pred1 = model1.predict(X_test)

plot_report(y_test,pred1)

# Logistic Regression with SMOTE

pipeline = Pipeline([('o',over),('Logistic Regression', model1)])

pipeline.fit(X_train,y_train)

predpl = pipeline.predict(X_test)

plot_report(y_test,predpl)

# Using Random Forest Classifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
plot_report(y_test,predicted)

# Hyper Paramter tuning the RandomForest Classifier:

# Define the parameter sets to test


# Number of trees in Random Forest
n_estimators = [int(x) for x in np.linspace(start=100,stop=1200,num=5)]

# Number of features to consider at every split
max_features = ['sqrt','log2']

# Maximum depth of Trees
max_depth = [int(x) for x in np.linspace(5,30,num=6)]

min_samples_leaf =  [3, 4, 5]
min_samples_split = [8, 10, 12]
n_estimators = [100, 200, 300, 1000]

# Creating the random grid
random_grid = {
                'n_estimators':n_estimators,
                'max_features':max_features,
                'max_depth':max_depth,
                'min_samples_leaf' : min_samples_leaf,
                'min_samples_split' : min_samples_split,
                'n_estimators' : n_estimators, 
                'class_weight' : ["balanced","balanced_subsample"]
                 }

# Define the model to use
model = RandomForestClassifier(random_state=0)

# Combine the parameter sets with the defined model
CV_model = GridSearchCV(estimator=model, param_grid=random_grid, cv=5, scoring='recall', n_jobs=-1)

# Fit the model to our training data and obtain best parameters
CV_model.fit(X_train, y_train)
bm = CV_model.best_params_

print(bm)

# Predicting the fruad cases using Hyperparameter tuned random forest.
ypred_cv = CV_model.predict(X_test)
plot_report(y_test,ypred_cv)



# Using a voting classifier ensemble method to predict fraud cases.

clf1 = LogisticRegression(class_weight={0:1, 1:15},
                          random_state=5,
                          solver='liblinear')

clf2 = RandomForestClassifier(class_weight={0:1, 1:12}, 
                              criterion='gini', 
                              max_depth=8, 
                              max_features='sqrt',
                              min_samples_leaf=3,
                              min_samples_split=8,
                              n_estimators=100, 
                              n_jobs=-1,
                              random_state=5)

clf3 = DecisionTreeClassifier(random_state=5,
                              class_weight="balanced")

# Combine the classifiers in the ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('dt', clf3)], voting='hard')

# Fit and predict as with other models
ensemble_model.fit(X_train, y_train)
votpred = ensemble_model.predict(X_test)
plot_report(y_test, votpred)

# We observe the best results using this model Recall = 0.95 , Precision =1
# The model was able to detect the fruad 18 out of 19 total fraud cases.







