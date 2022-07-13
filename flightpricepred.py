# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 00:24:55 2022

@author: amrit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
import pickle


sns.set()

# Importing the dataset
df = pd.read_excel("D:\\Projects\\Flight price pred\\archive\\Data_Train.xlsx")

df.info()

# Checking for null values in the dataset.
df.isnull().sum()

# Dropping null values as its just 1.
df.dropna(inplace=True)


###Exploratory Data Analysis

## Feature Engineering

# Creating new features based on Date of Journey

df["Journey_day"]= pd.to_datetime(df.Date_of_Journey,format = "%d/%m/%Y").dt.day


df["Journey_month"] =pd.to_datetime(df.Date_of_Journey,format = "%d/%m/%Y").dt.month

train = df.drop(["Date_of_Journey"],axis=1)

# Creating new features based on Departure time

train["departure_hr"] = pd.to_datetime(train.Dep_Time).dt.hour
train["departure_min"] = pd.to_datetime(train.Dep_Time).dt.minute

train.drop("Dep_Time",axis=1,inplace=True)

# Creating new features based on Arrival Time

train["arrival_hr"] = pd.to_datetime(train.Arrival_Time).dt.hour

train["arrival_min"] = pd.to_datetime(train.Arrival_Time).dt.minute

train.drop("Arrival_Time",axis=1,inplace=True)


# Creating new features based on Flight Duration.
duration_hr=[]
duration_min = []
for i in train.Duration:
    if len(i) in [2,3]:
        if 'h' in i:
            duration_hr.append(i.split("h")[0])
            duration_min.append(0)
        if 'm' in  i:
            duration_hr.append(0)
            duration_min.append(i.split('m')[0])
    else:
        hr= i.split("h")[0]
        min= i.split("m")[0].split()[-1]
        duration_hr.append(int(hr))
        duration_min.append(int(min))
        
train["duration_hrs"] = [int(i) for i in duration_hr]
train["duration_mins"] = duration_min

train.drop("Duration",axis=1,inplace=True)

# Handling Categorical Data 


# Converting Categorical data to Numerical Data

train.Airline.value_counts()

# As there is only one record for it we are dropping this category.
train = train.loc[train.Airline != 'Trujet',:]
df = df.loc[df.Airline != 'Trujet',:]

# As Airline is Nominal Categorical varibale , one hot encoding is used.

Airline = train[["Airline"]]
Airline = pd.get_dummies(Airline,drop_first=True)
Airline.head()

# As Source is Nominal Categorical variable, one hot encoding is used.

train.Source.value_counts()
Source = train[["Source"]]
Source = pd.get_dummies(Source,drop_first=True)

# As Destination is Nominal Categorical variable , one hot encoding is used.

train.Destination.value_counts()
train.Destination.replace({'Delhi':'New Delhi'},inplace=True)
df.Destination.replace({'Delhi':'New Delhi'},inplace=True)

Destination = train[["Destination"]]
Destination = pd.get_dummies(Destination,drop_first=True)

# As Total_Stops is a Ordinal Categorical variable, label encoding is performed.

train.Total_Stops.value_counts()

train.Total_Stops.replace({'non-stop':0,'1 stop': 1,'2 stops': 2,'3 stops': 3,'4 stops':4},inplace=True)

# Dropping 2 columns as they are unnecessary.
train.drop(["Route","Additional_Info"],inplace=True,axis=1)

train_data = pd.concat([train,Airline,Source,Destination],axis=1)

train_data.drop(["Airline","Source","Destination"],inplace=True,axis=1)

# Bivariate Analysis 

# Airline vs Price
sns.catplot(x="Airline",y="Price",data=df.sort_values("Price",ascending=False),kind='boxen',height=6,aspect =3)

# Source vs Price
sns.catplot(x="Source",y="Price",data=df.sort_values("Price",ascending=False),kind='boxen',height=6,aspect =3)

# Destination vs Price
sns.catplot(x="Destination",y="Price",data=df.sort_values("Price",ascending=False),kind='boxen',height=6,aspect=3)

# Total_Stops vs Price
sns.catplot(x="Total_Stops",y="Price",data=df.sort_values("Price",ascending=False),kind='boxen',height=6,aspect=3)


# Importing Test Data and performing preprocessing as Train data

test = pd.read_excel("D:\\Projects\\Flight price pred\\archive\\Test_set.xlsx")
    

test["Journey_day"]= pd.to_datetime(test.Date_of_Journey,format = "%d/%m/%Y").dt.day


test["Journey_month"] =pd.to_datetime(test.Date_of_Journey,format = "%d/%m/%Y").dt.month

test = test.drop(["Date_of_Journey"],axis=1)

# Creating new features based on Departure time

test["departure_hr"] = pd.to_datetime(test.Dep_Time).dt.hour
test["departure_min"] = pd.to_datetime(test.Dep_Time).dt.minute

test.drop("Dep_Time",axis=1,inplace=True)

# Creating new features based on Arrival Time

test["arrival_hr"] = pd.to_datetime(test.Arrival_Time).dt.hour

test["arrival_min"] = pd.to_datetime(test.Arrival_Time).dt.minute

test.drop("Arrival_Time",axis=1,inplace=True)


# Creating new features based on Flight Duration.
duration_hr=[]
duration_min = []
for i in test.Duration:
    if len(i) in [2,3]:
        if 'h' in i:
            duration_hr.append(i.split("h")[0])
            duration_min.append(0)
        if 'm' in  i:
            duration_hr.append(0)
            duration_min.append(i.split('m')[0])
    else:
        hr= i.split("h")[0]
        min= i.split("m")[0].split()[-1]
        duration_hr.append(int(hr))
        duration_min.append(int(min))
        
test["duration_hrs"] = [int(i) for i in duration_hr]
test["duration_mins"] = duration_min

test.drop("Duration",axis=1,inplace=True)

# Handling Categorical Data 


# Converting Categorical data to Numerical Data

test.Airline.value_counts()

# As Airline is Nominal Categorical varibale , one hot encoding is used.

Airline = test[["Airline"]]
Airline = pd.get_dummies(Airline,drop_first=True)
Airline.head()

# As Source is Nominal Categorical variable, one hot encoding is used.

test.Source.value_counts()
Source = test[["Source"]]
Source = pd.get_dummies(Source,drop_first=True)

# As Destination is Nominal Categorical variable , one hot encoding is used.

test.Destination.value_counts()
test.Destination.replace({'Delhi':'New Delhi'},inplace=True)

Destination = test[["Destination"]]
Destination = pd.get_dummies(Destination,drop_first=True)

# As Total_Stops is a Ordinal Categorical variable, label encoding is performed.

test.Total_Stops.value_counts()

test.Total_Stops.replace({'non-stop':0,'1 stop': 1,'2 stops': 2,'3 stops': 3,'4 stops':4},inplace=True)

# Dropping 2 columns as they are unnecessary.
test.drop(["Route","Additional_Info"],inplace=True,axis=1)

test_data = pd.concat([test,Airline,Source,Destination],axis=1)

test_data.drop(["Airline","Source","Destination"],inplace=True,axis=1)

# Splitting Independant X and Dependant Variable Y, price from train data

X = train_data.drop("Price",axis=1)

Y = train_data.iloc[:,1]

## Feature Selection

# Checking correlation between variables:
plt.figure(figsize=(18,18))
sns.heatmap(train_data.corr(),annot=True,cmap="RdYlGn")    

# Selecting important features in the dataset:
    
selection = ExtraTreesRegressor()
selection.fit(X,Y)

# Plotting graph for feature importance

feat_importance = pd.Series(selection.feature_importances_,index=X.columns)
feat_importance.nlargest(20).plot(kind='barh')
plt.title("List of Feature Importance in Asc order")

## MACHINE LEARNING MODEL BUILDING

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=1)


# Building a Radom Forest Regressor Model
rfr = RandomForestRegressor()

# Fitting a model
rfr.fit(x_train,y_train)

# Predicting the y label for x test
y_pred_rfr = rfr.predict(x_test)

# Calculating the score of Random Forest Regressor model on train data
rfr.score(x_train,y_train)

# Calculating the score of Random Forest Regressor model on test data
rfr.score(x_test,y_test)

# Plotting the predicted results
sns.distplot(y_test- y_pred_rfr)

# Calculating Evaluation metrics for the model

print("MAE",metrics.mean_absolute_error(y_test, y_pred_rfr))
print("MSE",metrics.mean_squared_error(y_test, y_pred_rfr))
print("RMSE",np.sqrt(metrics.mean_squared_error(y_test, y_pred_rfr)))
print("R2 score",metrics.r2_score(y_test, y_pred_rfr))

## Hyper Parameter Tuning using Randomized Search CV

# Declaring the features for Randomized SearchCV model

# Number of trees in Random Forest
n_estimators = [int(x) for x in np.linspace(start=100,stop=1200,num=12)]

# Number of features to consider at every split
max_features = ['auto','sqrt']

# Maximum depth of Trees
max_depth = [int(x) for x in np.linspace(5,30,num=6)]

# Minimum number of Samples required to split a node
min_sample_split = [2,5,10,15,100]

# Minimum number of samples required at each leaf node
min_sample_leaf = [1,2,5,10]

# Creating the random grid
random_grid = {
                'n_estimators':n_estimators,
                'max_features':max_features,
                'max_depth':max_depth,
                'min_samples_split':min_sample_split,
                'min_samples_leaf':min_sample_leaf }

# Creating a Randomized Search CV model
rscv = RandomizedSearchCV(estimator = rfr, param_distributions= random_grid,scoring='neg_mean_squared_error',n_iter=10,cv =5, verbose =2,random_state=1,n_jobs=1)

# Fitting Randomized Search CV model
rscv.fit(x_train,y_train)

# Checking the best parameters 
rscv.best_params_

# Predicting the price for x_test
y_pred_rscv = rscv.predict(x_test)

print("MAE using RandomizedSearchCV",metrics.mean_absolute_error(y_test, y_pred_rscv))
print("MSE using RandomizedSearchCV",metrics.mean_squared_error(y_test, y_pred_rscv))
print("RMSE using RandomizedSearchCV",np.sqrt(metrics.mean_squared_error(y_test, y_pred_rscv)))
print("R2 score using RandomizedSearchCV",metrics.r2_score(y_test, y_pred_rscv))

# Saving the model to reuse it again

file = open("D:\\Projects\\Flight price pred\\rscv.pkl",'wb')

# Dump information into the file
pickle.dump(rscv,file)

model = open("D:\\Projects\\Flight price pred\\rscv.pkl",'rb')

forest = pickle.load(model)

y_pred =  forest.predict(x_test)
metrics.r2_score(y_test, y_pred)












