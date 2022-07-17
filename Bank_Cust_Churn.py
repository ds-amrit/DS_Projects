# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 22:22:32 2022

@author: amrit
"""
# Importing Data Preprocessing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures,MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Importing libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Importing Machine Learning Models and evaluation metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

# Importing Deep Learning Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense,LeakyReLU,PReLU, ELU, Dropout
import tensorflow as tf
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

print(tf.test.is_gpu_available(cuda_only=True))
print(tf.config.list_physical_devices('GPU'))
# Importing the dataset

df = pd.read_csv("D:\\Projects\\BANK CUSTOMER CHURN\\Churn_Modelling.csv",index_col=0)

df.info()

# Dropping unnecessary columns

df.drop(["CustomerId","Surname"],axis=1,inplace=True)

# Getting unique count for each  variable

df.nunique()

# Checking the datatype of the columns
df.dtypes

# DATA VISUALIZATION

# Plotting the distribution of dependant variable i.e., Exited

df.Exited.value_counts()
labels = ['Retained',"Exited"]

fig1, ax1 = plt.subplots(figsize=(10,8))
ax1.pie(df.Exited.value_counts(),explode=[0,0.1],labels = labels,autopct='%1.1f%%',startangle=90,shadow=True)
ax1.axis('equal')
plt.title("Proportion of customer Churned and Retained",size=20)
plt.show()

"""
 The basic dataset  project a customer churn rate of 20%.
 Given that 20% is a relatively tiny percentage, we must make sure that the model we choose
 can accurately predict this 20%, as the bank is more interested in locating and retaining 
 this group than it is in precisely predicting the clients who are retained.
"""

# Plotting the distribution of categorical variables with customer status

fig2,ax2 = plt.subplots(2,2,figsize=(20,12))
sns.countplot(x='Geography',hue="Exited",data=df,ax= ax2[0][0])
sns.countplot(x="Gender",hue="Exited",data=df,ax=ax2[0][1])
sns.countplot(x="IsActiveMember",hue="Exited",data=df,ax=ax2[1][0])
sns.countplot(x="HasCrCard",hue="Exited",data=df,ax=ax2[1][1])

"""
Observations from the data:
1. The vast majority of the data comes from French citizens. 
However, the percentage of churned customers is inversely correlated to the number of customers,
suggesting that the bank may be experiencing difficulties (possibly due to inadequate customer service resources)
in areas where it has fewer customers.

2. Additionally, a higher percentage of women than men leave the bank.

3. It's interesting that customers who use credit cards make up the majority of churning customers, this might just be a coincidence.

4. Unsurprisingly, the churn rate is higher among inactive members. 
Worryingly, the bank may need to implement a programme to make this group active 
given that the overall percentage of inactive members is quite high.
"""

# Plotting the variation of numerical variables with different status

fig3,ax3 = plt.subplots(3,2,figsize=(21,12))
ax3.boxplot(x="Exited",y="CreditScore",data=df,hue="Exited",ax=ax3[0][0])
ax3.boxplot(x="Exited",y="Age",data=df,hue="Exited",ax=ax3[0][1])
ax3.boxplot(x="Exited",y="Tenure",data=df,hue="Exited",ax=ax3[1][0])
ax3.boxplot(x="Exited",y="Balance",data=df,hue="Exited",ax=ax3[1][1])
ax3.boxplot(x="Exited",y="NumOfProducts",data=df,hue="Exited",ax=ax3[2][0])
ax3.boxplot(x="Exited",y="EstimatedSalary",data=df,hue="Exited",ax=ax3[2][1])

df_train = df.sample(frac=0.8,random_state=1)
df_test = df.drop(df_train.index)
print(len(df_train))
print(len(df_test))

df_num = df_train.select_dtypes(include ='number')
df_cat = df_train.drop(list(df_num.columns),axis=1)
df_cat.reset_index(inplace=True,drop =True)

# Scaling Numerical features

scaler = StandardScaler()

df_num_s = pd.DataFrame(scaler.fit_transform(df_num),columns=df_num.columns)

# One Hot Encoding Categorical variables

df_cat_e = pd.get_dummies(df_cat,drop_first = True)

train_data = pd.concat([df_num_s,df_cat_e],axis=1)


## Model Building

x_train,x_test,y_train,y_test = train_test_split(train_data.drop('Exited',axis=1),df_train['Exited'],test_size=0.2)

# Defining a function to print best model score and parameters
def best_model(model):
    print(model.best_score_)    
    print(model.best_params_)
    print(model.best_estimator_)

# Model 1: Parameterized Logistic Regression

param_grid = {'C':[0.1,0.5,1,3,5,10], 'max_iter': [250],'fit_intercept' : [True],'intercept_scaling':[1,2,4],'penalty' : ['l2'], 'tol' : [0.0001,0.0001] }

log_grid = GridSearchCV(LogisticRegression(solver='lbfgs'),param_grid= param_grid,cv=5,refit = True)

log_grid.fit(x_train,y_train)    

best_model(log_grid)

print(classification_report(y_test,log_grid.best_estimator_.predict(x_test)))

# Model 2: Fitting Logistic Regression using Degree 2 Polynomial Kernel.

poly = PolynomialFeatures(degree=2)

poly_train = poly.fit_transform(x_train)

log_poly_grid = GridSearchCV(LogisticRegression(solver='liblinear'),param_grid= param_grid,cv=10,refit=True,n_jobs=-1)

log_poly_grid.fit(poly_train,y_train)

best_model(log_poly_grid)

print(classification_report(y_test,log_poly_grid.best_estimator_.predict(poly.fit_transform(x_test))))

# Model 3: Support Vector Machine 

param_grid = {'C':[0.5,5,50,100],'gamma':[0.1,0.01,0.001],'probability':[True],'kernel': ['rbf','poly']}

svm_grid = GridSearchCV(SVC(),param_grid,cv=6,refit=True,verbose=0,n_jobs=-1)

svm_grid.fit(x_train,y_train)

best_model(svm_grid)

print(classification_report(y_test,svm_grid.best_estimator_.predict(x_test)))


# Model 4: Random Forest Classifier

param_grid = {'max_depth':[3,5,7,9],'max_features':[2,4,6,8],'n_estimators':[50,100],'min_samples_split':[2,3,4,5]}

Rf_grid = GridSearchCV(RandomForestClassifier(),param_grid,cv=6,refit=True)    
    
Rf_grid.fit(x_train,y_train)

best_model(Rf_grid)

print(classification_report(y_test,Rf_grid.best_estimator_.predict(x_test)))


# 



# Testing the models on Test Data

def data_pp(test):
    
    df_num = test.select_dtypes(include ='number')
    df_cat = test.drop(list(df_num.columns),axis=1)
    df_cat.reset_index(inplace=True,drop =True)
    
    # Scaling Numerical features
    
    scaler = MinMaxScaler()
    
    df_num_s = pd.DataFrame(scaler.fit_transform(df_num),columns=df_num.columns)
    
    # One Hot Encoding Categorical variables
    
    df_cat_e = pd.get_dummies(df_cat,drop_first = True)
    
    test_data = pd.concat([df_num_s,df_cat_e],axis=1)
    return test_data

test_data = data_pp(df_test)
    


# Model 1: Parameterized Logistic Regression

X = test_data.drop('Exited',axis=1)
y = test_data.Exited



print(classification_report(y,log_grid.best_estimator_.predict(X)))

# Model 2: Fitting Logistic Regression using Degree 2 Polynomial Kernel.



print(classification_report(y,log_poly_grid.best_estimator_.predict(poly.fit_transform(X))))

# Model 3: Support Vector Machine 


print(classification_report(y,svm_grid.best_estimator_.predict(X)))


# Model 4: Random Forest Classifier

print(classification_report(y,Rf_grid.best_estimator_.predict(X)))

## Preparing data for deep learning model

df_num = df_train.select_dtypes(include ='number')
df_num.reset_index(inplace=True,drop=True)
df_cat = df_train.drop(list(df_num.columns),axis=1)
df_cat.reset_index(inplace=True,drop =True)


sscaler = StandardScaler()

df_num_s = pd.DataFrame(sscaler.fit_transform(df_num),columns=df_num.columns)

# One Hot Encoding Categorical variables

df_cat_e = pd.get_dummies(df_cat,drop_first = True)

train_data = pd.concat([df_num,df_cat_e],axis=1)

X_train,X_test,y_train,y_test = train_test_split(train_data.drop("Exited",axis=1),train_data.Exited,test_size=0.2,random_state=1)

X_train = sscaler.fit_transform(X_train)

X_test = sscaler.transform(X_test)

# Building Deep Learning Model
classifier = Sequential()

# Adding the first input layer and hidden layer

classifier.add(Dense(units=6, kernel_initializer = 'he_uniform',activation = 'relu',input_dim=11))

# Adding 2nd hidden layer
classifier.add(Dense(units=6, kernel_initializer = 'he_uniform',activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer = 'glorot_uniform',activation = 'sigmoid'))

# Compiling the model
classifier.compile(optimizer='Adamax',loss = 'binary_crossentropy',metrics=['accuracy'])

# Architecture summary for the model
classifier.summary()

# Fitting the ann to test data

model_history = classifier.fit(X_train,y_train,batch_size=10,epochs=100)

y_pred_ann = classifier.predict(X_test)
y_pred_ann = (y_pred_ann > 0.5 )

#print(confusion_matrix(y_test,y_pred_ann))
print(classification_report(y_test, y_pred_ann))

# Checking the performance on test data
accuracy_score(y_pred_ann,y_test)

# Defining a model using hyper parameter tuning the layers, units and learning rate.
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers',2,20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),min_value=32,max_value=512,step=32),activation='relu'))
    model.add(Dense(units=1, kernel_initializer = 'glorot_uniform',activation = 'sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',[0.001,0.0001,0.00001])),loss='binary_crossentropy',metrics=['accuracy'])
    return model

# Running the Keras Tuner
tuner = RandomSearch(
                     build_model,
                     objective = 'val_accuracy' ,
                    max_trials= 5,
                    executions_per_trial =3)

# Fitting the hyper-parameteric model to the data to find the best results.
tuner.search(X_train,y_train,epochs=5,validation_data=(X_test,y_test))

# Result summary of the top 10 best performing models
tuner.results_summary()

# Fetching the best hyper-parameters with highest accuracy.
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Building the deep learning model with optimized hyper-parameters.
model = tuner.hypermodel.build(best_hps)

# Fitting the optimized model on train data
history = model.fit(X_train,y_train,epochs=10,validation_split=0.2)

# Evaluation of model on test data
eval_result = model.evaluate(X_test,y_test)

print("[test loss, test accuracy]:", eval_result)


#X_s = sscaler.transform(X)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

hypermodel.fit(X_train,y_train,epochs=best_epoch,validation_split=0.2)


print("[test loss, test accuracy]: final tuned model", hypermodel.evaluate(X_test,y_test))

y_pred_hp = hypermodel.predict(X_test)   

y_pred_h = np.where(y_pred_hp > 0.5,1,0)

print(classification_report(y_test, y_pred_h))  
        










