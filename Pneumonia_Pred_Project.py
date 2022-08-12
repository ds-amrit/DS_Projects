# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:52:48 2020

@author: OMEN
"""
# Importing required libraries
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import jaccard_score,classification_report, log_loss
from sklearn import metrics
import imutils
import cv2



# Setting input path to dataset location.
input_path="D:\\delaPlex\\IMAGE SEGMENTATION\\chest_xray\\"

fig, ax = plt.subplots(2, 3, figsize=(15, 7))
ax = ax.ravel()
plt.tight_layout()

## VISUALIZATION OF NORMAL AND PNEUMONIA LUNG SCAN FROM TRAIN,TEST AND VALIDATION DATA

for i, _set in enumerate(['train', 'val', 'test']):
    set_path = input_path+_set
    ax[i].imshow(plt.imread(set_path+'\\NORMAL\\'+os.listdir(set_path+'\\NORMAL\\') [0]), cmap='gray')
    ax[i].set_title('Set: {}, Condition: Normal'.format(_set))
    ax[i+3].imshow(plt.imread(set_path+'\\PNEUMONIA\\'+os.listdir(set_path+'\\PNEUMONIA\\') [0]), cmap='gray')
    ax[i+3].set_title('Set: {}, Condition: Pneumonia'.format(_set))


# Seeting the directories to image folder
normal_example=os.listdir(input_path+'train\\NORMAL')[0]
normal_img=plt.imread(input_path+'train\\NORMAL\\'+normal_example)
pn_example=os.listdir(input_path+'train\\PNEUMONIA')[0]
pn_img=plt.imread(input_path+'train\\PNEUMONIA\\'+pn_example)
plt.imshow(normal_img)
plt.title("Normal Lung Scan")
plt.imshow(pn_img)
plt.title("Pneumonia Lung Scan")

for set in ["train","val","test"]:
    n_normal=len(os.listdir(input_path+set+"\\NORMAL"))
    n_pneumonia=len(os.listdir(input_path+set+"\\PNEUMONIA"))
    print("Set: {}   normal images: {}  pneumonia images:{}".format(set,n_normal,n_pneumonia))
    
##Preparing Training Data
    
train_data=[]
for i in range(len(os.listdir(input_path+"train\\NORMAL"))):
    train_data.append((input_path+'train\\NORMAL\\'+os.listdir(input_path+'train\\NORMAL')[i],0))
for i in range(len(os.listdir(input_path+"train\\PNEUMONIA"))):
    train_data.append((input_path+'train\\PNEUMONIA\\'+os.listdir(input_path+'train\\PNEUMONIA')[i],1))

#converting the traing data list into pandas df

train_data=pd.DataFrame(train_data,columns=["image","label"],index=None)

#Shuffling the different categories of label

train_data = train_data.sample(frac=1.).reset_index(drop=True)

#count of cases

cases_count=train_data['label'].value_counts()
print(cases_count) 
plt.figure(figsize=(15,15))
sns.barplot(x=cases_count.index,y=cases_count.values)
plt.title("Number of Cases",fontsize=15)
plt.xlabel("Case Type",fontsize=13)
plt.ylabel("Count", fontsize=13)
plt.xticks(cases_count.index,['Pneumonia(1)', 'Normal(0)'])
plt.show()




def image_to_vector(image,size=(32,32)):
    return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	# return the flattened histogram as the feature vector
	return hist.flatten()



rawImages = []
features = []
labels = []

for i in train_data["image"]:
    image=cv2.imread(i)
    pixels=image_to_vector(image)
    hist=extract_color_histogram(image)
    rawImages.append(pixels)
    features.append(hist)
for i in train_data["label"]:
    labels.append(i)

    

rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
x_train,x_test,y_train,y_test=train_test_split(rawImages, labels,test_size=0.25,random_state=0)

# Model 1: Building Machine Learning Model using Raw pixel.
    
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
yhat=model.predict(x_test)
acc = model.score(x_test, y_test)

print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

#Plotting Confusion Matrix


disp=plot_confusion_matrix(model,x_test,y_test)


# Hyper Parameter Tuning for Model 1



k=10
mean_acc=[]
std_acc = []
for n in range(1,k):
    mod=KNeighborsClassifier(n_neighbors=n).fit(x_train,y_train)
    ypred=mod.predict(x_test)
    mean_acc.append(metrics.accuracy_score(y_test,ypred))
    std_acc.append(np.std(ypred==y_test)/np.sqrt(ypred.shape[0]))
mean_acc/=100
std_acc/=100
print()
   
# MODEL 2 : Building Knn model Using Color Histogram Pixels

xtrain,xtest,ytrain,ytest = train_test_split(features,labels,test_size=0.25, random_state=0)
model1=KNeighborsClassifier(n_neighbors=3) 
model1.fit(xtrain,ytrain)
ypre=model1.predict(xtest)
acc = model1.score(xtest, ytest)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

#Plotting Confusion Matrix

from sklearn.metrics import plot_confusion_matrix
disp=plot_confusion_matrix(model1,xtest,ytest)

# Hyper Parameter tuning for Model 2 

k=15
meanacc=[]
stdacc = []
for n in range(1,k):
    mod=KNeighborsClassifier(n_neighbors=n).fit(xtrain,ytrain)
    y_pred=mod.predict(xtest)
    meanacc.append(metrics.accuracy_score(ytest,y_pred))
    stdacc.append(np.std(y_pred==ytest)/np.sqrt(y_pred.shape[0]))
    


    
# Model3: Building Logistic Regression using raw pixels 

lr=LogisticRegression(C=0.01,solver="saga",class_weight='balanced',n_jobs=-1).fit(x_train,y_train)
ypr=lr.predict(x_test)
ypr_prob=lr.predict_proba(x_test)
accu=metrics.accuracy_score(y_test,ypr)

# Accuracy of the model
print("Accuracy of Logistic Regression Model ={}%".format(accu*100))

# Printing Jaccard Similarity Score
print("Jaccard Similarity Score for Logistic Regression Model={}%".format(jaccard_score(y_test,ypr)))

# Classification report for logistic regression model.
print(classification_report(y_test,ypr))

# Log Loss for the logistic regression model.
print("Log Loss for Logistic Regression Model ={}".format(log_loss(y_test,ypr)))
