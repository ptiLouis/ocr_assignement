import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import os 
from sklearn import svm
from sklearn import metrics
from sklearn import datasets
extension = ".jpg"
x_trainSet = []
x_testSet = []
y_trainSet = []
y_testSet = []
directory  = "/Users/louisrioux/Downloads/dataset(7)/chars74k-lite/"
labels = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
currentImg = ""
summ = 0 

for letter in labels :
    
    directory = directory + letter

    directory1 = directory 

    for countPicture in range(0,len(os.listdir(directory))):
        directory1 =directory1 + "/" + letter + "_"+ str(countPicture) + extension
        img = cv2.imread(directory1)

        if(countPicture/len(os.listdir(directory)))*100 <=80 : 
            x_trainSet.append(img)
            y_trainSet.append(labels.index(letter))

   
        elif(countPicture/len(os.listdir(directory)))*100 >80 :
   
            x_testSet.append(img)
            y_testSet.append(labels.index(letter))
    
        directory1 = directory
    directory = "/Users/louisrioux/Downloads/dataset(7)/chars74k-lite/"

x_testSet = np.asarray(x_testSet)
x_trainSet = np.asarray(x_trainSet)
y_testSet = np.asarray(y_testSet)
y_trainSet = np.asarray(y_trainSet) 

print(np.shape(x_trainSet))
print(np.shape(y_trainSet))
""""#Create a svm Classifier
clf = svm.SVC(kernel='rbf') # Linear Kernel

#Train the model using the training sets
clf.fit(x_trainSet, y_trainSet)

#Predict the response for test dataset
y_pred = clf.predict(x_testSet)

#Import scikit-learn metrics module for accuracy calculation

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_testSet, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_testSet, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_testSet, y_pred))"""