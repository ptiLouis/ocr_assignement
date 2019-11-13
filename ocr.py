import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import os 
import random
from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
# 255 white and 0 black 
extension = ".jpg"
x_trainSet = []
x_testSet = []
y_trainSet = []
y_testSet = []
directory  = "/Users/louisrioux/Downloads/dataset(7)/chars74k-lite/"
labels = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
currentImg = ""
summ = 0 



# Feature engeeniring part 
# ìn this fonction i want to normalise the intensity pixels to get the same value for every pictures 
def forceColour(img) : 
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            if img[i][j] <= 130 : 
                img[i][j] = 0
            else : 
                img[i][j] = 1 
    return img

# I want to force the image to have the background in white and letter in black 
"""we are checking the corner of a picture so : 
(0,0); (0,20) ;(20,0), (20,20)
"""
def reversePattern(img): 
    sum = 0
    average = 0
    chosenPixels = 3
    nTotalPixels = chosenPixels * chosenPixels * 4   
    for i in range(0, chosenPixels) : 
        for j in range(0,chosenPixels) :
            sum += img[i][j] + img[19 - i][j] + img[i][19 - j] + img[19-i][19 - j]
    average = sum /nTotalPixels
    # we change the background to white
    if average < 130 :
        for i in range(0, len(img)):
            for j in range(0, len(img[i])):
                if img[i][j] == 1 : 
                    img[i][j] = 0
                else : 
                    img[i][j] = 1 
    return img
# dont apply the reversePattern before the forcedColor 
    

labelsImages = {}
countlabel = 0 
for letter in labels :
    labelsImages[letter] = []
    directory = directory + letter

    directory1 = directory 
    
    for countPicture in range(0,len(os.listdir(directory))):
        directory1 =directory1 + "/" + letter + "_"+ str(countPicture) + extension

        img = cv2.imread(directory1,0)
        img = forceColour(img)
        img = reversePattern(img)
        image_labeled = (img, labels.index(letter))
        labelsImages[letter].append(image_labeled)
        """"if(countPicture/len(os.listdir(directory)))*100 <=80 : 
            x_trainSet.append(img)
            y_trainSet.append(labels.index(letter))

   
        elif(countPicture/len(os.listdir(directory)))*100 >80 :
   
            x_testSet.append(img)
            y_testSet.append(labels.index(letter))"""
    
        directory1 = directory
    directory = "/Users/louisrioux/Downloads/dataset(7)/chars74k-lite/"

labeledFeatures = []
percentage = 0 
trainingSet = []
testSet = []
for letter in labels : 
    for i in labelsImages[letter] : 
        labeledFeatures.append(i)
    random.shuffle(labeledFeatures)
    for c in range(0, len(labeledFeatures)):
        percentage = (c/len(labeledFeatures))*100
        if percentage <= 80.0 : 
            trainingSet.append(labeledFeatures[c])
        else :
            testSet.append(labeledFeatures[c])
    labeledFeatures =[]
random.shuffle(trainingSet)
random.shuffle(testSet)

x_train = []
y_train = []
x_test = []
y_test = []
for i in range(0,len(trainingSet)):
    x_train.append(trainingSet[i][0])
    y_train.append(trainingSet[i][1])

for i in range(0,len(testSet)):
    x_test.append(testSet[i][0])
    y_test.append(testSet[i][1])

x_train = np.array(x_train)
x_train = np.reshape(x_train,[x_train.shape[0],x_train.shape[1]*x_train.shape[2]])
x_test = np.array(x_test)
x_test = np.reshape(x_test,[x_test.shape[0],x_test.shape[1]*x_test.shape[2]])
print(x_test.shape)


#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))