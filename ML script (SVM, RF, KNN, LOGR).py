#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:47:38 2019

@author: lucyrothwell
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:39:19 2019

@author: lucyrothwell
"""

# CONTENTS
# Data visualisation, missing values & downsampling
# K Nearest Neighbour (KNN)
# Support Vector Machine (SVM)
# Randomforest
# Logistic Regression
# Writing the outputs


# IMPORT LIBRARIES

import pandas as pd  #Pandas for managing data
import matplotlib.pyplot as plt #Matplotlib for plotting
import matplotlib

# K-fold cross validation
import sklearn
from sklearn.model_selection import cross_val_score 
#from sklearn.model_selection import KFold
#from sklearn.model_selection import LeaveOneOut
from sklearn import datasets

print('The scikit-learn version is {}.'.format(sklearn.__version__))

#The machine learning algorithms from the sklean libraries. 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# from sklearn.feature_extraction_text import CounterVectorizer  # (YouTube: https://www.youtube.com/watch?v=HuDIbSCnsqo)
# from sklearn.cross_validation import train_test_split # (YouTube: https://www.youtube.com/watch?v=HuDIbSCnsqo)
# from sklearn.naive_bayes import Multinomial_MBnp.array([0])np.array([0])

#The regression machine learning algorithms from the sklean libraries. 
from sklearn.ensemble import RandomForestRegressor

# Tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

# Scoring metrics from sklearn
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_curve, auc, confusion_matrix, balanced_accuracy_score, multilabel_confusion_matrix, classification_report # decent video guide: https://www.youtube.com/watch?v=TtIjAiSojFE
from sklearn.model_selection import cross_val_predict
# > good explanation: https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb 

# Numpy to convert to arrays
import numpy as np

# For imputation of missing values 
from sklearn.impute import SimpleImputer 
from fancyimpute import KNN


# LOAD/READ DATA 

#data_columns = list(dataFrame)
#imp.fit(dataFrame)
#SimpleImputer(copy=True, fill_value=None, strategy='mean', verbose=0)
dataFrame = "/Users/lucyrothwell/Google Drive copy/MSc  Psych - UCL/9. Dissertation (Y2)/Stuttering/*Data - EEG/PWS EEG/3. PROCESSED (.csv)/006_PWS_Social (P) - ICA - C(600) - BASELINE.csv"
dataFrame = pd.read_csv(dataFrame, encoding='utf-8', skiprows=0)
resultFileName = "Results - DELETE.csv"

# fNIRS or EEG ICA
colNum = dataFrame.shape[1]

# EEG CHANNEL
#colNum = 31

# **** ONCE THE ABOVE INFO HAS BEEN ENTERED, THE PROGRAMME CAN RUN
# AND WILL OUTPUT THE RESULTS INTO A CSV. NO MORE EDITING NEEDED FROM HERE ***

dataFrame_types = dataFrame.dtypes # variable types
dataFrame.shape

# Checking value counts (first)
valueCountsStut = dataFrame['Stutter'].value_counts()
print(valueCountsStut)
valueCountOnes = valueCountsStut[0]
valueCountZeros = valueCountsStut[1]
print(valueCountOnes)
print(valueCountZeros)

# Remove missing values
dataFrame.dropna(inplace=True)
dataFrame.shape

# Checking value counts (second)
valueCountsStut = dataFrame['Stutter'].value_counts()
print(valueCountsStut)
valueCountOnes = valueCountsStut[0]
valueCountZeros = valueCountsStut[1]
print(valueCountOnes)
print(valueCountZeros)


## RESAMPLE to balance classes (making sure there are 50% stutters and 50%
## non-stutters)
	
from sklearn.utils import resample

#Separate majority and minority classes
dataFrame_majority = dataFrame[dataFrame.Stutter==0]
dataFrame_minority = dataFrame[dataFrame.Stutter==1]

#DOWNSAMPLE majority class
dataFrame_majority_downsampled = resample(dataFrame_majority, 
                                 replace=False,     # sample with replacement
                                 n_samples=valueCountZeros, # to match N of majority class
                                 random_state=123) # reproducible results

# Combine minority class with downsampled majority class
dataFrame_downsampled = pd.concat([dataFrame_majority_downsampled, dataFrame_minority])
 
# Display new class counts
dataFrame_downsampled.Stutter.value_counts()

# Shuffle rows so that the 1s and 0s are in random order (previously the
# dataFrame showed all of class one then all of classs two which  distorts the
#  test_training set split we do later)
dataFrame = dataFrame_downsampled.sample(frac=1)

# Testing the number of 1s and 0s have balanced
valueCountsStut = dataFrame['Stutter'].value_counts()
valueCountOnes = valueCountsStut[0]
valueCountZeros = valueCountsStut[1]
print(valueCountOnes)
print(valueCountZeros)


# Make sure there is no missing data
percent_missing = dataFrame.isnull().sum()*100/len(dataFrame)
print(percent_missing)


# TRAINING
# Apply the below four algorithms  and evaluate their effectiveness.
# Algorithms:
    #K Nearest Neighbour (KNN)
    #Support Vector Machine (SVM)
    #Randomforest
    #Logistic Regression

#Separating the data into training & testing set (holdout and cross-val).
    
# 1) Creating variables for "Holdout method" split
first75 = round(len(dataFrame)*0.75)
training_set = dataFrame[0:first75]
test_set = dataFrame[first75:len(dataFrame)]

x_train = training_set.iloc[:,2:colNum]
x_test = test_set.iloc[:,2:colNum]

y_train = training_set[['Stutter']]
y_test = test_set[['Stutter']]


# 2)  Creating variables for cross validation
X = dataFrame.iloc[:,2:colNum]
y = dataFrame[['Stutter']]


# List features
feature_list = list(x_train.columns)
feature_list

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)


#K–NEAREST NEIGHBOUR (KNN)
knnmodel = KNeighborsClassifier() 
knnmodel.fit(x_train,np.ravel(y_train,order='C'))
knnpredictions = knnmodel.predict(x_test) # Why x_train not used here?
resultKNN = accuracy_score(y_test, knnpredictions)
print("K-nearest Neighbours accuracy: ", resultKNN)

#KNN with CV
knn_cv = KNeighborsClassifier()
#train model with cv of 5
cv_scores = cross_val_score(knn_cv, X, y, cv=10)
#print each cv score (accuracy) and average them
print(cv_scores)
KNN_cv = 'KNN CV (mean):{}'.format(np.mean(cv_scores))
print(KNN_cv) 


# Sensitivity and specificity - HOLDOUT
confMatrixKNN = confusion_matrix(y_test, knnpredictions)
print(confMatrixKNN)

TP = confMatrixKNN[1, 1]
TN = confMatrixKNN[0, 0]
FP = confMatrixKNN[0, 1]
FN = confMatrixKNN[1, 0]

sensitivityKNN = TP / float(TP + FN)
print(sensitivityKNN)

specificityKNN = TN / float(TN + FP)
print(specificityKNN)

classReportKNN = classification_report(y_test, knnpredictions)
print(classReportKNN) 
# > precision = what % of the observed cases were correctly predicted as negative
# > recall = what % of the observed cases were correctly predicted as positive (bottom two on confusion matrix?)
# > f1-score = combination of precision and recall ("harmonic mean")

# Create ROC curve of the KNN model - HOLDOUT
# >>> What does the graph tell us? To read: https://www.medcalc.org/manual/roc-curves.php 
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, knnpredictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic - KNN Model')
plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

AUC_KNN = label='%0.2f'% roc_auc
print(AUC_KNN) 



#SUPPORT VECTOR MACHINE (SVM)
svcmodel = SVC(kernel='rbf', gamma = "scale") # SVC is the function in sklearn
#svcmodel.fit(x_train,y_train)
svcmodel.fit(x_train,np.ravel(y_train,order='C'))
svcpredictions = svcmodel.predict(x_test)
resultSVM = accuracy_score(y_test, svcpredictions)
print('SVM accuracy: ', resultSVM)

#SVM with CV
svc_cv = SVC(kernel='rbf', gamma = "scale")
#train model with cv of 5 
# cv_scores = cross_val_score(svc_cv, X, y, cv=5) # old
cv_scores = cross_val_score(svc_cv, X, np.ravel(y,order='C'), cv=5)
#print each cv score (accuracy) and average them
print(cv_scores)
SVM_cv = ('SVM CV (mean):{}'.format(np.mean(cv_scores))) 
print(SVM_cv)

# Sensitivity & specificity
confMatrixSVM = confusion_matrix(y_test, svcpredictions)
print(confMatrixSVM)

TP = confMatrixSVM[1, 1]
TN = confMatrixSVM[0, 0]
FP = confMatrixSVM[0, 1]
FN = confMatrixSVM[1, 0]

sensitivitySVM = TP / float(TP + FN)
print(sensitivitySVM)

specificitySVM = TN / float(TN + FP)
print(specificitySVM)

classReportSVM = classification_report(y_test, svcpredictions)
print(classReportSVM) 

# Create ROC curve of the SVM model
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, svcpredictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic - SVM Model')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

AUC_SVM = label='%0.2f'% roc_auc
print(AUC_SVM)




#LOGISTIC REGRESSION
logreg = LogisticRegression(C=1,solver = 'lbfgs') 
logreg.fit(x_train,np.ravel(y_train,order='C'))
logpredictions = logreg.predict(x_test)
resultLogreg = accuracy_score(y_test, logpredictions)
print("Log Reg accuracy: ", resultLogreg)

#LOG REG with CV
logreg_cv = LogisticRegression(C=1, solver = 'lbfgs')
#train model with cv of 5 
cv_scores = cross_val_score(logreg_cv, X, y, cv=10)
#print each cv score (accuracy) and average them
logReg_cv = ('logReg CV (mean):{}'.format(np.mean(cv_scores))) 

# Sensitivity & specificity
confMatrix_logreg = confusion_matrix(y_test, logpredictions)
print(confMatrix_logreg)

TP = confMatrix_logreg[1, 1]
TN = confMatrix_logreg[0, 0]
FP = confMatrix_logreg[0, 1]
FN = confMatrix_logreg[1, 0]

sensitivity_logreg = TP / float(TP + FN)
print(sensitivity_logreg)

specificity_logreg = TN / float(TN + FP)
print(specificity_logreg)

classReportLR = classification_report(y_test, logpredictions)
print(classReportLR) 

# Create ROC curve of the Log Reg model
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, logpredictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic - Log Reg Model')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

AUC_logreg = label='%0.2f'% roc_auc
print(AUC_logreg)




#RANDOM FOREST
rfmodel = RandomForestClassifier(n_estimators=24) #rfmodel is the function in sklearn
rfmodel.fit(x_train,y_train)
rfpredictions = rfmodel.predict(x_test)
resultRfmodel = accuracy_score(y_test, rfpredictions)   
print('Random Forest accuracy: ', resultRfmodel)


#RAND FOR with CV
rfmodel_cv = RandomForestClassifier()
#train model with cv of 5 
cv_scores = cross_val_score(rfmodel_cv, X, y, cv=10)
#print each cv score (accuracy) and average them
print(cv_scores)
rfmodel_cv = ('RF CV (mean):{}'.format(np.mean(cv_scores))) 
print(rfmodel_cv)

# Sensitivity & specificity
confMatrix_rfmodel = confusion_matrix(y_test, rfpredictions)
print(confMatrix_rfmodel)

TP = confMatrix_rfmodel[1, 1]
TN = confMatrix_rfmodel[0, 0]
FP = confMatrix_rfmodel[0, 1]
FN = confMatrix_rfmodel[1, 0]

sensitivity_rfmodel = TP / float(TP + FN)
print(sensitivity_rfmodel)

specificity_rfmodel = TN / float(TN + FP)
print(specificity_rfmodel)

classReportRF = classification_report(y_test, rfpredictions)
print(classReportRF) 

# Create ROC curve of the Random Forest model
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, rfpredictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic - SVM Model')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

AUC_rfmodel = label='%0.2f'% roc_auc
print(AUC_rfmodel)



# IMPORTANCES (IN RF MODEL)
# Get numerical feature importances
importances = list(rfmodel.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)


# WRITING RESULTS
results = open("/Users/lucyrothwell/Google Drive copy/MSc  Psych - UCL/9. Dissertation (Y2)/Stuttering/*ML scripts (and results)/" + resultFileName, "a")

results.write("K-nearest Neighbours accuracy: " + str(resultKNN))
results.write("\n" + str(KNN_cv))
results.write("\n" + "Sensitivity KNN: " + str(sensitivityKNN))
results.write("\n" + "Specificity KNN: " + str(specificityKNN))
results.write("\n" + "AUC KNN: " + str(AUC_KNN)+ "\n" + str(confMatrixKNN)+ "\n" + str(classReportKNN))

results.write("\n" + "\n" + "SVM accuracy:" + str(resultSVM))
results.write("\n" + str(SVM_cv))
results.write("\n" + "Sensitivity SVM: " + str(sensitivitySVM))
results.write("\n" + "Specificity SVM: " + str(specificitySVM))
results.write("\n" + "AUC SVM: " + str(AUC_SVM)+ "\n" + str(confMatrixSVM)+ "\n" + str(classReportSVM))

results.write("\n" + "\n" + ("Log Reg accuracy: " + str(resultLogreg)))
results.write("\n" + str(logReg_cv))
results.write("\n" + "Sensitivity LogReg: " + str(sensitivity_logreg))
results.write("\n" + "Specificity LogReg: " + str(specificity_logreg))
results.write("\n" + "AUC LogReg: " + str(AUC_logreg) + "\n" + str(confMatrix_logreg)+ "\n" + str(classReportLR))

results.write("\n" + "\n" + ("Random Forest accuracy: " + str(resultRfmodel)))
results.write("\n" + str(rfmodel_cv))
results.write("\n" + "Sensitivity RF: " + str(sensitivity_rfmodel))
results.write("\n" + "Specificity RF: " + str(specificity_rfmodel))
results.write("\n" + "AUC RF: " + str(AUC_rfmodel) + "\n" + str(confMatrix_rfmodel) + "\n" + str(classReportRF))

#RFimportances_write = [(('Variable: {:20} Importance: {}'.format(*pair)) + '\n') for pair in feature_importances];
#results.write("\n" + "\n" + str(RFimportances_write))
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# COPY& PASTE THIS FOR NOW

results.close()
