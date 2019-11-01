#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:55:57 2019

@author: lucyrothwell
"""
import os # Function to see currernt working directory
print(os.getcwd()) 
import pandas as pd # For  data manipulation
import numpy as np # For ML algorithms


# ONCE EVENTS HAVE BEEN LABELLED ON THE AUDIO FILES IN AUDICATY, THEY CAN BE 
# EXPORTED TO TXT FILE. COPY THEM INTO A CSV. THIS PROGRAMME TAKES THAT CSV, 
# (ALONG WITH THE FNIRS/EEG DATA CSVs) AND OUTPUTS A VERTICAL LIST OF 1s and 0s. 
# THIS LIST CAN BE COPIED DIRECTLY INTO THE "STUTTER" COLUMN ON FNIRS/EEG DATA
# CSV TO SHOW WHICH ROWS REPRESENT STUTTERS (1) VERSUS NON-STUTTERS (0).

# LOAD RELEVANT FILES

# Loads the fNIRS or EEG csv file. *NOTE*: This programme assumes column 1 of this sheet is the timings)
fNIRS_EEG = '/Users/lucyrothwell/Google Drive copy/MSc  Psych - UCL/9. Dissertation (Y2)/Stuttering/*Data - EEG/PWS EEG/3. Processed (.csv)/018_PWS_Social (P) - CH - C(430).csv'
fNIRS_EEG = pd.read_csv(fNIRS_EEG, encoding='utf-8', skiprows=0)

# Load "labels" data (the label start and finish times, exported from audacity)
labels_raw = "/Users/lucyrothwell/Google Drive copy/MSc  Psych - UCL/9. Dissertation (Y2)/Stuttering/*Data - Audio/PWS/018PWSsocial - LABELS.csv"
labels = pd.read_csv(labels_raw, encoding='utf-8', skiprows=1)

# Create empty stutters list txt file (for listing 0s and 1s to represent stutters)
os.chdir('/Users/lucyrothwell/Google Drive copy/MSc  Psych - UCL/9. Dissertation (Y2)/Stuttering/*Data - Audio/PWS/')
stuttersList01 = open("018PWS_social - LABELS_01.csv","a")

# ^ ONCE THE DATA FILES ABOVE HAVE BEEN CREATED, YOU CAN JUST PRESS RUN.
# YOU SHOULD  GET THE  LIST 1s and 0s FOR COPY INTO THE "STUTTER COLUMN" IN 
# THE EEG/FNIRS DATA CSV

# PSEUDO
# Function for extracting each decisecond in the stutter ranges extracted 
# from Audacity 
# > Take start time of R0,C0 (row 0, column 0)
# > Take end time of R0,C1 (row 0, column 1)
# > Extract range between R0,C0 and R0,C1, and write (apend) it into "labels" doc

# Create labels variables in matrix
labelsMatrix = np.matrix(labels)

def addRange (labels, labelsMatrix):
    labels = open(labels_raw, "a")
    row = 0
    for i in labelsMatrix:
        x = round(labelsMatrix[row,0],1)
        y = labelsMatrix[row,1]
        labels.write("\n" + str(x))
        while x < y:
            x = x + 0.1
            x = round((x),1)
            labels.write("\n" + str(x))
        row = row + 1
    labels.close() 

addRange(labels, labelsMatrix)

# Updating the labels variable with the updated labels document
labels = labels_raw
labels = pd.read_csv(labels, encoding='utf-8', skiprows=1)

# PSEUDO
# Function for turning stutter labels into a column of 1s and 0s
# Take R1-C0 in fNIRS_EEG
# If the number exists in labels, then print "1" in R1-C0 in stutterList.
# If not print "0" in R1-C1 in stutterList and go to R2-C0 in fNIRS_EEG. 
# Then take R2-C1 in fNIRS_EEG
# If the number exists in Labels, then print "1" in R2-C0 in stutterList.
# If not print "0" in R2-C0 in stutterList and go to R3-C0 in fNIRS_EEG. 

# Create stuttersList variable
labelsC1 = list(labels.iloc[:,0]) # Column 1 - containing stutter start times. 
# ^ Must be list so it can be read by addEvents function.
#labelsC2 = labels.iloc[:,1] # Column 2 - containing stutter end times.

# Create a "time column" variable "fNIRS_EEG_CTime variable" (i.e., index the 
#fNIRS_EEG_CTime = fNIRS_EEG.iloc[:,0] # Column 1 (time column) of fNIRS_EEG doc 

# Create a column of 1s and 0s, correspondong to the fNIRS/EEG time points, 
# based on the labels doc
def addEvents(labelsC1, stuttersList01, fNIRS_EEG_CTime):
    stuttersList01
    for i in fNIRS_EEG_CTime:
        if i in labelsC1:
            stuttersList01.write("\n" + "1")
        else:
            stuttersList01.write("\n" + "0")
        print(i)
    stuttersList01.close()

# Execute function "addEvent" using variables created above
addEvents(labelsC1, stuttersList01, fNIRS_EEG_CTime)

# The document stutterList01 should now be a column of 1s and 0s, with 
# the same number of rows as your fNIRS_EEG data file (15k or so). This
# can now be pasted into your fNIRS_EEG data file as labels for supervised
# learning . NOTE - A line may have been skipped at the top of the 
# stuttersLIst01 doc - delete this to ensure numbers are alligned.

