#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lucyrothwell
"""
import pandas as pd # For  data manipulation
import numpy as np # For ML algorithms

# Load your files and sample rate

# Loads the data csv file.
# *NOTE*: This programme assumes column 1 of this sheet is the timings
data_file_name = 'data_time_series_18PWS(S).csv' # This will also be
# # used for your labelled output file
data = pd.read_csv(data_file_name, encoding='utf-8', skiprows=0)

# Load event start and finish times doc. I.e., your csv where each row holds [start_time, end_time] of events.
durations = pd.read_csv("durations_events_18PWS(S).csv", encoding='utf-8', skiprows=1)

# Your sample rate (ie seconds, deciseconds)
sample_rate = 0.1

# ************ NOTHING ELSE NEEDS EDITED FROM HERE ***************
#            (unless further customisation is needed)

# Function for extracting each decisecond in the stutter ranges extracted
# PSEUDO:
# > Take start time of R0,C0 (row 0, column 0)
# > Take end time of R0,C1 (row 0, column 1)
# > Extract range between R0,C0 and R0,C1, and write (apend) it into "labels" doc

def addRange (durations):
    print("Running...")
    global durations_split # A new df which will hold the data split into secs/decisecs
    durations_split = pd.DataFrame()
    durationsMatrix = np.matrix(durations)  # Put durations in matrix - allows us to iterate
    row = 0
    for i in durationsMatrix: # For each row in labels durations
        x = round(durationsMatrix[row,0],1) # Start time
        y = round(durationsMatrix[row,1],1) # End time
        durations_split = durations_split.append(pd.DataFrame([x]))
        while x < y:
            x = x + sample_rate
            x = round((x),1)
            durations_split = durations_split.append(pd.DataFrame([x]))
        row = row + 1
    return durations_split

addRange(durations)

# PSEUDO
# Function for turning stutter labels into a column of 1s and 0s
# Take R1-C0 in data
# If the number exists in labels, then print "1" in R1-C0 in stutterList.
# If not print "0" in R1-C1 in stutterList and go to R2-C0 in data.
# Then take R2-C1 in data
# If the number exists in Labels, then print "1" in R2-C0 in stutterList.
# If not print "0" in R2-C0 in stutterList and go to R3-C0 in data.

# Create a variable containing the sampling times of your data
data_time_col = data.iloc[:,0]
data_time_col = pd.DataFrame([data_time_col])
data_time_col = data_time_col.Tcon

# Function to create a column of 1s and 0s, corresponding to the sampling times
def addEvents(data_time_col):
    labels_01 = open("0_1_list " + data_file_name, "a") # Create new csv to record the labels
    for i in data_time_col.values:
        if i in durations_split.values:
            labels_01.write("1" + "\n")
        else:
            labels_01.write("0" + "\n")
    global labels_01_df
    labels_01_df = pd.read_csv("0_1_list " + data_file_name)
addEvents(data_time_col)

# Insert labels column into data dataframe
data.insert(loc=0, column="labels", value = labels_01_df)
data.to_csv(data_file_name + " - LABELLED.csv") # Convert dataframe back to csv
print("Done!")

# You should now have a file in your directory titled "your original data file name - LABELLED".
# (NOTE 1: The file will have an extra column at the beginning (col[0]); it can be deleted.
# (NOTE 2: The "0_1_list..." csv files can also be deleted as are no longer needed).

