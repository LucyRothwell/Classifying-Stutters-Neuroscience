# Applying Machine Learning to EEG and fNIRS (Neural) Data to Classify Stutters

This is the code used in a psychology research project which classified verbal stutter events by applying machine learning to neural (EEG and fNIRS) data. Participants were males who experience child onset fluency disorder (stuttering). Participants were linked up to fNIRS and EEG neural imaging technology and recorded speaking for around 20 minutes at a time. The audio recordings allowed the stutter events to be marked in time. These markings could then be synced and used to label the  stutters in the fNIRS/EEG data in csv files for use in classification.

Two scripts:

1. A script that labels stutter events on the EEG and fNIRS data stored in csv files. In each participant's EEG/fNIRS csv files, there were appropximately 15,000 rows each representing a 0.1 second neural reading. Of these 15,000 rows, around 3-7% represented stutter events. This script takes as in input the markings from the audio file which shows the start and end times of the stutters (they lasted between 0.2 and 3 seconds) and time-syncs this with the fNIRS and EEG data recordings. It then prints a column of 1s and 0s into the column "stutter" on the fNIRS/EEG csv exports which are then inputted to the machine learning classifiers.

![GitHub Logo](/Users/lucyrothwell/Google_Drive/MSc Psych - UCL/9. Dissertation (Y2)/*Stuttering/Labelling/Labelling output example.png)

2. The machine learning script used to classifiy stutter events (using SVM, KNN, Random Forest and Logistic Regression).
