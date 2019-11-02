# Applying Machine Learning to EEG and fNIRS (Neural) Data to Predict Stutters

Code used in a psychology research project aimed at classifiying verbal stutter events by applying machine learning to neural (EEG and fNIRS) data. Participants were males who stutter. Participants were linked up to fNIRS and EEG neural imaging technology and recorded speaking for around 20 minutes at a time. The audio recordings allowed the stutter events to be marked in time. These markings could then be synced and used to label the fNIRS/EEG data for use in classification.

Two scripts:

1. A script that labels stutter events on the EEG and fNIRS data stored in csv files, using exported markings from audio recordings of the participants speaking. In each participant's EEG/fNIRS csv files, there were appropximately 15,000 rows each representing a 0.1 second neural reading. Of these 15,000 rows, around 5% represented stutter events. This script takes the markings from the audio file which shows when the stutters started and ended (they lasted between 0.2 and 3 seconds) and syncs this with the fNIRS and EEG data. It then uses this to print a column of 1s and 0s into the column "stutter" on the fNIRS/EEG csv exports which are then inputted to the machine learning classifiers.

2. The machine learning script used to classifiy stutter events (using SVM, KNN, RandomForest and Logistic Regression).
