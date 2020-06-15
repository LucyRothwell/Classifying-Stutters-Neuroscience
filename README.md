# The question

Can the occurence of verbal stutters be classified using neural readings?

# Overview

This repository contains code used in a neuroscience research project which aimed to classify verbal stutter events by applying machine learning to neural (EEG and fNIRS) data. It also contains the full study write up (called [Thesis paper](https://github.com/LucyRothwell/Classifying-Stutters-Neuroscience/blob/master/Thesis%20paper.pdf)). Grade for project: 80% (dissertation for MSc in Psychology Research Methods, UCL).

# The data

4 participants each with 15,000 samples (0.1 second readings) of fNIRS neural readings and 15,000 (0.1 second readings) of EEG neural readings. Labelled.

Participants were males who experience child onset fluency disorder (stuttering). Participants were linked up to fNIRS and EEG neural imaging technology and recorded speaking for around 20 minutes at a time. The audio recordings allowed the stutter events to be time-labelled using Audacity software (see script below). These markings could then be used to label the stutters in the fNIRS/EEG csv files, for use in classification. 

# The code

1. **A [script](https://github.com/LucyRothwell/Classifying-Stutters-Neuroscience/blob/master/Labelling_Script.py) that labels stutter events (1 or 0) on the EEG and fNIRS readings stored in csv files.** In each participant's EEG/fNIRS csv files, there were appropximately 14,000 rows each representing a 0.1 second neural reading. Of these 14,000 rows, around 5% represented stutter events. This script takes as input, the markings from the audio file showing the start and end times of the stutters (they lasted between 0.2 and 3 seconds), and time-syncs these with the fNIRS and EEG data recordings. It then prints a column of 1s and 0s, which can be inserted into the y column "stutter" on the fNIRS/EEG csv exports, meaning each row of data (0.1 second neural reading) is now labelled as being a stutter (1) or not a stutter (0). This labelled csv is then inputted to the machine learning classifiers. This process is explained in detail in the code comments.

2. **The machine learning [script](https://github.com/LucyRothwell/Classifying-Stutters-Neuroscience/blob/master/ML%20script%20(SVM%2C%20RF%2C%20KNN%2C%20LOGR).py)** used to classifiy stutter events (using SVM, KNN, Random Forest and Logistic Regression). Explained in the code comments.

# The results

New investigation of overfitting is currently underway (June 2020). To be updated shortly.

<br>
<br>
<br>
<br>

##########################################################################################


# Technical Abstract

**Objective**

The aims of this study were 1) to determine whether supervised machine learning classifiers can be
used to classify stutter events from neural data, and 2) to investigate best practices for doing so in terms of optimal machine learning classifiers (for example random forest, support vector machine), data structures (high volumes of data versus balanced data) and data types (for example independent EEG components (ICs) versus channel data as predictor variables). In being able to classify stutters that have already occurred, it is hoped science can move towards real-time prediction of stutters (in advance of their occurrence), and as such, prediction-based interventions (using stimulation such as tDCS).

**Sample**

The baseline data set comprised of concurrent fNIRS, EEG and audio data from four male participants who displayed overt audible symptoms of childhood onset fluency disorder (COFD). Each participant carried out two verbal tasks, generating a total of 8 separate data sets (each of around fourteen thousand 0.1 second samples).

**Method**

Four machine learning algorithms (K-Nearest Neighbour, Support Vector Machine, Logistic Regression and Random Forest) were applied to several different compositions of the data including large volumes of data (8 data sets combined) with unbalanced classes (around 10% stutter events and 90% non-stutter events), and smaller sets of balanced data (50% stutter and 50% non-stutter). For EEG, data sets were also compared in which the machine learning features were either independent components generated through ICA (independent component analysis), or EEG channels (the pre-processed but un-transformed readings taken from the electrodes).

**Findings**

The balanced data (50% stutter, 50% non-stutter) performed better across both classes and so was used throughout. The best performing machine learning algorithm was random forest on the fNIRS data which correctly classified stutters versus non-stutters with an accuracy (measured by AUC) of 0.90. The third joint highest performers were the KNN and the SVM on the EEG IC data and the SVM on the full EEG set of EEG channel data, all with an AUC 0.67. This suggests machine learning can classify stutters through fNIRS data which is a critical step towards prediction-based intervention. Stutters could potentially be classified through EEG data with some adjustments. These are explained in the discussion section.

