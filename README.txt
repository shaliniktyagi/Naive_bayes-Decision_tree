*****************************************************************************************************************
*                                                                                                               *
*                      This is the read me file for the course work submission code.                            *
*                                                Date: 26th November 2018                                       *
*                                                                                                               *
*****************************************************************************************************************
                                     
                             ***************Version: Matlab R2018b***************


Coursework Title: Comparison of Naive Bayes and Decision Trees for Classification of Gamma Rays, Authors: Vanessa Do and Shalini Tyagi

Data Source: https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope

Original Owner: 
R. K. Bock 
Major Atmospheric Gamma Imaging Cherenkov Telescope project (MAGIC) 
http://wwwmagic.mppmu.mpg.de 
rkb '@' mail.cern.ch 

Donor: 
P. Savicky 
Institute of Computer Science, AS of CR 
Czech Republic 
savicky '@' cs.cas.cz 

*******************************************************************************************************************

What is this: This code is a combination of machine learning tools such as Decision Tree and Naive Bayes algorithms.

In first section: % LOAD THE DATA AND PREPARE TRAINING AND TEST DATASETS %, 

The code loads the data set from the local directory. if the directory is not the working directory then code will 
suggest to change the directory. The code then split the dataset into test and training sets, using stratified sampling
 to maintain the same class balance in the training and test sets. Further to this it calculates summary statistics and 
correlation coefficients. Creates target and predictor tables for the training and testing.

In second section: % TRAINING THE DECISION TREE % 

The code trains Decision Trees with different combinations of hyper parameters and cross-validate with 10 folds, then 
calculate the cross validation error to find the set of hyper parameters that yield a Decision Tree with the lowest cross 
validation error. It uses hyperparameter to vary for grid search and initialise variables for the grid search loop.

The code run through multiple iterations, modifying the hyper parameters each time to build Decision Tree models. 

It plots a chart to visualise how varying the tree splits and leaf size affects cross validation error (using all predictors
and deviance as the split criteria), creates a boxplot to visualise the distribution of cross val errors by the number of 
features sampled. It also creates a histogram to visualise the distribution of cross val errors by split criteria.

Then it finds the row index of the Decision Tree model with the lowest cross validation error. It assigns the Decision Tree 
model with the lowest cross validation error to 'bestDT' and list the hyperparameter values of this model. At last it plots
 a chart to identify the most important features for discriminating between classes.

The code also records the run time for this section. The code gives an indication of the grid search number to know what stage
is the code at.


In third section: % TRAINING THE NAIVE BAYES CLASSIFIER % 

This part of the code trains Naive Bayes Classifier models by varying the hyper parameters for Predictor Distribution and
 Kernel Smoothing Method. At first it initialise variables for building models, the chosen number of trials are 10 (suggested) 
which means it runs 10 number of iterations for all models. If more number of iterations are required we can change the value to
 a higher number.

First of all a hyper parameter was chosen to run to suggest the best distribution parameters. This has been kept in comments as 
after this further parameters were tested from the chosen distribution that was indicated by hyper parameters.

Then the code runs through multiple trials to find the cross validation errors for the different models such as: normal distribution,
kernel with normal smoothing, kernel with box smoothing, kernel with epanechnikov smoothing, kernel with triangle smoothing, 
and finally kernel with priors.

Then the code extracts the model with the lowest cross validation error. The code also records the run time for this section. 


In final and fourth Section:  % FINAL TEST % 

The code runs the best Decision Tree Model and the best Naive Bayes classifier model on the test data. It 
converts the table of test targets to arrays for creating the confusion matrix of the best Decision Tree Model and the best 
Naive Bayes classifier model

The code finally plots the ROC curves and calculates the area under the curve (AUC). At last the code calculates the accuracy of the 
models.


What it contains:

This submission contains the code as explained above. Also it contains the associated data set and a poster explaining the results.


How to use:

Please copy the code and the data set in working directory in Matlab R2018b. Run the code by run command. The code may take approximately 
1.5-2 Hrs.

