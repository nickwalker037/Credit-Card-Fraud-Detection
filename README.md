# Credit-Card-Fraud-Detection
Using undersampling techniques and logistic regression in order to predict credit card fraud 

This is the Kernel submission for the Kaggle competition "Credit Card Fraud Detection". The dataset contains 28 PCA transformed features of transactions made by credit cards in September 2013 by european cardholders. The dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 total transactions (0.172% of total).

Because of the highly unbalanced nature of the dataset, I used a the confusion matrix to calculate the Precision and Recall of my results. I also used the Undersampling technique in order to take a smaller amount of the normal transactions that occurred and train a logistic regressor based on this. I trained and applied the logistic regressor on all of the data, on only the undersampled data, and then I used the logistic regressor trained on the undersampled data and applied it to all of the data. My recall scores for each were as follows:

The logistic regressor trianed on and applied to all of the data: 0.52
The logistic regressor trained on and applied to only the undersampled data: 0.91
The logistic regressor trained on the undersampled data and applied to all of the data: 0.92

As you can see from my results above, the logistic regressor trained on the undersampled data and applied to all of the data had the best results, with a 92% recall rate. A very good start for applying the undersampling technique on only a logistic regressor. 

To Do:
  - Try different classfication models (SVM's, Decision Trees)
  - Apply K-Fold CV to tune the different models and see which has the best results
  - Figure out the best test/train percentages
  - See which variables are skewed to begin with, and apply various feature manipulation techniques to these
  
