import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

df = pd.read_csv("...../Credit_Card_Fraud_Data.csv")
#print (df.describe())


X = df.ix[:, df.columns != 'Class']
Y = df.ix[:, df.columns == 'Class']


# Graphing the time of day that fraud occurs:
# Time represents the amount of seconds from when the first transaction occured... so this converts it to hours.
df['Hour'] = df['Time'].apply(lambda x: np.ceil(float(x)/3600) % 24)

# Breaking down legit vs. fraud transactions via a pivot table:
df.pivot_table(values='Amount', index='Hour',columns='Class',aggfunc='count')


def PlotHistogram(df,norm):
    bins = np.arange(df['Hour'].min(),df['Hour'].max()+2)
    plt.figure(figsize=(15,4))
    sns.distplot(df[df['Class']==0.0]['Hour'],
                 norm_hist=norm,
                 bins=bins,
                 kde=False,
                 color='b',
                 hist_kws={'alpha':.5},
                 label='Legit Transactions')
    sns.distplot(df[df['Class']==1.0]['Hour'],
                 norm_hist=norm,
                 bins=bins,
                 kde=False,
                 color='r',
                 hist_kws={'alpha':.5},
                 label='Fradulent Transactions')
    plt.title("Normalized Histogram of Legit/Fraud Transactions Per Hour of the Day")
    plt.xticks(range(0,24))
    plt.legend()
    plt.show()

PlotHistogram(df,True)

df = df.drop(['Hour'], axis=1)


# calculating the percentage of normal vs. fradulent transactions:
Total_Normal_Transactions = len(df[df["Class"]==0])
Total_Fraud_Transactions = len(df[df["Class"]==1])
Percent_Normal_Transactions = Total_Normal_Transactions/(Total_Normal_Transactions + Total_Fraud_Transactions)
Percent_Fraud_Transactions = Total_Fraud_Transactions/(Total_Fraud_Transactions+Total_Normal_Transactions)
print ("% of Transactions That Are Normal: ",Percent_Normal_Transactions*100)
print ("% of Transactions That Are Fraud: ",Percent_Fraud_Transactions*100)
print("----------")
#print("")
# you can see here that the data is clearly unbalanced as 0.17% of records are fraud

"""
# normalize the amount column since it's not in line with the other columns:
df['Amount_Scaled'] = StandardScaler().fit_transform(df['Amount'].reshape(-1,1))
df = df.drop(['Time','Amount'],axis=1)
#print (df.head())
"""
# number of records in the fraud class:
Num_Fraud = len(df[df["Class"]==1])

# indices of fraud and normal cases:
Fraud_Indices = np.array(df[df["Class"]==1].index)
Normal_Indices = np.array(df[df["Class"]==0].index)

# randomly select "n" number of indices of the normal class... n being the total number of fraud cases
Random_Normal_Indices = np.random.choice(Normal_Indices,Num_Fraud, replace=False)
Random_Normal_Indices = np.array(Random_Normal_Indices)

# Append the two indices:
under_sample_indices = np.concatenate([Fraud_Indices,Random_Normal_Indices])

# Under Sampling Dataset:
under_sample_data = df.iloc[under_sample_indices,:]
print ("% of Normal Transactions: ",len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print ("% of Fraudulent Transactions: ",len(under_sample_data[under_sample_data.Class != 0])/len(under_sample_data))
print ("Total number of transactions in resampled data: ", len(under_sample_data))
print("----------")
#print("")



# ---------- Organizing the Data ---------- #

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
print ("Number of Transactions in Train Dataset: ",len(X_train))
print ("Number of Transactions in Test Dataset: ",len(X_test))
print ("Total Number of Transactions: ",len(X_train)+len(X_test))
print("")

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
Y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample = train_test_split(X_undersample,Y_undersample, test_size=0.3, random_state=0)
print ("Number of Transactions in Undersampled Train Dataset: ",len(X_train_undersample))
print ("Number of Transactions in Undersampled Test Dataset: ",len(X_test_undersample))
print ("Total Number of Transactions in Undersampled Dataset: ",len(X_train_undersample)+len(X_test_undersample))







# ------------ Predicting the values using the whole dataset ------------ #

lr_model = LogisticRegression()
lr_model.fit(X_train,Y_train)
pred = lr_model.predict(X_test)

Y_test_legit = len(Y_test[Y_test["Class"]==0])
Y_test_fraud = len(Y_test[Y_test["Class"]==1])

def PlotConfusionMatrix(Y_test,pred,Y_test_legit,Y_test_fraud):

    cfn_matrix = confusion_matrix(Y_test,pred)
    cfn_norm_matrix = np.array([[1.0 / Y_test_legit, 1.0 / Y_test_legit],[1.0/Y_test_fraud,1.0/Y_test_fraud]])
    norm_cfn_matrix = cfn_matrix * cfn_norm_matrix

    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,2,1)
    sns.heatmap(cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')

    ax = fig.add_subplot(1,2,2)
    sns.heatmap(norm_cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')
    plt.show()

    print('----------- Classification Report --- All Data -----------')
    print(classification_report(Y_test,pred))
    
PlotConfusionMatrix(Y_test,pred,Y_test_legit,Y_test_fraud)
    
    

# ------------ Predicting the values of the undersampled dataset ------------ #

lr_model_undersample = LogisticRegression()
lr_model_undersample.fit(X_train_undersample,Y_train_undersample)
pred_undersample = lr_model_undersample.predict(X_test_undersample)

Y_test_legit_undersample = len(Y_test_undersample[Y_test_undersample["Class"]==0])
Y_test_fraud_undersample = len(Y_test_undersample[Y_test_undersample["Class"]==1])


def PlotConfusionMatrixUndersample(Y_test_undersample,pred_undersample,Y_test_legit_undersample,Y_test_fraud_undersample):

    cfn_matrix_undersample = confusion_matrix(Y_test_undersample,pred_undersample)
    cfn_norm_matrix_undersample = np.array([[1.0 / Y_test_legit_undersample, 1.0 / Y_test_legit_undersample],[1.0/Y_test_fraud_undersample,1.0/Y_test_fraud_undersample]])
    norm_cfn_matrix_undersample = cfn_matrix_undersample * cfn_norm_matrix_undersample

    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,2,1)
    sns.heatmap(cfn_matrix_undersample,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
    plt.title('Undersampled Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')

    ax = fig.add_subplot(1,2,2)
    sns.heatmap(norm_cfn_matrix_undersample,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
    plt.title('Undersampled Normalized Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')
    plt.show()

    print('----------- Classification Report --- Undersampled Data -----------')
    print(classification_report(Y_test_undersample,pred_undersample))
    
PlotConfusionMatrixUndersample(Y_test_undersample,pred_undersample,Y_test_legit_undersample,Y_test_fraud_undersample)



# ------------ Apply the undersampled data classfier to the entire dataset: ------------ #

lr_model_all_undersample = LogisticRegression()
lr_model_all_undersample.fit(X_train_undersample,Y_train_undersample)
pred_all_undersample = lr_model_all_undersample.predict(X_test.values)


def PlotConfusionMatrixAllUndersample(Y_test,pred_all_undersample,Y_test_legit,Y_test_fraud):

    cfn_matrix = confusion_matrix(Y_test,pred_all_undersample)
    cfn_norm_matrix = np.array([[1.0 / Y_test_legit, 1.0 / Y_test_legit],[1.0/Y_test_fraud,1.0/Y_test_fraud]])
    norm_cfn_matrix = cfn_matrix * cfn_norm_matrix

    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,2,1)
    sns.heatmap(cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
    plt.title('Undersampled Confusion Matrix on All Data')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')

    ax = fig.add_subplot(1,2,2)
    sns.heatmap(norm_cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
    plt.title('Undersampled Normalized Confusion Matrix on All Data')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')
    plt.show()

    print('----------- Classification Report --- Undersampled Classifier on All Data -----------')
    print(classification_report(Y_test,pred_all_undersample))
    
PlotConfusionMatrixAllUndersample(Y_test,pred_all_undersample,Y_test_legit,Y_test_fraud)

