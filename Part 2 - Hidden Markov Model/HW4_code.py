# BIA-656
# HW 4
# Author: Vivek John D Martins
# SID: 10406763

#Pilgrim Bank. Second Part .


import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("\HW4\pilgrim.csv")

df.describe()

#IMPUTING VALUES______________________________________________________________________________________________________________________________________
df['Age0']= np.where(df['9Age'].isnull(),0,df['9Age'])
df['Inc0']= np.where(df['9Inc'].isnull(),0,df['9Inc'])

#- Substitute missing observations with averages: AgeAvg and IncAvg:
df['AgeAvg']= np.where(df['9Age'].isnull(),df['9Age'].mean(),df['9Age'])
df['IncAvg']= np.where(df['9Inc'].isnull(),df['9Inc'].mean(),df['9Inc'])

#- Include additional dummy variables where 1 if there is data and 0 otherwise: AgeExist and IncExist
df['AgeExist']= np.where(df['9Age'].isnull(),0,1)
df['IncExist']= np.where(df['9Inc'].isnull(),0,1)

#dummy variables for destrict
df['D1100']= np.where(df['9District']==1100,1,0)
df['D1200']= np.where(df['9District']==1200,1,0)

#

#Checking for null values
print pd.value_counts(df['0Profit'].isnull())

df.head()
#Dataset without 0Profit Null Values
df_1 = df[df['0Profit'].isnull() == False]
#______________________________________________________________________________________________________________________________________________________


#5.a. Evaluate the drivers of customer profitability for the year 2000 (Hint: you can evaluate the variables explored for profitability of 1999).
group= df_1[['9Online','AgeAvg','AgeExist','IncAvg','IncExist','9Tenure','D1100','D1200']]
group = sm.add_constant(group)
model5a = sm.OLS(df_1['0Profit'],group)
results5a= model5a.fit()
print(results5a.summary())

#after removing insignificant predictor variables
group= df_1[['9Online','AgeAvg','IncAvg','IncExist','9Tenure']]
group = sm.add_constant(group)
model5a = sm.OLS(df_1['0Profit'],group)
results5a= model5a.fit()
print(results5a.summary())

#5.b. Evaluate if the variable 9Profit should be included in the customer profitability analysis for 2000.
group= df_1[['9Profit','9Online','AgeAvg','IncAvg','IncExist','9Tenure']]
group = sm.add_constant(group)
model5b = sm.OLS(df_1['0Profit'],group)
results5b= model5b.fit()
print(results5b.summary())

#considering onnly imp factors
group= df_1[['9Profit','9Online','IncAvg','9Tenure']]
group = sm.add_constant(group)
model5b = sm.OLS(df_1['0Profit'],group)
results5b= model5b.fit()
print(results5b.summary())

#5.c. Evaluate the drivers of customer profitability for 2000 after adding electronic billpay using OLS and Random Forest regression. Compare these two methods based on their R-squared.
group= df_1[['9Profit','9Online','IncAvg','9Tenure','9Billpay']]
group = sm.add_constant(group)
model5c1 = sm.OLS(df_1['0Profit'],group)
results5c1= model5c1.fit()
print(results5c1.summary())

from sklearn.ensemble import RandomForestRegressor
Rfreg=RandomForestRegressor()
group= df_1[['9Online','AgeAvg','AgeExist','IncAvg','IncExist','9Tenure','D1100','D1200','9Profit','9Billpay']]
results5c2=Rfreg.fit(group,df_1['0Profit'])
Rsqrd=Rfreg.score(group,df_1['0Profit'])
print Rsqrd

#6.a Evaluate the drivers of customer retention for the year 2000. Compare OLS with logistic regression results and indicate which one of these two methods is more appropriate for this problem. Retain takes a value of 0 when 0Profit has a missing observation and 1 otherwise.

#Creating Retains Variable
df['Retain']= np.where(df['0Profit'].isnull(),0,1)

#OLS
group= df[['9Online','AgeAvg','AgeExist','IncAvg','IncExist','9Tenure','D1100','D1200','9Profit','9Billpay']]
group = sm.add_constant(group)
model6a = sm.OLS(df['Retain'],group)
results6a= model6a.fit()
print(results6a.summary())

#considering onnly imp factors
group= df[['9Online','AgeExist','IncAvg','IncExist','9Tenure','9Profit','9Billpay']]
group = sm.add_constant(group)
model6a = sm.OLS(df['Retain'],group)
results6a= model6a.fit()
print(results6a.summary())

#Logisitc Regression
group= df[['9Online','AgeAvg','AgeExist','IncAvg','IncExist','9Tenure','D1100','D1200','9Profit','9Billpay']]
group = sm.add_constant(group)
logit = sm.Logit(df['Retain'],group)
result=logit.fit()
print result.summary()

#6.b. Calculate and rank the odds ratio (exp(coefficients of logistic regression)) of the logistic regression and explain the impact of the 9Online and Billpay variables on customer's retention.
oddsr= np.exp(result.params)#gets odds ratio

oddsr1=pd.DataFrame(oddsr.sort_values(ascending=False))#convert to df
oddsr1.columns= ["Odds Ratio"]#name column
oddsr1["rank"]=np.arange(1,len(oddsr1)+1,1)#adds ranks
print oddsr1


#7. Evaluate the effect of the online channel and billpay on customer's retention using a hidden Markov model (HMM) with the variables 9Online, 9Billpay, 0Online, 0Billpay and the new variable Retain. Retain takes a value of 0 when 0Profit has a missing observation and 1 otherwise. 

#print pd.value_counts(df['0Online'].isnull())

#print pd.value_counts(df['0Billpay'].isnull())

df['0OnlineNA']= np.where(df['0Online'].isnull(),0,df['0Online'])
df['0BillpayNA']= np.where(df['0Billpay'].isnull(),0,df['0Billpay'])
df.head()

from hmmlearn.hmm import GaussianHMM
group= df[['9Online','9Billpay','0OnlineNA','0BillpayNA','Retain']]
model = GaussianHMM(n_components=4, covariance_type="diag").fit(group)
hidden_states = model.predict(group)

for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()

#hdf=pd.DataFrame(hidden_states)
#hdf.describe() #verifying the meaning of the hidden states

#8. Build a transition matrix (online, billpay) from 1999 to 2000 from those different customers' states:those that were online, offline without electronic billpay, online with electronic billpay for 2000, customers who left the bank. Explain the billpay effect on customers' retention.
print("Transition matrix")
print(model.transmat_)
print()

#print("Means and vars of each hidden state")
#for i in range(model.n_components):
#    print("{0}th hidden state".format(i))
#    print("mean = ", model.means_[i])
#    print("var = ", np.diag(model.covars_[i]))
#    print()
