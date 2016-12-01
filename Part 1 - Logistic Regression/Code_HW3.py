# BIA-656
# HW 3
# Author: Vivek John D Martins
# SID: 10406763

#Pilgrim Bank. First part.

import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv("C:\Users\Vivek\SkyDrive\Documents\Stevens Inst of Tech\Spring 2016\BIA 656 - Stat Learn & Analytics\Assignments\HW3\pilgrim.csv")

##Exploratory Data Analysis
df.head()

df.describe()
df.dtypes

#------------------------------------------------------------------------------------------------------------------------------------
#1. Calculate average customer profitability with 95% confidence level

#1999
df['9Profit'].mean()


from scipy import stats
conf_int = stats.norm.interval(0.95, loc=df['9Profit'].mean(), scale=df['9Profit'].std(ddof=1)/np.sqrt(len(df['9Profit'])))

print 'average customer profitability with 95% confidence level in 1999 '
print 'Average:',df['9Profit'].mean()
print 'CI:',conf_int

#2000
df_1 = df[df['0Profit'].isnull() == False]
conf_int_1 = stats.norm.interval(0.95, loc=df_1['0Profit'].mean(), scale=df_1['0Profit'].std(ddof=1)/np.sqrt(len(df_1['0Profit'])))

print 'average customer profitability with 95% confidence level in 2000'
print 'Average:',df_1['0Profit'].mean()
print 'CI:',conf_int_1

#------------------------------------------------------------------------------------------------------------------------------------

#2.a. Evaluate if online channel has a significant impact on 1999 profitability (9Profit).
group= df['9Online']
group = sm.add_constant(group)
model2a = sm.OLS(df['9Profit'],group)
results2a = model2a.fit()
print(results2a.summary())

#2.b. Does age help to explain if online channel has a significant impact on 1999 profitability?

df_2 = df[df['9Age'].isnull() == False]
group= df_2[['9Online','9Age']]
group = sm.add_constant(group)
model2b = sm.OLS(df_2['9Profit'],group)
results2b = model2b.fit()
print(results2b.summary())

#------------------------------------------------------------------------------------------------------------------------------------
#3. To adjust for missing observations in the case of the variables 9Age and 9Inc (income), create the following variables::
#- Substitute missing observations with zeros: Age0 and Inc0:

df['Age0']= np.where(df['9Age'].isnull(),0,df['9Age'])
df['Inc0']= np.where(df['9Inc'].isnull(),0,df['9Inc'])

#- Substitute missing observations with averages: AgeAvg and IncAvg:
df['AgeAvg']= np.where(df['9Age'].isnull(),df['9Age'].mean(),df['9Age'])
df['IncAvg']= np.where(df['9Inc'].isnull(),df['9Inc'].mean(),df['9Inc'])

#- Include additional dummy variables where 1 if there is data and 0 otherwise: AgeExist and IncExist
df['AgeExist']= np.where(df['9Age'].isnull(),0,1)
df['IncExist']= np.where(df['9Inc'].isnull(),0,1)

#To test for bias of missing data, evaluate if missing data has an effect on profitability analysis: 
#3a. Evaluate the effect of online channel on 1999 profits when Age0 is included.
group= df[['9Online','Age0']]
group = sm.add_constant(group)
model3a = sm.OLS(df['9Profit'],group)
results3a = model3a.fit()
print(results3a.summary())

#3b. Evaluate if adjusting missing data using Age0 or AgeAvg is relevant. In both cases, it is still necessary to 
#include the additional variable AgeExist to control for the missing data
group= df[['9Online','Age0','AgeExist']]
group = sm.add_constant(group)
model3b1 = sm.OLS(df['9Profit'],group)
results3b1 = model3b1.fit()
print(results3b1.summary())

group= df[['9Online','AgeAvg','AgeExist']]
group = sm.add_constant(group)
model3b2 = sm.OLS(df['9Profit'],group)
results3b2 = model3b2.fit()
print(results3b2.summary())

#3c. Repeat above steps with income. Evaluate if adjusting missing data using Inc0 or IncAvg is relevant.
# Include AgeExist and AgeAvg in the calculations.
group= df[['9Online','AgeAvg','AgeExist','Inc0','IncExist']]
group = sm.add_constant(group)
model3c1 = sm.OLS(df['9Profit'],group)
results3c1 = model3c1.fit()
print(results3c1.summary())

group= df[['9Online','AgeAvg','AgeExist','IncAvg','IncExist']]
group = sm.add_constant(group)
model3c2 = sm.OLS(df['9Profit'],group)
results3c2 = model3c2.fit()
print(results3c2.summary())

#------------------------------------------------------------------------------------------------------------------------------------
#4. Evaluate if online channel has a significant impact on 1999 profitability after controlling for demographic variables:
# age, income, tenure, and geographic district.

#To evaluate the latter, create dummy variables D1100 and D1200 for districts 1100 and 1200 respectively from the variable 9District.
df['D1100']= np.where(df['9District']==1100,1,0)
df['D1200']= np.where(df['9District']==1200,1,0)

group= df[['9Online','AgeAvg','AgeExist','IncAvg','IncExist','9Tenure','D1100','D1200']]
group = sm.add_constant(group)
model4 = sm.OLS(df['9Profit'],group)
results4= model4.fit()
print(results4.summary())

# Checking for correlations to explain in report
#df.corr()