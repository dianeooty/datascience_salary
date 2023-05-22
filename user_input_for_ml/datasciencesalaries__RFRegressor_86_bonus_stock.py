#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://scikit-learn.org/stable/modules/multiclass.html


# In[1]:


# Import libraries
import pandas as pd
import numpy as np
pd.set_option('max_colwidth', 400)
import requests
import matplotlib.pyplot as plt
import json
import re
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# In[2]:


from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[3]:


salaries_df = pd.read_csv('salaries_cleaned.csv')


# In[ ]:


# from google.colab import files
# uploaded = files.upload()


# In[4]:


salaries_df.head()


# In[5]:


salaries_df.columns


# In[285]:





# In[6]:


salaries_df['company'].value_counts()


# In[7]:


val=salaries_df['company'].value_counts()
print(val[val>25])


# In[8]:


#companies over 25 jobs
comapanies_to_replace=[]
company=salaries_df['company'].value_counts()
company=pd.DataFrame(company)
company.head(20)


# In[9]:


comapanies_to_replace.append(company[company.company<15].index)


# In[ ]:





# In[10]:


#Replaced with other for all companies besides the top 20
for x in comapanies_to_replace:
    salaries_df['company'] = salaries_df['company'].replace(x,"Other")
    
# Check to make sure binning was successful
salaries_df['company'].value_counts()


# In[11]:


year_df=salaries_df['date'].str.split('-', expand=True)


# In[12]:


salaries_df['date']=year_df[0]


# In[13]:


salaries_df['month']=year_df[1]


# In[14]:


salaries_df


# In[15]:


salaries_df['level'].value_counts()


# In[16]:


#companies over 25 jobs
levels_to_replace=[]
levels=salaries_df['level'].value_counts()
levels=pd.DataFrame(levels)
levels.head(20)


# In[17]:


levels_to_replace.append(levels[levels.level<15].index)


# In[18]:


for x in levels_to_replace:
    salaries_df['level'] = salaries_df['level'].replace(x,"Other")
    
# Check to make sure binning was successful
salaries_df['level'].value_counts()


# In[19]:


reduced_df = salaries_df[[ 'company', 'date', 'title', 'totalyearlycompensation', 'gender', 'level',
       'yearsofexperience', 'yearsatcompany', 'latitude', 'longitude', 'month', 'bonus', 'stockgrantvalue']]


# In[20]:


reduced_df


# In[21]:


reduced_df.info()


# In[22]:


reduced_df['date']=reduced_df['date'].astype(int)
reduced_df['month']=reduced_df['month'].astype(int)
reduced_df


# In[23]:


reduced_df.info()


# In[24]:


dummies_df = pd.get_dummies(reduced_df)
dummies_df.head()


# In[25]:


dummies_df.to_csv("dummies_table.csv")


# In[26]:


dummies_df.columns


# In[27]:


# Split our preprocessed data into our features and target arrays
X = dummies_df.drop(columns=["totalyearlycompensation"]).values
y = dummies_df["totalyearlycompensation"].values

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[28]:


# # Split our preprocessed data into our features and target arrays
# X = reduced_df.drop(columns=["totalyearlycompensation"]).values
# y = reduced_df["totalyearlycompensation"].values

# # Split the preprocessed data into a training and testing dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[29]:


# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# In[30]:


X.shape


# In[31]:


#linear regression
# Import required libraries
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LinearRegression


# In[32]:


#random forest
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


#https://stackoverflow.com/questions/52648383/how-to-get-coefficients-and-feature-importances-from-multioutputregressor
#rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=500, random_state=78, max_depth=20))


# In[34]:


rf_model = RandomForestRegressor(n_estimators=500, random_state=78, max_depth=20, n_jobs=-1, verbose=3, oob_score=True)
#rf_model = RandomForestRegressor(n_estimators=500, random_state=7, max_depth=10)


# In[35]:


rf_model = rf_model.fit(X_train_scaled, y_train)


# In[36]:


predictions = rf_model.predict(X_test_scaled)


# In[37]:


plt.scatter( predictions, y_test)
plt.xlabel("predictions")
plt.ylabel("y_test")
plt.show()


# In[38]:


acc_score = rf_model.score(X_test_scaled, y_test)
acc_score
#0.7402413883001011 with bonus
#0.8639637152876211 with stock
#0.8623347451720718 with updated company list


# In[39]:


# import pickle
# #save model
# s = pickle.dumps(rf_model)
# #load model
# clf2 = pickle.loads(s)


# In[40]:


#save the model 
#https://scikit-learn.org/stable/model_persistence.html
from joblib import dump, load
dump(rf_model, 'salary_ml_model.joblib') 


# In[50]:


# have to run inputs throught get dummies_df, maybe set all 700+ values to zero and then just change individual values based on the input
#maybe do throught for loop 
#scales the data
#then take the scaled inputs 

