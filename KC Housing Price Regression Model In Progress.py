#!/usr/bin/env python
# coding: utf-8

# # KC House Data Analysis and Modeling

# #### The purpose of this exercise is to gain experience using python pandas, matplotlib, and scikit learn libraries. The data is downloaded from https://www.kaggle.com/harlfoxem/housesalesprediction/data. Based both on the description of the data from Kaggle, the goal will be to create a regression model.

# ### Import Necessary Libraries

# In[6]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from sklearn import linear_model
import seaborn as sns 


# ### Import Data

# In[4]:


df = pd.read_csv("C:/python/Data/kc_house_data.csv")


# ### Take an initial look at the data

# In[42]:


#Look at number of rows and columns
df.shape
#(21613, 21)

#Take a look at the data
df.head()

#Get a list of column names
#for col in df.columns:
#    display(col)
#'id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view'
#'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long','sqft_living15'
#'sqft_lot15'

#Check for missing or blank data
#df.isnull().sum()
#No blank rows or missing values

#Check for duplicates by id
duplicateRowsDF = df[df.duplicated()]
duplicateRowsDF
#No duplicates

#Look at data structure of each column
df.dtypes
'''
id                 int64
date              object
price            float64
bedrooms           int64
bathrooms        float64
sqft_living        int64
sqft_lot           int64
floors           float64
waterfront         int64
view               int64
condition          int64
grade              int64
sqft_above         int64
sqft_basement      int64
yr_built           int64
yr_renovated       int64
zipcode            int64
lat              float64
long             float64
sqft_living15      int64
sqft_lot15         int64
'''

df.head()


# ### Create some basic descriptive statistics of the data

# In[34]:


# A glossery of terms can be found at https://info.kingcounty.gov/assessor/esales/Glossary.aspx?type=r

# Descriptions
perc = [.2, .4, .6, .8]
df[['price','bedrooms','bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'grade', 'sqft_above', 
    'sqft_basement','sqft_living15', 'sqft_lot15']].describe(percentiles = perc)


# In[20]:


g = sns.pairplot(df, vars = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
                            'condition', 'grade', 'sqft_above', 'sqft_basement'])
g.fig.suptitle("Pair Plots", y=1.08)


# #### There is some linear relationship between several of the variables and price. Generally, the number of bedrooms, bathrooms, and square feet appear to be positively associated with price. We might want to explore a potential interaction between number of bedrooms and number of bathrooms. Grade is positively associated with price and we might want to check if there is a small polynomial relationship.

# #### Since location might affect prices, a new region column is created based on King County Geographies

# In[52]:


#Combine zip codes into four regions: North, South, East, and Seattle
# https://www.communitiescount.org/king-county-geographies

def get_region(row):
    East = [98004,98005,98006,98007,98008,98009,98014,98015,98019,98024,98027,98029,98033,98034,
            98039,98040,98045,98050,98052,98053,98065,98073,98074,98075,98077,98083,98224,98288]
    North = [98004,98005,98006,98007,98008,98009,98011,98014,98015,98019,98024,98027, 98028, 98029,98033,98034,
             98039,98040,98045,98050,98052,98053,98065,98072,98073,98074,98075,98083,98224,98288]
    Seattle = [98101,98102,98103,98104,98105,98106,98107,98108,98109,98111,98112,98113,98114,98115,
               98116,98117,98118,98119,98121,98122,98124,98125,98126,98127,98129,98131,98133,98134,
               98136,98138,98139,98141,98144,98145,98146,98148,98154,98155,98158,98160,98161,98164,
               98165,98166,98168,98170,98174,98175,98177,98178,98181,98185,98188,98190,98191,98194,
               98195,98198,98199]
    South = [98001,98002,98003,98010,98013,98022,98023,98025,98030,98031,98032,98035,98038,98042,
             98047,98051,98055,98056,98057,98058,98059,98062,98063,98064,98070,98071,98089,98092,98093]
    if row['zipcode'] in East:
        val = "East"
    elif row['zipcode'] in North:
        val = "North"
    elif row['zipcode'] in Seattle:
        val = "Seattle"
    elif row['zipcode'] in South:
        val = "South"
    else:
        return "Needs Assignment"
    return val

#Create new column with region information
#df1 = df.copy()
#df1['region'] = df1.apply(get_region, axis = 1)
#df1.head()

#Check for zipcodes that were not assigned
#needs_assignment = df1.groupby(['region'])[['region','zipcode']]
#display(needs_assignment.get_group('Needs Assignment'))

#Once missing zip codes are assigned, comment out above code

df['region'] = df.apply(get_region, axis = 1)
df.head()
    


# In[55]:


h = sns.pairplot(df, hue = 'region', vars = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade'])
h.fig.suptitle("Pair Plots\nBy Region", y=1.08)


# In[ ]:




