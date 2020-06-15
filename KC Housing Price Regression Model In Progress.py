#!/usr/bin/env python
# coding: utf-8

# # KC House Data Analysis and Modeling

# #### The purpose of this exercise is to gain experience using python pandas, matplotlib, and scikit learn libraries. The data is downloaded from https://www.kaggle.com/harlfoxem/housesalesprediction/data. Based both on the description of the data from Kaggle, the goal will be to create a regression model.

# ### Import Necessary Libraries

# In[232]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import seaborn as sns 
import scipy.stats as ss
import statsmodels.api as sm
from sklearn import (datasets, neighbors,
                    model_selection as skms,
                    linear_model, 
                    metrics)


# ### Import Data

# In[2]:


df = pd.read_csv("C:/python/Data/kc_house_data.csv")


# ### Take an initial look at the data

# In[3]:


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

# In[4]:


# A glossery of terms can be found at https://info.kingcounty.gov/assessor/esales/Glossary.aspx?type=r

# Descriptions
perc = [.2, .4, .6, .8]
df[['price','bedrooms','bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'grade', 'sqft_above', 
    'sqft_basement','sqft_living15', 'sqft_lot15']].describe(percentiles = perc)


# In[5]:


g = sns.pairplot(df, vars = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
                            'condition', 'grade', 'sqft_above', 'sqft_basement'])
g.fig.suptitle("Pair Plots", y=1.08)


# #### There is some linear relationship between several of the variables and price. Generally, the number of bedrooms, bathrooms, and square feet appear to be positively associated with price. We might want to explore a potential interaction between number of bedrooms and number of bathrooms. Grade is positively associated with price and we might want to check if there is a small polynomial relationship.

# In[208]:


plt.scatter(df['lat'], df['long'], label='Lat/Long Coords', color = 'b', marker = 'o', s = 1)

plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Plot of House Locations')


# #### Would it be useful to divide King County into regions? Done for practice here.

# In[6]:


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
    


# In[165]:


g1 = sns.pairplot(df, vars = ['price', 'lat', 'long'])
g1.fig.suptitle("Pair Plots", y=1.08)


# In[7]:


h = sns.pairplot(df, hue = 'region', vars = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade'])
h.fig.suptitle("Pair Plots\nBy Region", y=1.08)


# #### Check correlations

# In[131]:


corr = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'condition', 'grade']].corr()
corr
#sns.heatmap(corr)


# In[8]:


dfgbmean = df.groupby(['region']).mean()
display(dfgbmean)

dfgbsd = df.groupby(['region']).std()
display(dfgbsd)

dfgbmed = df.groupby(['region']).median()
display(dfgbmed)


# In[9]:


import statistics
print('Mean: ', statistics.mean(df['price']))
print('Std Dev: ', statistics.stdev(df['price']))


# 
# #### Look at several distributions relative to a histogram

# In[91]:


#The x axis is artificially constrained to $2 million. Prices range up to about $8 million.

from scipy.stats import weibull_min
from scipy.stats import norm
from scipy.stats import lognorm
from matplotlib.patches import Rectangle

n, bins, patches = plt.hist(df['price'], 500, density = 1, facecolor = 'b', alpha = .75)

#Overlay distributions that might be an appropriate fit
x = np.linspace(df['price'].min(), df['price'].max(), 100)

#Weibull Distribution
shape, loc, scale = weibull_min.fit(df['price'], floc=0)
plt.plot(x, weibull_min(shape, loc, scale).pdf(x), color = 'g')

#Normal Distribution
shape, loc = norm.fit(df['price'])
plt.plot(x, norm(shape, loc).pdf(x), color = 'r')

#Lognormal Distribution
shape, loc, scale = lognorm.fit(df['price'], floc=0)
plt.plot(x, lognorm(shape, loc, scale).pdf(x), color = 'y')

plt.xlabel('Price')
plt.ticklabel_format(style = 'plain')
plt.xticks(rotation='vertical')
plt.ylabel('Probability')
plt.title('Histogram of All Prices\nAnd Some Distributions')
#plt.text(700000, 0.00000175, r'$\mu=540088,\ \sigma=367127$')
plt.axis([0,2000000,0,0.0000025])

handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in ['b', 'r', 'y', 'g']]
labels= ["Price","Normal", "Lognormal", "Weibull"]
plt.legend(handles, labels)


# #### Divide the data set into regions

# In[80]:


dfeast = df[df.region == 'East'].copy()
dfnorth = df[df.region == 'North'].copy()
dfseattle = df[df.region == 'Seattle'].copy()
dfsouth = df[df.region == 'South'].copy()


# #### Check that the price distribution holds across all three regions (note different parameters are used for each region)

# In[158]:


n, bins, patches = plt.hist(dfeast['price'], 100, density = 1, facecolor = 'b', alpha = .75)

#Overlay distributions that might be an appropriate fit
x = np.linspace(dfeast['price'].min(), dfeast['price'].max(), 100)

#Weibull Distribution
shape, loc, scale = weibull_min.fit(dfeast['price'], floc=0)
plt.plot(x, weibull_min(shape, loc, scale).pdf(x), color = 'g')

#Normal Distribution
shape, loc = norm.fit(dfeast['price'])
plt.plot(x, norm(shape, loc).pdf(x), color = 'r')

#Lognormal Distribution
shape, loc, scale = lognorm.fit(dfeast['price'], floc=0)
plt.plot(x, lognorm(shape, loc, scale).pdf(x), color = 'y')

plt.xlabel('Price')
plt.ticklabel_format(style = 'plain')
plt.xticks(rotation='vertical')
plt.ylabel('Probability')
plt.title('Histogram of Region East Prices\nAnd Some Distributions')
#plt.text(700000, 0.00000175, r'$\mu=540088,\ \sigma=367127$')
plt.axis([0,2000000,0,0.0000025])

handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in ['b', 'r', 'y', 'g']]
labels= ["Price","Normal", "Lognormal", "Weibull"]
plt.legend(handles, labels)


# In[157]:


n, bins, patches = plt.hist(dfnorth['price'], 50, density = 1, facecolor = 'b', alpha = .75)

#Overlay distributions that might be an appropriate fit
x = np.linspace(dfnorth['price'].min(), dfnorth['price'].max(), 100)

#Weibull Distribution
shape, loc, scale = weibull_min.fit(dfnorth['price'], floc=0)
plt.plot(x, weibull_min(shape, loc, scale).pdf(x), color = 'g')

#Normal Distribution
shape, loc = norm.fit(dfnorth['price'])
plt.plot(x, norm(shape, loc).pdf(x), color = 'r')

#Lognormal Distribution
shape, loc, scale = lognorm.fit(dfnorth['price'], floc=0)
plt.plot(x, lognorm(shape, loc, scale).pdf(x), color = 'y')

plt.xlabel('Price')
plt.ticklabel_format(style = 'plain')
plt.xticks(rotation='vertical')
plt.ylabel('Probability')
plt.title('Histogram of Region North Prices\nAnd Some Distributions')
#plt.text(700000, 0.00000175, r'$\mu=540088,\ \sigma=367127$')
plt.axis([0,2000000,0,0.0000035])

handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in ['b', 'r', 'y', 'g']]
labels= ["Price","Normal", "Lognormal", "Weibull"]
plt.legend(handles, labels)


# In[163]:


n, bins, patches = plt.hist(dfseattle['price'], 200, density = 1, facecolor = 'b', alpha = .75)

#Overlay distributions that might be an appropriate fit
x = np.linspace(dfseattle['price'].min(), dfseattle['price'].max(), 100)

#Weibull Distribution
shape, loc, scale = weibull_min.fit(dfseattle['price'], floc=0)
plt.plot(x, weibull_min(shape, loc, scale).pdf(x), color = 'g')

#Normal Distribution
shape, loc = norm.fit(dfseattle['price'])
plt.plot(x, norm(shape, loc).pdf(x), color = 'r')

#Lognormal Distribution
shape, loc, scale = lognorm.fit(dfseattle['price'], floc=0)
plt.plot(x, lognorm(shape, loc, scale).pdf(x), color = 'y')

plt.xlabel('Price')
plt.ticklabel_format(style = 'plain')
plt.xticks(rotation='vertical')
plt.ylabel('Probability')
plt.title('Histogram of Region Seattle Prices\nAnd Some Distributions')
#plt.text(700000, 0.00000175, r'$\mu=540088,\ \sigma=367127$')
plt.axis([0,2000000,0,0.0000025])

handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in ['b', 'r', 'y', 'g']]
labels= ["Price","Normal", "Lognormal", "Weibull"]
plt.legend(handles, labels)


# In[164]:


n, bins, patches = plt.hist(dfsouth['price'], 200, density = 1, facecolor = 'b', alpha = .75)

#Overlay distributions that might be an appropriate fit
x = np.linspace(dfsouth['price'].min(), dfsouth['price'].max(), 100)

#Weibull Distribution
shape, loc, scale = weibull_min.fit(dfsouth['price'], floc=0)
plt.plot(x, weibull_min(shape, loc, scale).pdf(x), color = 'g')

#Normal Distribution
shape, loc = norm.fit(dfsouth['price'])
plt.plot(x, norm(shape, loc).pdf(x), color = 'r')

#Lognormal Distribution
shape, loc, scale = lognorm.fit(dfsouth['price'], floc=0)
plt.plot(x, lognorm(shape, loc, scale).pdf(x), color = 'y')

plt.xlabel('Price')
plt.ticklabel_format(style = 'plain')
plt.xticks(rotation='vertical')
plt.ylabel('Probability')
plt.title('Histogram of Region South Prices\nAnd Some Distributions')
#plt.text(700000, 0.00000175, r'$\mu=540088,\ \sigma=367127$')
plt.axis([0,2000000,0,0.0000050])

handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in ['b', 'r', 'y', 'g']]
labels= ["Price","Normal", "Lognormal", "Weibull"]
plt.legend(handles, labels)


# #### Split the data into training, testing, and validation sets

# In[218]:


#Create target and independent variable data sets
target = df['price']
ftrs = df.drop(['id', 'price', 'region', 'date'], axis=1)

#Split the data sets into training and testing data. Then split the testing data into testing and validation data
ftrs_train, ftrs_test1, target_train, target_test1 = skms.train_test_split(ftrs, target, 
                                                                                              test_size = 0.8, 
                                                                                              random_state = 1)
ftrs_test, ftrs_val, target_test, target_val = skms.train_test_split(ftrs_test1, target_test1, 
                                                                                        test_size = 0.5, 
                                                                                        random_state = 1)


# #### Run some models

# In[253]:


#KNN model
knn = neighbors.KNeighborsRegressor(n_neighbors=10)
fit = knn.fit(ftrs_train, target_train)
preds = knn.predict(ftrs_test)

#Evaluate teh predictions
msq = metrics.mean_squared_error(target_test, preds)
msq
#display(kc_test_target, preds)

####Try some different parameters####


# In[251]:


# Linear regression model
# Create linear regression object
regr = linear_model.LinearRegression()

#Train the model using the training sets
regr.fit(ftrs_train, target_train)

#Make predictions using the testing set
pred = regr.predict(ftrs_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
mse = metrics.mean_squared_error(target_test, pred)
print('Mean squared error: ', "{:.2f}".format(mse))
# Explained variance
result_expl_var = metrics.explained_variance_score(target_test, pred)
print('Score: ', "{:.2f}".format(result_expl_var))

# Plot outputs
plt.scatter(target_test, pred,  label='Actual vs Predicted', color = 'b', marker = 'o', s = 1)

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Plot of Actual Vs Predicted Results')



# In[ ]:


#Transform the target by logging the price, then run the regression again


# In[ ]:


#Tree Regression


# In[ ]:


#Random Forest






