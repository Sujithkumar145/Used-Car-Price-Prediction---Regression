#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Importing Necessary Libraries


# In[169]:


import pandas as pd
import numpy as np


# In[170]:


df=pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')


# In[171]:


df


# In[173]:


df.info()


# In[174]:


df.describe()


# In[175]:


df['km_driven'].value_counts()


# In[176]:


df['year'].value_counts()


# In[178]:


print(df['name'].unique())
print(df['fuel'].unique())
print(df['seller_type'].unique())
print(df['transmission'].unique())
print(df['owner'].unique())


# In[179]:


data.isnull().sum()


# In[128]:


# There is no null values to drop,so we can proceed further


# In[185]:


# Converting selling price into Lakhs
df['price_inlakh']=df['selling_price']/100000


# In[187]:


df.drop(['selling_price'],axis=1,inplace=True)


# In[191]:


df


# In[ ]:



Exploratory Data Analysis(EDA)


# In[196]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.displot(df['price_inlakh'])
plt.show()


# In[197]:


import warnings
warnings.filterwarnings('ignore')


# In[198]:


import matplotlib.pyplot as plt
sns.boxplot(df['price_inlakh'])
plt.show()


# In[199]:


sns.boxplot(data['km_driven'])
plt.show()


# In[201]:


sns.countplot(df['seller_type'])
plt.show()


# In[203]:


sns.countplot(df['owner'])
plt.xticks(rotation=90)
plt.show()


# In[135]:


# First and Second owner cars mostly sold when compared to more owners


# In[205]:


sns.countplot(df['fuel'])
plt.show()


# In[206]:


sns.countplot(df['transmission'])
plt.show()


# In[207]:


corr=df.corr()
sns.heatmap(corr)
plt.show()


# In[139]:


# There is no such relation between the features


# In[208]:


sns.boxplot(y='price_inlakh',x='owner',data=df)
plt.show()


# In[210]:


sns.boxplot(y='price_inlakh',x='transmission',data=df)
plt.show()


# In[146]:


sns.boxplot(y='price_inlakh',x='fuel',data=data)
plt.show()


# In[ ]:


# After looking transmission,fuel,owner vs price_in lakh and boxplot of price there are few outliers 
# above forty lakh.We can remove that outliers


# In[211]:


df2=df[df['price_inlakh']<40]


# In[215]:


df2=df2[df2['km_driven']<300000]


# In[216]:


df2.shape


# In[ ]:


# 39 records identified as outliers and removed


# In[ ]:


# Data Preprocessing


# In[217]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[218]:


df2.name=le.fit_transform(df2.name)
df2.fuel=le.fit_transform(df2.fuel)
df2.transmission=le.fit_transform(df2.transmission)
df2.owner=le.fit_transform(df2.owner)
df2.seller_type=le.fit_transform(df2.seller_type)


# In[219]:


df2


# In[ ]:


# Model Building


# In[220]:


X=df2.drop('price_inlakh',axis=1)
Y=df2['price_inlakh']


# In[221]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.20,random_state =8)


# In[222]:


#Linear Regressor
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[223]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pre_lr=lr.predict(x_test)
mae = round(mean_absolute_error(y_test, y_pre_lr), 2)
mse = round(mean_squared_error(y_test, y_pre_lr), 2)
rmse = round(np.sqrt(mse), 2)
r2 = round(r2_score(y_test, y_pre_lr), 2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("r-squared:", r2)


# In[224]:


print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))


# In[235]:


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(x_train, y_train)
y_pre_rfr= rf_reg.predict(x_test)
print("Accuracy on Traing set: ",rfr.score(x_train,y_train))
print("Accuracy on Testing set: ",rfr.score(x_test,y_test))


# In[236]:


mae = round(mean_absolute_error(y_test, y_pre_rfr), 2)
mse = round(mean_squared_error(y_test, y_pre_rfr), 2)
rmse = round(np.sqrt(mse), 2)
r2 = round(r2_score(y_test, y_pre_rfr), 2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("r-squared:", r2)


# In[231]:


# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
y_pre_gbr= gbr.predict(x_test)
print("Accuracy on Traing set: ",gbr.score(x_train,y_train))
print("Accuracy on Testing set: ",gbr.score(x_test,y_test))


# In[232]:


mae = round(mean_absolute_error(y_test, y_pre_gbr), 2)
mse = round(mean_squared_error(y_test, y_pre_gbr), 2)
rmse = round(np.sqrt(mse), 2)
r2 = round(r2_score(y_test, y_pre_gbr), 2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("r-squared:", r2)


# In[233]:


# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
y_preddtr= dtr.predict(x_test)
print("Accuracy on Traing set: ",dtr.score(x_train,y_train))
print("Accuracy on Testing set: ",dtr.score(x_test,y_test))


# In[234]:


mae = round(mean_absolute_error(y_test, y_preddtr), 2)
mse = round(mean_squared_error(y_test, y_preddtr), 2)
rmse = round(np.sqrt(mse), 2)
r2 = round(r2_score(y_test, y_preddtr), 2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("r-squared:", r2)


# In[237]:


# AdaBoost Regressor
from sklearn.ensemble import AdaBoostRegressor
abr = AdaBoostRegressor()
abr.fit(x_train, y_train)
y_pre_abr= abr.predict(x_test)
print("Accuracy on Traing set: ",abr.score(x_train,y_train))
print("Accuracy on Testing set: ",abr.score(x_test,y_test))


# In[238]:


mae = round(mean_absolute_error(y_test, y_pre_abr), 2)
mse = round(mean_squared_error(y_test, y_pre_abr), 2)
rmse = round(np.sqrt(mse), 2)
r2 = round(r2_score(y_test, y_pre_abr), 2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("r-squared:", r2)


# In[239]:


regression_models = [lr,rfr,gbr,dtr,abr]
score_train = list()
score_test = list()

for model in regression_models : 
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    
    score_train.append(model.score(x_train,y_train))
    score_test.append(model.score(x_test,y_test))


# In[240]:


model_names = ['Linear Regression','Random Forest Regressor','Gradient Boosting Regressor','Decision Tree Regressor',
               'AdaBoostRegressor']

scores = pd.DataFrame([model_names,score_train,score_test])
scores


# In[241]:


scores = scores.transpose()
scores.columns = [ 'Model','Training Set Accuracy','Testing set Accuracy']
scores


# In[168]:


# Conclusion : Random Forest with 81 % accuracy and 
              0.89 as R-Squared value has chosen for prediction

