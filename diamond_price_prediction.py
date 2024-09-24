# -*- coding: utf-8 -*-
"Diamond Price Prediction"

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
df = pd.read_csv('diamonds.csv')
df.head()

"""## Data Preprocessing"""

df.shape

#checking for null values
df.info()

#checking descriptive statistics
df.describe()

#values count of categorical variables
print(df.cut.value_counts(),'\n',df.color.value_counts(),'\n',df.clarity.value_counts())

df.head(10)

"""## Exploratory Data Analysis"""

sns.histplot(df['price'],bins = 20)

sns.histplot(df['carat'],bins=20)

"""Most of the diamonds are less then 1 carat in weight."""

plt.figure(figsize=(5,5))
plt.pie(df['cut'].value_counts(),labels=['Ideal','Premium','Very Good','Good','Fair'],autopct='%1.1f%%')
plt.title('Cut')
plt.show()

plt.figure(figsize=(5,5))
plt.bar(df['color'].value_counts().index,df['color'].value_counts())
plt.ylabel("Number of Diamonds")
plt.xlabel("Color")
plt.show()

plt.figure(figsize=(5,5))
plt.bar(df['clarity'].value_counts().index,df['clarity'].value_counts())
plt.title('Clarity')
plt.ylabel("Number of Diamonds")
plt.xlabel("Clarity")
plt.show()

sns.histplot(df['table'],bins=10)
plt.title('Table')
plt.show()

"""### Comparing Diamond's features with Price"""

sns.barplot(x='cut',y='price',data=df)

sns.barplot(x='color',y='price',data=df)
plt.title('Price vs Color')
plt.show()

sns.barplot(x = 'clarity', y = 'price', data = df)


#changing categorical variables to numerical variables
df['cut'] = df['cut'].map({'Ideal':5,'Premium':4,'Very Good':3,'Good':2,'Fair':1})
df['color'] = df['color'].map({'D':7,'E':6,'F':5,'G':4,'H':3,'I':2,'J':1})
df['clarity'] = df['clarity'].map({'IF':8,'VVS1':7,'VVS2':6,'VS1':5,'VS2':4,'SI1':3,'SI2':2,'I1':1})

"""## Coorelation"""

#coorelation matrix
df.corr()

#plotting the correlation heatmap
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

"""#### Ploting the relationship between Price and Carat"""

sns.lineplot(x='carat',y='price',data=df)
plt.title('Carat vs Price')
plt.show()

fig, ax = plt.subplots(2,3,figsize=(15,5))
sns.scatterplot(x='x',y='carat',data=df, ax=ax[0,0])
sns.scatterplot(x='y',y='carat',data=df, ax=ax[0,1])
sns.scatterplot(x='z',y='carat',data=df, ax=ax[0,2])
sns.scatterplot(x='x',y='price',data=df, ax=ax[1,0])
sns.scatterplot(x='y',y='price',data=df, ax=ax[1,1])
sns.scatterplot(x='z',y='price',data=df, ax=ax[1,2])
plt.show()


## Train Test Split

from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train = train_test_split(df.drop('price',axis=1),df['price'],test_size=0.2,random_state=42)

"""## Model Building

### Decision Tree Regressor
"""

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt

#training the model
dt.fit(x_train,y_train)
#train accuracy
dt.score(x_train,y_train)

#predicting the test set
dt_pred = dt.predict(x_test)

"""### Random Forest Regressor"""

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf

#training the model
rf.fit(x_train,y_train)
#train accuracy
rf.score(x_train,y_train)

#predicting the test set
rf_pred = rf.predict(x_test)

"""## Model Evaluation"""

from sklearn.metrics import mean_squared_error,mean_absolute_error

"""### Decision Tree Regressor"""

#distribution plot for actual and predicted values
ax = sns.distplot(y_test,hist=False,color='r',label='Actual Value')
sns.distplot(dt_pred,hist=False,color='b',label='Fitted Values',ax=ax)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of Diamonds')
plt.show()

print('Decision Tree Regressor RMSE:',np.sqrt(mean_squared_error(y_test,dt_pred)))
print('Decision Tree Regressor Accuracy:',dt.score(x_test,y_test))
print('Decision Tree Regressor MAE:',mean_absolute_error(y_test,dt_pred))

"""### Random Forest Regressor"""

#distribution plot for actual and predicted values
ax = sns.distplot(y_test,hist=False,color='r',label='Actual Value')
sns.distplot(rf_pred,hist=False,color='b',label='Fitted Values',ax=ax)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of Diamonds')
plt.show()

print('Random Forest Regressor RMSE:',np.sqrt(mean_squared_error(y_test,rf_pred)))
print('Random Forest Regressor Accuracy:',rf.score(x_test,y_test))
print('Random Forest Regressor MAE:',mean_absolute_error(y_test,rf_pred))

"""## Conclusion

Both the models have almost same accuracy. However, the Random Forest Regressor model is slightly better than the Decision Tree Regressor model.

There is something interesting about the data. The price of the diamonds with J color and I1 clarity is higher than the price of the diamonds with D color and IF clarity which couldn't be explained by the models. This could be because of the other factors that affect the price of the diamond.
"""
