import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sb
sb.set()
from datetime import datetime

data = pd.read_csv('ipl.csv')

# Doing EDA

data.drop(['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker'], axis = 1, inplace = True)

consistent_teams =  ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                     'Kings XI Punjab' , 'Mumbai Indians','Royal Challengers Bangalore',
                     'Delhi Daredevils', 'Sunrisers Hyderabad']

data = data[(data['bat_team'].isin(consistent_teams) & data['bowl_team'].isin(consistent_teams))]

data['bat_team'].unique()
data['bowl_team'].unique()

data = data[data['overs']>=5.0]

#data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
data['date'] = pd.to_datetime(data.date, format='%Y/%m/%d')

data.reset_index(drop = True, inplace = True)

# Train Test split like a Time-Series data

X_train = data[data['date'].dt.year<=2016]
X_test = data[data['date'].dt.year>2016]
Y_train = data[data['date'].dt.year<=2016]
Y_test = data[data['date'].dt.year>2016]

X_train = X_train.iloc[:, 1:-1]
X_test = X_test.iloc[:, 1:-1]
Y_train = Y_train.iloc[:, -1]
Y_test = Y_test.iloc[:, -1]

# Onehot Encoding the data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers =[('encoder', OneHotEncoder(), [0, 1])] , remainder = 'passthrough')

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# Machine Learning

from catboost import CatBoostRegressor
regressor = CatBoostRegressor()
regressor.fit(X_train, Y_train)

from sklearn.linear_model import Ridge
regressor = Ridge(alpha = 40)
regressor.fit(X_train, Y_train)

from sklearn.linear_model import Lasso
regressor = Lasso(alpha = 1)
regressor.fit(X_train, Y_train)

# Cross Validation and checknig accuracy

from sklearn.model_selection import cross_val_score 
accuracies = cross_val_score(regressor, X = X_test, y = Y_test, cv = 5, scoring = 'neg_mean_squared_error') #Cant take training value randomly in Time-Series type data
accuracies.mean()

Y_test = Y_test.reset_index()
Y_test = Y_test.iloc[:,1]
yhat = regressor.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error
accuracy = r2_score(Y_test, yhat)
mse = mean_squared_error(Y_test, yhat)

sb.distplot(Y_test-yhat) # It should be a normal distribution

# Exporting model

with open('regressor.pkl', 'wb') as filename:
  pickle.dump(regressor, filename)