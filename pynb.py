import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('insurance.csv')
data.head()
data.shape
data.info()
data.isnull().sum()
data.sex.value_counts()
data.region.value_counts()
encoder = LabelEncoder()
labels = encoder.fit_transform(data.sex)
data['sex'] = labels
data.head()
labels = encoder.fit_transform(data.region)
data['region'] = labels
data.head()
labels = encoder.fit_transform(data.smoker)
data['smoker'] = labels
data.head()
X = data.drop(columns='charges',axis=1)
Y = data['charges']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=101)
print(X.shape,X_train.shape,X_test.shape)
model = RandomForestRegressor()
model.fit(X_train,Y_train)
testing_data_prediction = model.predict(X_test)
score = metrics.r2_score(Y_test,testing_data_prediction)
score
input_data = (19,0,27.9,0,1,3)
input_data_array = np.asarray(input_data)
input_data_reshaped = input_data_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print('Predicted Medical Insurance Cost : ',str(prediction))
from joblib import dump
filename = 'medical_insurance_cost_predictor.joblib'
dump(model, filename)