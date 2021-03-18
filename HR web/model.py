 # -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 17:56:07 2021

@author: HP
"""


import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data=pd.read_csv("1614238459_salarydata.csv")
label_encoder=preprocessing.LabelEncoder()
for i in ['workclass', 'education', 'marital-status', 'occupation',
       'relationship',"race",'sex','native-country']:
    data[i]=label_encoder.fit_transform(data[i])
    
y=data['salary']
x=data.drop('salary',axis=1) 

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
gb.fit(x_train,y_train)
y.predict=gb.predict(x_test)

pickle.dump(label_encoder,open('label.pkl','wb'))
pickle.dump(gb, open('model.pkl','wb'))
