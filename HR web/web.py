# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 08:37:35 2021

@author: HP
"""
import  numpy as np
import pickle
from flask import Flask, render_template, request
model=pickle.load(open('model.pkl','rb'))
label=pickle.load(open('label.pkl','rb'))
app=Flask(__name__)
@app.route('/')  
def home():
    return render_template('home.html')   
@app.route("/index")
def index():
    return render_template('index.html')
@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')
@app.route('/predict', methods=['POST','GET'])
def predict():
    age=request.form['age']
    workclass=request.form['workclass']
    educationnum=request.form['educationnum']
    education=request.form['education']
    maritalstatus=request.form['maritalstatus']
   
    occupation=request.form['occupation']
    relationship=request.form['relationship']
    race=request.form['race']
    sex=request.form['sex']
    capitalgain=request.form['capitalgain']
    
    capitalloss=request.form['capitalloss']
    hoursperweek=request.form['hoursperweek']
    nativecountry=request.form['nativecountry']
    
    values=[age,workclass,education,educationnum,maritalstatus,occupation,relationship,race,
            sex,capitalgain,capitalloss,hoursperweek,nativecountry]
    data=[]
    for i in values:
        data.append(i)
    data = label.fit_transform(data)
    
    result=np.array(data).reshape(1,-1)
    prediction_result=model.predict(result)
    output=prediction_result.item()
    
    return render_template('result.html',prediction_text=format(output))


if __name__ == "__main__":                            
    app.run()
    
