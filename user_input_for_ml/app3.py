from flask import Flask, render_template, request
from flask import jsonify
from joblib import dump, load
import pandas as pd
import numpy as np
#pd.set_option('max_colwidth', 400)
import requests
import matplotlib.pyplot as plt
import json
import re
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump, load

#Im not sure if this should be with the open clause, I added that based on youtube video code
#rf_model=load('salary_ml_model.joblib')



app = Flask(__name__)
# webcode = open('webcode.html').read() - not needed

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/result', methods=['POST', 'GET'])
def resutls():
    dummies_df = pd.read_csv('dummies_table.csv')
    dummies_df = dummies_df.drop(columns=['Unnamed: 0', 'totalyearlycompensation'])
    dummies_df_columns_list=list(dummies_df.columns)
    blank_dummies_df=pd.DataFrame(dummies_df_columns_list)
    blank_dummies_df[1]=0
    blank_dummies_df=blank_dummies_df.T
    blank_dummies_df=blank_dummies_df.set_axis([dummies_df_columns_list], axis=1, inplace=False)
    blank_dummies_df=blank_dummies_df.drop([0])
    #input year
    input_year= request.form['year_salary']
    blank_dummies_df['date'] = input_year
    #input_year_experience
    input_year_experience=request.form['year_experience']
    blank_dummies_df['yearsofexperience'] = input_year_experience
    #input_year_at_company
    input_year_at_company=request.form['year_experience_company']
    blank_dummies_df['yearsatcompany'] = input_year_at_company
    # #input_latitude
    # input_latitude=request.form['latitude']
    # blank_dummies_df['latitude'] = input_latitude
    # #input_long
    # input_longitude=request.form['longitude']
    # blank_dummies_df['longitude'] = input_longitude
    #input_month
    input_month=request.form['month']
    blank_dummies_df['month'] =input_month
    #input_bonus
    input_bonus=request.form['bonus']
    blank_dummies_df['bonus'] =input_bonus
    #input_stock
    input_stock=request.form['stock']
    blank_dummies_df['stockgrantvalue'] =input_stock
    #input_company
    input_company=request.form['company']
    for x in input_company:
        if x =='Amazon':
            blank_dummies_df['company_Amazon'] = 1
        elif x=='Apple':
            blank_dummies_df['company_Apple'] = 1
        elif x=='Bloomberg':
            blank_dummies_df['company_Bloomberg'] = 1
        elif x=='Capital One':
            blank_dummies_df['company_Capital One'] = 1
        elif x=='Cisco':
            blank_dummies_df['company_Cisco'] = 1
        elif x=='Facebook':
            blank_dummies_df['company_Facebook'] = 1
        elif x=='Goldman Sachs':
            blank_dummies_df['company_Goldman Sachs'] = 1
        elif x=='Google':
            blank_dummies_df['company_Google'] = 1
        elif x=='IBM':
            blank_dummies_df['company_IBM'] = 1
        elif x=='Intel':
            blank_dummies_df['company_Intel'] = 1
        elif x=='Intuit':
            blank_dummies_df['company_Intuit'] = 1
        elif x=='JPMorgan Chase':
            blank_dummies_df['company_JPMorgan Chase'] = 1
        elif x=='LinkedIn':
            blank_dummies_df['company_LinkedIn'] = 1
        elif x=='Microsoft':
            blank_dummies_df['company_Microsoft'] = 1
        elif x=='Oracle':
            blank_dummies_df['company_Oracle'] = 1
        elif x=='PayPal':
            blank_dummies_df['company_PayPal'] = 1
        elif x=='Qualcomm':
            blank_dummies_df['company_Qualcomm'] = 1
        elif x=='Salesforce':
            blank_dummies_df['company_Salesforce'] = 1
        elif x=='Uber':
            blank_dummies_df['company_Uber'] = 1
        elif x=='VMware':
            blank_dummies_df['company_VMware'] = 1
        elif x=='Other':
            blank_dummies_df['company_Other'] = 1
        else: 
            print("Company not found")
    #input_job_title
    input_title=request.form['job_title']
    for x in input_title:
        if x =='Business Analyst':
            blank_dummies_df['title_Business Analyst'] = 1
        elif x=='Data Scientist':
            blank_dummies_df['title_Data Scientist'] = 1
        elif x=='Hardware Engineer':
            blank_dummies_df['title_Hardware Engineer'] = 1
        elif x=='Human Resources':
            blank_dummies_df['title_Human Resources'] = 1
        elif x=='Management Consultant':
            blank_dummies_df['title_Management Consultant'] = 1
        elif x=='Marketing':
            blank_dummies_df['title_Marketing'] = 1
        elif x=='Mechanical Engineer':
            blank_dummies_df['title_Mechanical Engineer'] = 1
        elif x=='Product Designer':
            blank_dummies_df['title_Product Designer'] = 1
        elif x=='Product Manager':
            blank_dummies_df['title_Product Manager'] = 1
        elif x=='Recruiter':
            blank_dummies_df['title_Recruiter'] = 1
        elif x=='Sales':
            blank_dummies_df['title_Sales'] = 1
        elif x=='Software Engineer':
            blank_dummies_df['title_Software Engineer'] = 1
        elif x=='Software Engineering Manager':
            blank_dummies_df['title_Software Engineering Manager'] = 1
        elif x=='Solution Architect':
            blank_dummies_df['title_Solution Architect'] = 1
        elif x=='Technical Program Manager':
            blank_dummies_df['title_Technical Program Manager'] = 1
        else: 
            print("Title not found")
    #input_gender
    input_gender=request.form['gender']
    for x in input_gender:
        if x =='Female':
            blank_dummies_df['gender_Female'] = 1
        elif x=='Male':
            blank_dummies_df['gender_Male'] = 1
        elif x=='Other':
            blank_dummies_df['gender_Other'] = 1
        elif x=='Unknown':
            blank_dummies_df['gender_Unknown'] = 1
        else: 
            print("Input not found")


    #input_location
    input_location=request.form['location']
    for x in input_location:
        if x=="Amsterdam, NH, Netherlands": 
            blank_dummies_df["title_Amsterdam, NH, Netherlands"] = 1
        elif x=="Arlington, VA": 
            blank_dummies_df["title_Arlington, VA"] = 1
        elif x=="Atlanta, GA": 
            blank_dummies_df["title_Atlanta, GA"] = 1
        elif x=="Austin, TX": 
            blank_dummies_df["title_Austin, TX"] = 1
        elif x=="Bangalore, KA, India": 
            blank_dummies_df["title_Bangalore, KA, India"] = 1
        elif x=="Bellevue, WA": 
            blank_dummies_df["title_Bellevue, WA"] = 1
        elif x=="Bengaluru, KA, India": 
            blank_dummies_df["title_Bengaluru, KA, India"] = 1
        elif x=="Berlin, BE, Germany": 
            blank_dummies_df["title_Berlin, BE, Germany"] = 1
        elif x=="Boston, MA": 
            blank_dummies_df["title_Boston, MA"] = 1
        elif x=="Boulder, CO": 
            blank_dummies_df["title_Boulder, CO"] = 1
        elif x=="Cambridge, MA": 
            blank_dummies_df["title_Cambridge, MA"] = 1
        elif x=="Chicago, IL": 
            blank_dummies_df["title_Chicago, IL"] = 1
        elif x=="Cupertino, CA": 
            blank_dummies_df["title_Cupertino, CA"] = 1
        elif x=="Dallas, TX": 
            blank_dummies_df["title_Dallas, TX"] = 1
        elif x=="Denver, CO": 
            blank_dummies_df["title_Denver, CO"] = 1
        elif x=="Dublin, DN, Ireland": 
            blank_dummies_df["title_Dublin, DN, Ireland"] = 1
        elif x=="Hillsboro, OR": 
            blank_dummies_df["title_Hillsboro, OR"] = 1
        elif x=="Houston, TX": 
            blank_dummies_df["title_Houston, TX"] = 1
        elif x=="Hyderabad, TS, India": 
            blank_dummies_df["title_Hyderabad, TS, India"] = 1
        elif x=="Irvine, CA": 
            blank_dummies_df["title_Irvine, CA"] = 1
        elif x=="London, EN, United Kingdom": 
            blank_dummies_df["title_London, EN, United Kingdom"] = 1
        elif x=="Los Angeles, CA": 
            blank_dummies_df["title_Los Angeles, CA"] = 1
        elif x=="Los Gatos, CA": 
            blank_dummies_df["title_Los Gatos, CA"] = 1
        elif x=="Menlo Park, CA": 
            blank_dummies_df["title_Menlo Park, CA"] = 1
        elif x=="Minneapolis, MN": 
            blank_dummies_df["title_Minneapolis, MN"] = 1
        elif x=="Moscow, MC, Russia": 
            blank_dummies_df["title_Moscow, MC, Russia"] = 1
        elif x=="Mountain View, CA": 
            blank_dummies_df["title_Mountain View, CA"] = 1
        elif x=="New York, NY": 
            blank_dummies_df["title_New York, NY"] = 1
        elif x=="Other": 
            blank_dummies_df["title_Other"] = 1
        elif x=="Palo Alto, CA": 
            blank_dummies_df["title_Palo Alto, CA"] = 1
        elif x=="Philadelphia, PA": 
            blank_dummies_df["title_Philadelphia, PA"] = 1
        elif x=="Pittsburgh, PA": 
            blank_dummies_df["title_Pittsburgh, PA"] = 1
        elif x=="Plano, TX": 
            blank_dummies_df["title_Plano, TX"] = 1
        elif x=="Pleasanton, CA": 
            blank_dummies_df["title_Pleasanton, CA"] = 1
        elif x=="Portland, OR": 
            blank_dummies_df["title_Portland, OR"] = 1
        elif x=="Raleigh, NC": 
            blank_dummies_df["title_Raleigh, NC"] = 1
        elif x=="Redmond, WA": 
            blank_dummies_df["title_Redmond, WA"] = 1
        elif x=="Redwood City, CA": 
            blank_dummies_df["title_Redwood City, CA"] = 1
        elif x=="San Diego, CA": 
            blank_dummies_df["title_San Diego, CA"] = 1
        elif x=="San Francisco, CA": 
            blank_dummies_df["title_San Francisco, CA"] = 1
        elif x=="San Jose, CA": 
            blank_dummies_df["title_San Jose, CA"] = 1
        elif x=="Santa Clara, CA": 
            blank_dummies_df["title_Santa Clara, CA"] = 1
        elif x=="Seattle, WA": 
            blank_dummies_df["title_Seattle, WA"] = 1
        elif x=="Singapore, SG, Singapore": 
            blank_dummies_df["title_Singapore, SG, Singapore"] = 1
        elif x=="Sunnyvale, CA": 
            blank_dummies_df["title_Sunnyvale, CA"] = 1
        elif x=="Sydney, NS, Australia": 
            blank_dummies_df["title_Sydney, NS, Australia"] = 1
        elif x=="Taipei, TP, Taiwan": 
            blank_dummies_df["title_Taipei, TP, Taiwan"] = 1
        elif x=="Toronto, ON, Canada": 
            blank_dummies_df["title_Toronto, ON, Canada"] = 1
        elif x=="Vancouver, BC, Canada": 
            blank_dummies_df["title_Vancouver, BC, Canada"] = 1
        elif x=="Washington, DC": 
            blank_dummies_df["title_Washington, DC"] = 1
        elif x=="Zurich, ZH, Switzerland": 
            blank_dummies_df["title_Zurich, ZH, Switzerland"] = 1
        else: 
            print("Location not found")
    #input_list=[input_year, input_year_experience, input_year_at_company, input_month, input_bonus, input_stock, input_company, input_title, input_gender]
    #input_list=output["input_list"]
    #reduced_df_input_example = [ 'Amazon', 2018, 'Sofeware Engineer',  'Male', 'L4',
     #  0, 0, 47.603832, -122.330062, 3, 0, 0]

    # Create a StandardScaler instances
    from pickle import load
    scaler = load(open('scaler_salary.pkl', 'rb'))
    X_input_test_scaled = scaler.transform(blank_dummies_df)
  
    # Scale the data
    #X_input_test_scaled = X_input_test_scaled.transform(blank_dummies_df)

    #pickle.dump(rf_model, open('rf_model.pkl', 'wb'))
    pickled_model = load(open('salary_ml_model.pkl', 'rb')) 

    
    prediction_output = pickled_model.predict(X_input_test_scaled)
    prediction_output=prediction_output[0]



    return render_template('index.html',  prediction_output= prediction_output) 

if __name__ == '__main__':
    app.run(debug=True)