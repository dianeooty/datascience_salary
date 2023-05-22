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
rf_model=load('salary_ml_model.joblib')



app = Flask(__name__)
# webcode = open('webcode.html').read() - not needed

@app.route('/')
def webprint():
    return render_template('input.html') 

@app.route('/predict', methods=['POST', 'GET'])
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
    #input_latitude
    input_latitude=request.form['latitude']
    blank_dummies_df['latitude'] = input_latitude
    #input_long
    input_longitude=request.form['longitude']
    blank_dummies_df['longitude'] = input_longitude
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

    return render_template('input.html') 

if __name__ == '__main__':
    app.run(debug=True)