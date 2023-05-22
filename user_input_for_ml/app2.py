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
def home():
    return render_template('input.html') 

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
    #input_level
    input_level=request.form['job_level']
    for x in input_level:
        if x =='1':
            blank_dummies_df['level_1'] = 1
        elif x=='2':
            blank_dummies_df['level_2'] = 1
        elif x=='3':
            blank_dummies_df['level_3'] = 1
        elif x=='4':
            blank_dummies_df['level_4'] = 1
        elif x=='5':
            blank_dummies_df['level_5'] = 1
        elif x=='6':
            blank_dummies_df['level_6'] = 1
        elif x=='7':
            blank_dummies_df['level_7'] = 1
        elif x=='9':
            blank_dummies_df['level_9'] = 1
        elif x=='8':
            blank_dummies_df['level_8'] = 1
        elif x=='59':
            blank_dummies_df['level_59'] = 1
        elif x=='60':
            blank_dummies_df['level_60'] = 1
        elif x=='61':
            blank_dummies_df['level_61'] = 1
        elif x=='62':
            blank_dummies_df['level_62'] = 1
        elif x=='63':
            blank_dummies_df['level_63'] = 1
        elif x=='64':
            blank_dummies_df['level_64'] = 1
        elif x=='65':
            blank_dummies_df['level_65'] = 1
        elif x=='66':
            blank_dummies_df['level_66'] = 1
        elif x=='67':
            blank_dummies_df['level_67'] = 1
        elif x=='Analyst':
            blank_dummies_df['level_Analyst'] = 1
        elif x=='Associate':
            blank_dummies_df['level_Associate'] = 1
        elif x=='Associate Software Eng':
            blank_dummies_df['level_Associate Software Eng'] = 1
        elif x=='Band 7':
            blank_dummies_df['level_Band 7'] = 1
        elif x=='Band 8':
            blank_dummies_df['level_Band 8'] = 1
        elif x=='Consultant':
            blank_dummies_df['level_Consultant'] = 1
        elif x=='Director':
            blank_dummies_df['level_Director'] = 1
        elif x=='E3':
            blank_dummies_df['level_E3'] = 1
        elif x=='E4':
            blank_dummies_df['level_E4'] = 1
        elif x=='E5':
            blank_dummies_df['level_E5'] = 1
        elif x=='E6':
            blank_dummies_df['level_E6'] = 1
        elif x=='Engineer':
            blank_dummies_df['level_Engineer'] = 1
        elif x=='Grade 10':
            blank_dummies_df['level_Grade 10'] = 1
        elif x=='Grade 6':
            blank_dummies_df['level_Grade 6'] = 1
        elif x=='Grade 7':
            blank_dummies_df['level_Grade 7'] = 1
        elif x=='Grade 8':
            blank_dummies_df['level_Grade 8'] = 1
        elif x=='Grade 9':
            blank_dummies_df['level_Grade 9'] = 1
        elif x=='IC1':
            blank_dummies_df['level_IC1'] = 1
        elif x=='IC2':
            blank_dummies_df['level_IC2'] = 1
        elif x=='IC-2':
            blank_dummies_df['level_IC-2'] = 1
        elif x=='IC3':
            blank_dummies_df['level_IC3'] = 1
        elif x=='IC-3':
            blank_dummies_df['level_IC-3'] = 1
        elif x=='IC4':
            blank_dummies_df['level_IC4'] = 1
        elif x=='IC-4':
            blank_dummies_df['level_IC-4'] = 1
        elif x=='IC5':
            blank_dummies_df['level_IC5'] = 1
        elif x=='IC6':
            blank_dummies_df['level_IC6'] = 1
        elif x=='ICT2':
            blank_dummies_df['level_ICT2'] = 1
        elif x=='ICT3':
            blank_dummies_df['level_ICT3'] = 1
        elif x=='ICT4':
            blank_dummies_df['level_ICT4'] = 1
        elif x=='ICT5':
            blank_dummies_df['level_ICT5'] = 1
        elif x=='L1':
            blank_dummies_df['level_L1'] = 1
        elif x=='L2':
            blank_dummies_df['level_L2'] = 1
        elif x=='L3':
            blank_dummies_df['level_L3'] = 1
        elif x=='L4':
            blank_dummies_df['level_4'] = 1
        elif x=='L5':
            blank_dummies_df['level_L5'] = 1
        elif x=='L5A':
            blank_dummies_df['level_L5A'] = 1
        elif x=='L6':
            blank_dummies_df['level_L6'] = 1
        elif x=='L6 SDM':
            blank_dummies_df['level_L6 SDM'] = 1
        elif x=='L7':
            blank_dummies_df['level_L7'] = 1
        elif x=='L8':
            blank_dummies_df['level_L8'] = 1
        elif x=='Lead MTS':
            blank_dummies_df['level_Lead MTS'] = 1
        elif x=='M1':
            blank_dummies_df['level_M1'] = 1
        elif x=='M2':
            blank_dummies_df['level_M2']=1
        elif x=='M3':
            blank_dummies_df['level_M3']= 1
        elif x=='M4':
            blank_dummies_df['level_M4']=1
        elif x=='Manager':
            blank_dummies_df['level_Manager'] = 1
        elif x=='MTS':
            blank_dummies_df['level_MTS'] = 1
        elif x=='MTS 2':
            blank_dummies_df['level_MTS 2'] = 1                         
        elif x=='MTS 3':
            blank_dummies_df['level_MTS 3'] = 1                         
        elif x=='P2':
            blank_dummies_df['level_P2'] = 1
        elif x=='P3':
            blank_dummies_df['level_P3'] = 1
        elif x=='P4':
            blank_dummies_df['level_P4'] = 1                         
        elif x=='P5':
            blank_dummies_df['level_P5'] = 1                         
        elif x=='Principal':
            blank_dummies_df['level_Principal'] = 1
        elif x=='Principal Associate':
            blank_dummies_df['level_Principal Associate'] = 1
        elif x=='Principal Engineer':
            blank_dummies_df['level_Principal Engineer'] = 1                         
        elif x=='Principal MTS':
            blank_dummies_df['level_Principal MTS'] = 1                         
        elif x=='SDE I':
            blank_dummies_df['level_SDE I'] = 1
        elif x=='SDE II':
            blank_dummies_df['level_SDE II'] = 1
        elif x=='SDE III':
            blank_dummies_df['level_SDE III'] = 1                         
        elif x=='Senior':
            blank_dummies_df['level_Senior'] = 1                         
        elif x=='Senior Associate':
            blank_dummies_df['level_Senior Associate'] = 1
        elif x=='Senior Consultant':
            blank_dummies_df['level_Senior Consultant'] = 1
        elif x=='Senior Engineer':
            blank_dummies_df['level_Senior Engineer'] = 1                         
        elif x=='Senior Manager':
            blank_dummies_df['level_Senior Manager'] = 1                         
        elif x=='Senior MTS':
            blank_dummies_df['level_Senior MTS'] = 1
        elif x=='Senior Product Manager':
            blank_dummies_df['level_Senior Product Manager'] = 1
        elif x=='Senior Software Engineer':
            blank_dummies_df['level_Senior Software Engineer'] = 1                         
        elif x=='Senior SWE':
            blank_dummies_df['level_Senior SWE'] = 1                                                                              
        elif x=='Software Engineer':
            blank_dummies_df['level_Software Engineer'] = 1
        elif x=='Software Engineer 1':
            blank_dummies_df['level_Software Engineer 1'] = 1
        elif x=='Software Engineer 2':
            blank_dummies_df['level_Software Engineer 2'] = 1                         
        elif x=='Software Engineer 3':
            blank_dummies_df['level_Software Engineer 3'] = 1                         
        elif x=='Software Engineer I':
            blank_dummies_df['level_Software Engineer I'] = 1
        elif x=='Software Engineer II':
            blank_dummies_df['level_Software Engineer II'] = 1
        elif x=='Staff':
            blank_dummies_df['level_Staff'] = 1                         
        elif x=='Staff Engineer':
            blank_dummies_df['level_Staff Engineer'] = 1                         
        elif x=='Staff Software Engineer':
            blank_dummies_df['level_Staff Software Engineer'] = 1
        elif x=='SWE II':
            blank_dummies_df['level_SWE II'] = 1
        elif x=='T2':
            blank_dummies_df['level_T2'] = 1                         
        elif x=='T3':
            blank_dummies_df['level_T3'] = 1      
        elif x=='T4':
            blank_dummies_df['level_T4'] = 1
        elif x=='T5':
            blank_dummies_df['level_T5'] = 1
        elif x=='Vice President':
            blank_dummies_df['level_Vice President'] = 1                         
        elif x=='Other':
            blank_dummies_df['level_Other'] = 1                         
        else: 
            print("Level not found")
    input_list=[input_year, input_year_experience, input_year_at_company, input_latitude, input_longitude, input_month, input_bonus, input_stock, input_company, input_title, input_gender, input_level]
    #input_list=output["input_list"]
    #reduced_df_input_example = [ 'Amazon', 2018, 'Sofeware Engineer',  'Male', 'L4',
     #  0, 0, 47.603832, -122.330062, 3, 0, 0]

    return render_template('input.html', input_list=input_list ) 

if __name__ == '__main__':
    app.run(debug=True)