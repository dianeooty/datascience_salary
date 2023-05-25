from flask import Flask, render_template
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

#Im not sure if this should be with the open clause, I added that based on youtube video code
rf_model=load('salary_ml_model.joblib')



app = Flask(__name__)
# webcode = open('webcode.html').read() - not needed

@app.route('/')
def webprint():
    return render_template('input.html') 

@app.route('/predict', methods=['POST', 'GET'])
def resutls():

    return render_template('input.html') 

if __name__ == '__main__':
    app.run(debug=True)