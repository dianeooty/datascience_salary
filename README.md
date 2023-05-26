# Predicting Our Future: A Deep Dive into Salary and Layoff Data
The world of data science is evolving so fast and data science was named 
the fastest-growing job in 2017 by LinkedIn. A recent study by 
PriceWaterhouseCoopers said, “the best jobs right now in America include 
titles like data scientist, data engineer, and business analyst.” As a 
result of this growing trend, our team will be analyzing the salaries of 
the data science profession based on years of experience, type of work 
you’re open to, your location and the size of the company you’re joining.
Additionally, our team will be analyzing layoff trends to have a more 
comprehensive view of the current job market. 
In order to achieve these goals, our team will use the salaries and 
layoffs dataset to analyze trends, create data visualizations to reflect 
the findings and build machine learning models for salary and layoff 
predictions.  

## Table of Contents
* [Project Status](#project-status)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [General Information](#general-information)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Project Submission Files](#project-submission-files)
* [Acknowledgements and Resources](#acknowledgements-and-resources)
* [Group](#group)

## Project Status
Project is complete

## Technologies Used
- Jupyter Notebook
- Google Colab
- Python
- HTML
- CSS
- Javascript
- Tableau
- Matplotlib
- RandomForest Regressor
- RandomForest Classifier
- RandomForest MultiOutput Regressor
- Pandas
- Sklearn
- Postgresql
- MongoDB
- Seaborn

## Features
List the ready features here:
- Cleaned and ready to use Salary and Layoff datasets
- Jupyter Notebooks housing a variety of data visualizations that can easily be exported
- HTML/CSS site that has interactive visuzalizations and is integrated with Tableau interactive maps
- Interactive Salary Predictor

## General Information
This project uses a variety of files and technologies to accomplish the 
goals stated in the introduction. To see the main result of the team’s 
work, we recommend running the file “app3.py” which uses Flask. This will
show interactive visualizations of the results, the accuracy of the 
salary and layoff machine learning models. Additionally, the Flask 
application includes an interactive form where the user can submit 
different inputs and the machine learning model will predict the annual 
total income. For “app3.py” to work correctly, the file structure will 
need to stay as currently seen in GitHub. For more static visualizations 
please see the “Screenshots” section below, where the results of the 
statistical analysis can be found along with additional information the 
team would like to highlight. 
To view the progress history of how the machine learning models were 
built, we recommend referencing the folder titled “ml_archived_files” in 
Kelsey’s branch.

## Screenshots
Top 20 Highest Paying Companies (based on Company's Average Salary)
![2](https://github.com/dianeooty/datascience_salary/assets/117790100/57494e5c-59e1-4bb8-9cc2-8aa5c0cff12d)


Average Salary by Title YoY
![1](https://github.com/dianeooty/datascience_salary/assets/117790100/e6d07154-bbb0-497e-900f-2ab70a7b781e)


Layoff Employee Distribution Map
![1](https://github.com/dianeooty/datascience_salary/assets/117790100/510e3854-9208-488c-9c49-c12f445d91d5)


Layoff Trends 
![2](https://github.com/dianeooty/datascience_salary/assets/117790100/c5667145-ae65-4ef9-b624-958568be0144)


Postgresql 

![1](https://github.com/dianeooty/datascience_salary/assets/117790100/9919c84a-d566-4b34-a024-158dbe46bce6)


MongoDB Compass

![2](https://github.com/dianeooty/datascience_salary/assets/117790100/d86f91d7-f1d5-46f9-9fba-a8b5bcf4744f)

Statistical Analysis<br>

Gender<br>
<img width="739" alt="image" src="https://github.com/dianeooty/datascience_salary/assets/118244319/76519536-d64f-436d-b2b5-8bd21acb42f2">
<img width="616" alt="image" src="https://github.com/dianeooty/datascience_salary/assets/118244319/18dffdee-46df-40de-96f6-c54f2d3067ed">

Race<br>
<img width="732" alt="image" src="https://github.com/dianeooty/datascience_salary/assets/118244319/486f96aa-8391-429d-824e-cbd7472e13af">
<img width="669" alt="image" src="https://github.com/dianeooty/datascience_salary/assets/118244319/02e798ec-c4ee-4b1b-b8a5-1eb41423ab50">

Education <br>
<img width="664" alt="image" src="https://github.com/dianeooty/datascience_salary/assets/118244319/2df337e8-2a75-4a3a-93d9-3e272fa08212">
<img width="681" alt="image" src="https://github.com/dianeooty/datascience_salary/assets/118244319/d6e368df-5177-4934-a528-de2b36c676b4">

## Setup
- Data Cleanup: All original data files can be found in the Resources 
folder.  The notebook titled data_cleanup.ipynb contains all the codes 
for data cleaning and preparation for database upload.
- Data Exploration: The notebook titled Project4-Visualizations.ipynb 
ingests the clean data and does a variety of exploratory views of the 
salary and layoff dataset. There are cooresponding visualizations for the
different views and datasets.
- Statistical Analysis: The notebook titled 
datasciencesalaries_corr_update2.ipynb contains all the statistical 
analysis on the salaries dataset.
- Machine Learning: 
  * The notebook titled “salary_ml_RFRegressor_final.ipynb” contains the
code to create the final machine learning model (sklearn 
RandomForestRegressor) for salary prediction and it exports the 
files needed for another notebook to run predictions based on the 
model.
  * The notebook titled “salary_ml_RFRegressor_prediction.ipynb” 
contains the code to make salary predictions based on importing the 
relevant salary machine learning files.
  * The notebook titled “layoff_ml_RFClassifier_final.ipynb” contains 
the code to create the final machine learning model (sklearn 
RandomForestClassifier) for predicting whether or not a layoff is 
anticipated. 
- Interactive Data Visualizations:
  * The interactive data visualizations are run through Flask, in the 
file titled “app3.py”. The relevant html (index.html), css 
(styles.css), and js (scripts.js) files can be found in the 
“Templates” and “Static” folders

## Room for Improvement

Room for improvement:
- Salaries dataset used in project contain years 2017-2021 and layoffs dataset contain years 2020-2023. Retrieve updated salary and layoff data at https://www.levels.fyi/ and https://layoffs.fyi/ using Web Scraping and ETL.
- Create an Interactive Layoff Predictor on the HTML

## Project Submission Files
* Proposal: Project 4 Proposal - Group 6.pdf
* Salaries Dataset: data_cleanup.csv
* Layoffs Dataset: layoffs_cleaned.csv
* Entity Relationship Diagram: ERD.png
* Notebook with Visualizations: Project4-Visualizations.ipynb
* Notebook with Statistical Analysis: datasciencesalaries_corr_update2.ipynb
* Machine Learning Notebook: ENTER FILE NAME HERE
* Postgresql Database: schemas.sql and ERD.png
* MongoDB: data_cleanup.ipynb

* Presentation Slide Deck: ENTER FILE NAME
* Presentation HTML/CSS Code: ENTER FILE NAME
* Presentation Website: ENTER WEB ADDRESS unless still running off Flask


## Acknowledgements and Resources
- Many thanks to our instructional team
- Referenced for tutorial making a hat graph: https://matplotlib.org/stable/gallery/lines_bars_and_markers/hat_graph.html
- Dataset: https://www.kaggle.com/datasets/jackogozaly/data-science-and-stem-salaries
- Dataset: https://www.kaggle.com/datasets/swaptr/layoffs-2022
- Reference: https://layoffs.fyi/
- Reference: https://www.levels.fyi/
- Reference for Seaborn graph: https://seaborn.pydata.org/
- Referenced the following site to learn how to capture user input, process it, and then display an output:
  * https://www.youtube.com/watch?v=0meTbQQaosU
  * https://www.youtube.com/watch?v=i3RMlrx4ol4
-Referenced to learn about hyperarameters in ML: 
  * https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
  * https://www.youtube.com/watch?v=SctFnD_puQI
- Referenced to learn about TensorFlow ML (final models do not use this information): https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel
- Reference to learn about GridSearchCV (the final models do not use this): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- Referenced to learn about RandomForestRegressor: 
  * https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
  * https://scikit-learn.org/0.16/modules/generated/sklearn.ensemble.RandomForestRegressor.html
  * https://stackoverflow.com/questions/52648383/how-to-get-coefficients-and-feature-importances-from-multioutputregressor
- Referenced for how to learn how to export and import ml models: https://scikit-learn.org/stable/model_persistence.html
- Referenced for how to save user inputs: https://stackoverflow.com/questions/17433557/how-to-save-user-input-into-a-variable-in-html-and-javascript

## Group
Created by Diane Guzman, Kelsey Brantner, Jiamin Li, Laura Jordan and Xiaolin Ruan

<!-- ## License -->

