import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = pd.read_csv('Financial_inclusion_dataset.csv')
df = data.copy()
df.drop('uniqueid', inplace = True , axis = 1)

categoricals = df.select_dtypes(include = ['object', 'category'])
numericals = df.select_dtypes(include = 'number')

scaler = StandardScaler()
encoder = LabelEncoder()

for i in numericals.columns: 
    if i in df.columns: 
        df[i] = scaler.fit_transform(df[[i]]) 
for i in categoricals.columns:
    if i in df.columns: 
        df[i] = encoder.fit_transform(df[i])

x = df.drop('bank_account',axis = 1)
y = df.bank_account

# # - Using XGBOOST to find feature importance
# import xgboost as xgb
# model = xgb.XGBClassifier()
# model.fit(x, y)

# # Print feature importance scores
# xgb.plot_importance(model)

sel_cols = ['age_of_respondent', 'education_level', 'household_size', 'job_type', 'marital_status']
x = df[sel_cols]

df= pd.concat([x, y], axis =1) 
class1 = df.loc[df['bank_account'] == 1]
class0 = df.loc[df['bank_account'] == 0] 
class1_3000 = class0.sample(5000) 
new_dataframe = pd.concat([class1_3000, class1], axis = 0)

x = new_dataframe.drop('bank_account', axis = 1)
y = new_dataframe['bank_account']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 40, stratify = y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier() 
model.fit(xtrain, ytrain) 
cross_validation = model.predict(xtrain)
pred = model.predict(xtest) 

# save model
model = pickle.dump(model, open('Financial_inclusion.pkl', 'wb'))
print('\nModel is saved\n')

# ..............STREAMLIT DEVELOPEMENT..........
model = pickle.load(open('Financial_inclusion.pkl','rb'))

st.markdown("<h1 style = 'color: #0174BE; text-align: center;font-family: Arial, Helvetica, sans-serif; '>FINANCIAL INCLUSION</h1>", unsafe_allow_html= True)
st.markdown("<h3 style = 'margin: -25px; color: #FFC436; text-align: center;font-family: Arial, Helvetica, sans-serif; '>BY OLUWAYOMI ROSEMARY</h3>", unsafe_allow_html= True)
st.image('pngwing.com (7).png', width = 400)
st.markdown("<h2 style = 'color: #0F0F0F; text-align: center;font-family: Arial, Helvetica, sans-serif; '>BACKGROUND OF STUDY </h2>", unsafe_allow_html= True)

st.markdown('<br><br>', unsafe_allow_html= True)

st.markdown("<p>Financial inclusion prediction involves using various data-driven techniques to forecast and analyze trends related to expanding access to financial services. It typically encompasses assessing and predicting the likelihood of individuals or communities gaining access to banking, credit, insurance, and other financial services. This predictive analysis relies on datasets encompassing demographic information, economic indicators, technological advancements, and government policies, among other factors. Machine learning models, statistical analysis, and predictive analytics are often employed to identify patterns and predict future trends in financial inclusion. The aim is to anticipate and address barriers to financial access, understand the impact of interventions or policy changes, and forecast the potential success of initiatives aimed at improving financial inclusion for underserved populations globally..</p>",unsafe_allow_html= True)

st.sidebar.image('pngwing.com (9).png')


dx = data[['age_of_respondent', 'household_size', 'job_type', 'education_level', 'marital_status']]
st.write(data.head())

age_of_respondent = st.sidebar.number_input("age_of_respondent", data['age_of_respondent'].min(), data['age_of_respondent'].max())
household_size = st.sidebar.number_input("household_size", data['household_size'].min(), data['household_size'].max())
job_type = st.sidebar.selectbox("Job Type", data['job_type'].unique())
education_level = st.sidebar.selectbox("education_level", data['education_level'].unique())
marital_status = st.sidebar.selectbox("marital_status", data['marital_status'].unique())


# Bring all the inputs into a dataframe
input_data = pd.Series({
    'age_of_respondent': age_of_respondent,
    'education_level': education_level,
    'household_size': household_size,
    'job_type': job_type,
    'marital_status': marital_status
})

# Reshape the Series to a DataFrame
input_variable = input_data.to_frame().T

st.write(input_variable)


categoricals = input_variable.select_dtypes(include = ['object', 'category'])
numericals = input_variable.select_dtypes(include = 'number')
# Standard Scale the Input Variable.

for i in numericals.columns:
    if i in input_variable.columns:
      input_variable[i] = StandardScaler().fit_transform(input_variable[[i]])
for i in categoricals.columns:
    if i in input_variable.columns: 
        input_variable[i] = LabelEncoder().fit_transform(input_variable[i])

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("<h2 style = 'color: #0A2647; text-align: center; font-family: helvetica '>Model Report</h2>", unsafe_allow_html = True)



if st.button('Press To Predict'):
    predicted = model.predict(input_variable)
    st.toast('bank_account Predicted')
    st.image('pngwing.com (10).png', width = 100)
    st.success(f'Inclusion is {predicted}')
