#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


# In[2]:


# Step 1: Fetch Data from the API
response = requests.get('https://disease.sh/v3/covid-19/historical/all?lastdays=365')
data = response.json()
df = pd.DataFrame.from_dict(data['cases'], orient='index', columns=['cases'])


# In[3]:


# Step 2: Clean the Data
df.index = pd.to_datetime(df.index)  # Convert the index to datetime
df = df.reset_index().rename(columns={'index': 'date'})  # Reset index and rename columns
df['date'] = pd.to_datetime(df['date'])  # Convert the 'date' column to datetime


# In[4]:


# Step 3: Pre-process the Data
df['days_since'] = (df['date'] - df['date'].min()).dt.days  # Convert dates to numerical features
scaler = MinMaxScaler()
df['normalized_cases'] = scaler.fit_transform(df[['cases']])  # Normalize and scale the 'cases' column


# In[5]:


# Step 4: Perform EDA
fig = px.line(df, x='date', y='cases', title='COVID-19 Cases Over Time')  # Create a line plot


# In[6]:


# Step 5: Choose a Supervised Algorithm for Prediction
X = df[['days_since']]
y = df['cases']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[7]:


# Step 6: Deploy the Model Using Streamlit
st.title('COVID-19 Prediction')
date_input = st.date_input('Date')
input_date = pd.to_datetime(date_input)
input_days_since = (input_date - df['date'].min()).days
prediction = model.predict([[input_days_since]])
st.write('Predicted Cases:', prediction[0])


# In[ ]:




