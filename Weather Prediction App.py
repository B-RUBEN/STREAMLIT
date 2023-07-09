#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import speech_recognition as sr
import sqlite3
import streamlit as st
from sklearn.preprocessing import LabelEncoder


# In[2]:


def retrieve_weather_data(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    return data


# In[3]:


def preprocess_weather_data(data):
    weather_main = data["weather"][0]["main"]
    weather_description = data["weather"][0]["description"]
    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    wind_speed = data["wind"]["speed"]
    weather_data = {
        "Weather Main": weather_main,
        "Weather Description": weather_description,
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind Speed": wind_speed
    }
    return weather_data


# In[4]:


# Data visualization and analysis
def visualize_weather_data(weather_df):
    plt.hist(weather_df["Temperature"], bins=20)
    plt.xlabel("Temperature")
    plt.ylabel("Frequency")
    plt.title("Temperature Distribution")
    plt.show()

    weather_main_counts = weather_df["Weather Main"].value_counts()
    plt.bar(weather_main_counts.index, weather_main_counts.values)
    plt.xlabel("Weather Main")
    plt.ylabel("Count")
    plt.title("Weather Main Distribution")
    plt.show()


# In[5]:


def train_xgboost_model(weather_df):
    X = weather_df.drop(["Weather Main", "Weather Description"], axis=1)  
    y = weather_df["Weather Main"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except ValueError:
        X = pd.concat([X, X], ignore_index=True)
        y = pd.Series(np.concatenate((y, y), axis=None))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    return model, accuracy, precision, recall, f1


# In[6]:


def store_weather_data_in_database(weather_data):
    conn = sqlite3.connect("weather.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS weather (
                    WeatherMain TEXT,
                    WeatherDescription TEXT,
                    Temperature REAL,
                    Humidity INTEGER,
                    WindSpeed REAL
                    )''')
    c.execute("INSERT INTO weather VALUES (?, ?, ?, ?, ?)",
              (weather_data["Weather Main"], weather_data["Weather Description"],
               weather_data["Temperature"], weather_data["Humidity"], weather_data["Wind Speed"]))
    conn.commit()
    conn.close()


# In[7]:


def speech_recognition():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Speak:")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.write("Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        st.write("Could not request results from Speech Recognition service; {0}".format(e))
        return ""


# In[8]:


data = retrieve_weather_data("5e83403a482f89974555898a1a1c9033", "New York")

weather_data = preprocess_weather_data(data)

weather_df = pd.DataFrame([weather_data])
visualize_weather_data(weather_df)

model, accuracy, precision, recall, f1 = train_xgboost_model(weather_df)

store_weather_data_in_database(weather_data)


# In[9]:



st.title("Weather Prediction App")

st.header("Weather Prediction")
st.subheader("Enter a city name:")
city_name = st.text_input("City Name", "")
if st.button("Predict Weather"):
    data = retrieve_weather_data("5e83403a482f89974555898a1a1c9033", city_name)
    weather_data = preprocess_weather_data(data)
    weather_df = pd.DataFrame([weather_data])
    prediction = model.predict(weather_df.drop("Weather Main", axis=1))[0]
    st.write("Predicted Weather Main:", prediction)

st.header("Speech Recognition")
st.subheader("Speak a city name:")
if st.button("Start Recording"):
    st.write("Recording...")
    spoken_text = speech_recognition()
    st.write("Spoken Text:", spoken_text)


# In[10]:


st.header("SQLite Database")
conn = sqlite3.connect("weather.db")
weather_df_from_db = pd.read_sql_query("SELECT * FROM weather", conn)
conn.close()
st.write(weather_df_from_db)

