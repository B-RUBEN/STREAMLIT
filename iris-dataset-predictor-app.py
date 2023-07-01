#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import sklearn.datasets as datasets
from sklearn.ensemble import RandomForestClassifier


# In[2]:


# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target


# In[3]:


# Set up the Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X, Y)


# In[4]:


# Create Streamlit app
st.title("Iris Flower Prediction")
st.header("Enter the values for sepal length, sepal width, petal length, and petal width")

# Input fields for sepal length, sepal width, petal length, and petal width
sepal_length = st.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Prediction button
if st.button("Predict"):
    # Create a feature vector from the user input
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    # Make the prediction
    prediction = clf.predict(input_data)
    # Map the prediction to the target names
    predicted_class = iris.target_names[prediction[0]]
    # Display the predicted type of iris flower
    st.write("Predicted Iris Flower Type:", predicted_class)


# In[ ]:




