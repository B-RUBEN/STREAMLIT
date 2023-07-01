#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


# In[10]:


get_ipython().run_line_magic('env', 'STREAMLIT_APP=Iris Dataset Predictor.py')
get_ipython().system('streamlit run $STREAMLIT_APP')


# In[11]:


# Load the iris dataset
iris = load_iris()
X = iris.data
Y = iris.target


# In[12]:


# Set up the Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X, Y)


# In[13]:


# Create Streamlit app
st.title("Iris Flower Prediction")
st.header("Enter the values for sepal length, sepal width, petal length, and petal width")


# In[14]:


# Input fields for sepal length, sepal width, petal length, and petal width
sepal_length = st.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))


# In[15]:


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




