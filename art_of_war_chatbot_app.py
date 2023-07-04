#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


# Preprocess the data
def preprocess(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    preprocessed_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [word for word in words if word.lower() not in stop_words]
        preprocessed_sentence = " ".join(words)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return preprocessed_sentences


# In[5]:


# Define the similarity function
def get_most_relevant_sentence(query, sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(tfidf_matrix, query_vector)
    most_similar_sentence = sentences[similarities.argmax()]
    return most_similar_sentence

# Define the chatbot function
def chatbot(query, sentences):
    most_relevant_sentence = get_most_relevant_sentence(query, sentences)
    response = "The most relevant sentence to your query is: " + most_relevant_sentence
    return response

# Create a Streamlit app
get_ipython().run_line_magic('env', 'STREAMLIT_APP=art_of_war_chatbot-app')
get_ipython().system('streamlit run $STREAMLIT_APP')
def main():
    st.title("Chatbot: The Art of War")
    text = open("art_of_war.txt").read()  # Replace with the path to your text file
    
    sentences = preprocess(text)
    
    user_input = st.text_input("User Query:", "")
    
    if user_input:
        response = chatbot(user_input, sentences)
        st.text("Chatbot Response:")
        st.write(response)

if __name__ == "__main__":
    main()


# In[ ]:




