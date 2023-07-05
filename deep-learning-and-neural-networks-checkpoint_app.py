#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import speech_recognition as sr


# In[2]:


# Load the necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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

def transcribe_speech(language):
    # Initialize recognizer class
    r = sr.Recognizer()

    # Set the speech recognition API based on the selected language
    if language == "English":
        recognizer_api = "google"
    elif language == "French":
        recognizer_api = "sphinx"
    # Add more options for other languages and their respective APIs

    # Reading Microphone as source
    with sr.Microphone() as source:
        st.info("Speak now...")

        # Pause and resume speech recognition
        r.pause_threshold = 1

        # listen for speech and store in audio_text variable
        audio_text = r.listen(source)
        st.info("Transcribing...")

        try:
            # using selected speech recognition API
            if recognizer_api == "google":
                text = r.recognize_google(audio_text, language=language)
            elif recognizer_api == "sphinx":
                text = r.recognize_sphinx(audio_text, language=language)
            # Add more conditions for other speech recognition APIs

            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand what you said."
        except sr.RequestError as e:
            return f"Sorry, an error occurred during speech recognition: {e}"


# In[ ]:


# Create a Streamlit app
def main():
    st.title("Speech-Enabled Chatbot")

    # Get user input (text or speech)
    input_type = st.radio("Input type:", ("Text", "Speech"))

    if input_type == "Text":
        input_text = st.text_input("Enter your message:")
    else:
        input_text = transcribe_speech("English")  # Change the language if required

    # Process user input and generate chatbot response
    if input_text:
        processed_input = preprocess(input_text)
        response = chatbot(processed_input[0], processed_input)  # Assuming the first sentence is the query
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
    get_ipython().system('streamlit run C:\\Users\\PC\\Desktop\\DATASCIENCE\\deep-learning-and-neural-networks-checkpoint_app.py')


# In[ ]:




