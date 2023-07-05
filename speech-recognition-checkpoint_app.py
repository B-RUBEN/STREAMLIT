#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import speech_recognition as sr


# In[2]:


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



# In[3]:


def main():
    st.title("Speech Recognition App")
    st.write("Click on the microphone to start speaking:")

    # Language selection
    language = st.selectbox("Select Language", ["English", "French"])
    st.write(f"Selected Language: {language}")

    # add a button to trigger speech recognition
    if st.button("Start Recording"):
        text = transcribe_speech(language)
        st.write("Transcription: ", text)

        # Save transcribed text to a file
        if st.button("Save Transcription"):
            with open("transcription.txt", "w") as file:
                file.write(text)
            st.success("Transcription saved successfully.")

if __name__ == "__main__":
    main()


# In[4]:


get_ipython().system('streamlit run speech-recognition-checkpoint_app.py --server.enableCORS false')


# In[ ]:




