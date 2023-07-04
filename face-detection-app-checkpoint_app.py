#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install opencv-python')
import cv2
import streamlit as st


# In[5]:


face_cascade = cv2.CascadeClassifier('C:/sers/PC/Desktop/DATASCIENCE/haarcascade_frontalface_default.xml')


# In[7]:


def detect_faces(image, min_neighbors, scale_factor, rectangle_color):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minNeighbors=min_neighbors, scaleFactor=scale_factor)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), rectangle_color, 2)

    return image

def main():
    st.title("Face Detection App")
    st.write("Upload an image and adjust the parameters to detect faces.")

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
        st.image(image, channels="BGR", caption="Original Image")

        min_neighbors = st.slider("Adjust minNeighbors", min_value=1, max_value=10, value=5)

        scale_factor = st.slider("Adjust scaleFactor", min_value=1.1, max_value=2.0, step=0.1, value=1.2)

        rectangle_color = st.color_picker("Choose rectangle color", value="#FF0000")

        if st.button("Detect Faces"):
            result_image = detect_faces(image, min_neighbors, scale_factor, rectangle_color)
            st.image(result_image, channels="BGR", caption="Detected Faces")

            if st.button("Save Image"):
                cv2.imwrite("detected_faces.jpg", result_image)
                st.success("Image saved successfully.")

if __name__ == "__main__":
    main()


# In[8]:


get_ipython().run_line_magic('env', 'STREAMLIT_APP=face-detection-app-checkpoint_app.py')
get_ipython().system('streamlit run $STREAMLIT_APP')


# In[ ]:




