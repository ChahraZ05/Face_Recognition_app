import os
import cv2
import faiss
import pickle
import numpy as np
import torch
import streamlit as st
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import LabelEncoder

# Function to preprocess the image using OpenCV
def preprocess_image_cv2(img):
    img = cv2.resize(img, (160, 160))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = torch.tensor(img).unsqueeze(0)  # Add batch dimension
    return img

# Load the embeddings and index
index = faiss.read_index('/kaggle/working/faiss_index.index')

# Load all embeddings and map to the respective names
all_embeddings = {}
dataset_folder = '/kaggle/input/celebrity-faces-dataset'
for person in os.listdir(dataset_folder):
    person_folder = os.path.join(dataset_folder, person)
    if os.path.isdir(person_folder):
        all_embeddings[person] = []  # List to hold embeddings for each person

# Build an array to store the person corresponding to each embedding
names = []
for person, embeddings in all_embeddings.items():
    for _ in range(100):  # Assuming each person has 100 images
        names.append(person)

# Load the model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Streamlit app
st.title("Face Recognition App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and preprocess the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_tensor = preprocess_image_cv2(img)
    
    # Get the embedding for the uploaded image
    with torch.no_grad():
        query_embedding = model(img_tensor).numpy().flatten()

    # Search the FAISS index for the most similar embeddings
    D, I = index.search(np.expand_dims(query_embedding, axis=0), k=1)  # Top 1 match

    # Get the index of the closest match
    closest_index = I[0][0]
    
    # Retrieve the name of the closest match using the `names` list
    closest_name = names[closest_index]
    
    # Display the result
    st.write(f"Closest Match: {closest_name}")
