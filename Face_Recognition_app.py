import streamlit as st
import numpy as np
import torch
from PIL import Image
import faiss
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2
import io

# Load the pre-trained face embedding model
model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=True)

# Load FAISS index
index = faiss.read_index('faiss_index.index')

# Load embeddings
import pickle
with open('lfw_embeddings.pkl', 'rb') as f:
    all_embeddings = pickle.load(f)

# Function to preprocess images
def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (160, 160))
    image = (image / 255.0 - 0.5) / 0.5  # Normalize to range [-1, 1]
    image = np.transpose(image, (2, 0, 1))  # Convert to C x H x W
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return image

# Streamlit app
st.title("Face Recognition App")
st.write("Upload an image to identify the closest match from the dataset.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and preprocess the image
    img = Image.open(uploaded_file)
    img_tensor = preprocess_image(img)
    
    # Generate embedding for the uploaded image
    with torch.no_grad():
        query_embedding = model(img_tensor).numpy().flatten()
    
    # Search for the closest match
    D, I = index.search(np.expand_dims(query_embedding, axis=0), k=1)
    
    # Find the closest match label
    closest_index = I[0][0]
    closest_label = labels[closest_index]
    
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"Closest match: {closest_label}")
