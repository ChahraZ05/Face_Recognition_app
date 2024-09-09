import streamlit as st
import cv2
import numpy as np
import torch
import pickle
import faiss
from facenet_pytorch import InceptionResnetV1

# Load the model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load FAISS index
index = faiss.read_index('/kaggle/working/faiss_index.index')

# Load embeddings and associate them with their respective person names
with open('/kaggle/working/embeddings.pkl', 'rb') as f:
    all_embeddings = pickle.load(f)

# Define preprocessing function
def preprocess_image(img):
    # Convert image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to 160x160
    img = cv2.resize(img, (160, 160))
    
    # Convert image to tensor
    img = torch.tensor(img, dtype=torch.float32)
    img = img.permute(2, 0, 1)  # Change to (C, H, W)
    img = img.unsqueeze(0)  # Add batch dimension
    
    # Normalize image
    img = (img / 255.0 - 0.5) / 0.5
    return img

# Function to find the closest person match
def find_closest_person(embedding, k=1):
    # Search in the FAISS index
    D, I = index.search(np.expand_dims(embedding, axis=0), k=k)  # Search for top k matches
    
    # Map the index to the person name
    current_count = 0
    for person, embeddings in all_embeddings.items():
        if I[0][0] < current_count + len(embeddings):
            return person, D[0][0]  # Return the closest person and the distance
        current_count += len(embeddings)
    
    return None, None

# Streamlit UI
st.title("Face Recognition System")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess the image
    img_tensor = preprocess_image(img)
    
    # Generate face embedding
    with torch.no_grad():
        query_embedding = model(img_tensor).numpy().flatten()
    
    # Find the closest match
    closest_person, distance = find_closest_person(query_embedding)
    
    # Display result
    if closest_person:
        st.write(f"Closest Match: {closest_person} (Distance: {distance:.2f})")
    else:
        st.write("No match found.")
