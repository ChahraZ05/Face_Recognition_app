import streamlit as st
import faiss
import torch
import numpy as np
import pickle
import cv2
from facenet_pytorch import InceptionResnetV1
from PIL import Image

# Load FAISS index and embeddings
index = faiss.read_index('faiss_index.index')

# Load the embeddings dictionary
with open('lfw_embeddings.pkl', 'rb') as f:
    all_embeddings = pickle.load(f)

# Initialize InceptionResnetV1 model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Preprocess image using OpenCV
def preprocess_image(img):
    img = cv2.resize(img, (160, 160))  # Resize to 160x160
    img = img / 255.0  # Normalize pixel values
    img = np.transpose(img, (2, 0, 1))  # Change to (C, H, W)
    img_tensor = torch.tensor([img], dtype=torch.float32)  # Add batch dimension
    return img_tensor

# Function to perform face embedding and matching
def identify_face(image):
    # Convert PIL image to OpenCV format
    img = np.array(image)

    # Preprocess image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = preprocess_image(img)

    # Generate embedding
    with torch.no_grad():
        query_embedding = model(img_tensor).numpy().flatten()

    # Search for the top k matches in FAISS index
    k = 5  # Increase the number of neighbors
    D, I = index.search(np.expand_dims(query_embedding, axis=0), k=k)
    
    # Debug: Print distances and indices
    st.write(f"Distances: {D}")
    st.write(f"Indices: {I}")
    
    # Retrieve the names of the closest matches
    matched_persons = []
    for idx in I[0]:
        current_index = 0
        for person, embeddings in all_embeddings.items():
            if current_index <= idx < current_index + len(embeddings):
                matched_persons.append(person)
                break
            current_index += len(embeddings)

    # Debug: Print matched persons
    st.write(f"Matched Persons: {matched_persons}")
    
    # Majority voting to determine the closest match
    if matched_persons:
        closest_person = max(set(matched_persons), key=matched_persons.count)
        return closest_person
    else:
        return "No match found."

# Streamlit app UI
st.title("Face Recognition App")
st.write("Upload an image to identify the person.")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process and identify the face in the image
    result = identify_face(image)
    st.write(f"Closest match: {result}")
