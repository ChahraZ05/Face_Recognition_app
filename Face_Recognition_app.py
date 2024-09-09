import streamlit as st
import torch
from PIL import Image
import numpy as np
import faiss
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN

# Load the pre-trained model and FAISS index
model = InceptionResnetV1(pretrained='vggface2').eval()
index = faiss.read_index('faiss_index.index')

# Load the embeddings
with open('lfw_embeddings.pkl', 'rb') as f:
    all_embeddings = pickle.load(f)

def preprocess_image(img):
    try:
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        img = cv2.resize(img, (160, 160))  # Resize to match model input size
        img = (img / 255.0).astype(np.float32)  # Normalize to [0, 1]
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # Convert to tensor and add batch dimension
        return img
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def find_similar_faces(img_tensor):
    try:
        # Generate the embedding for the uploaded image
        with torch.no_grad():
            query_embedding = model(img_tensor).numpy().flatten()
        # Search for similar faces
        D, I = index.search(np.expand_dims(query_embedding, axis=0), k=5)
        return I[0]
    except Exception as e:
        st.error(f"Error finding similar faces: {e}")
        return None

# Streamlit app interface
st.title("Face Recognition App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    st.write("File uploaded successfully")
    st.write(f"File size: {uploaded_file.size} bytes")
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        img_tensor = preprocess_image(img)
        
        if img_tensor is not None:
            matches = find_similar_faces(img_tensor)
            if matches is not None:
                matched_persons = [list(all_embeddings.keys())[i] for i in matches]
                st.write(f"Top Matches: {', '.join(matched_persons)}")
            else:
                st.write("No matches found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
