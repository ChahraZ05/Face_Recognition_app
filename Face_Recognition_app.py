import streamlit as st
import numpy as np
import cv2
import torch
from facenet_pytorch import InceptionResnetV1

# Load the model
model = InceptionResnetV1(pretrained='vggface2').eval()

def preprocess_image_cv2(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    img = np.transpose(img, (2, 0, 1))  # Change from HWC to CHW
    img = torch.tensor(img).float().unsqueeze(0) / 255.0
    return img

def recognize_face(image):
    img_tensor = preprocess_image_cv2(image)
    with torch.no_grad():
        query_embedding = model(img_tensor).numpy().flatten()

    # Assuming FAISS index is loaded here as 'index'
    D, I = index.search(np.expand_dims(query_embedding, axis=0), k=5)
    matched_persons = []
    for i in I[0]:
        for person, embeddings in all_embeddings.items():
            if i < len(embeddings):
                matched_persons.append(person)
                break
    return matched_persons if matched_persons else "No matches found"

st.title("Face Recognition App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    result = recognize_face(img)
    st.write("Top Matches:", result)
