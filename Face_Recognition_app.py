from PIL import Image
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

# Load the model
model = InceptionResnetV1(pretrained='vggface2').eval()

def preprocess_image_pillow(img):
    img = img.convert('RGB')  # Ensure image is in RGB mode
    img = img.resize((160, 160))  # Resize image
    img = np.array(img).transpose((2, 0, 1))  # Change from HWC to CHW
    img = torch.tensor(img).float().unsqueeze(0) / 255.0
    return img

def recognize_face(image):
    img_tensor = preprocess_image_pillow(image)
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

# Example usage
from io import BytesIO
uploaded_file = BytesIO(b'your_image_binary_data')
img = Image.open(uploaded_file)
result = recognize_face(img)
print("Top Matches:", result)
