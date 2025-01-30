import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# 1. Load the model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model(model_path):
    model = models.efficientnet_b1(weights=None)  # Initialize the model without pre-trained weights
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)  # Adjust for your number of classes
    
    # Load the state dictionary
    state_dict = torch.load(
        model_path, 
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()  # Set the model to evaluation mode
    return model

# 2. Define the image preprocessing pipeline
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# 3. Predict the class of the input image
def predict(model, image_tensor, class_labels):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
        return class_labels[predicted.item()]  # Return the class label

# 4. Streamlit App Layout
st.title("Brain Tumor Classification Using Deep Learning")
st.write("Upload an image to classify it.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Load model
    model_path = "efficientnet_b1_epoch_9.pth"
    model = load_model(model_path)

    # Preprocess the image
    image_tensor = preprocess_image(image)

    # Define class labels (update as per your dataset)
    class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    # Perform prediction
    prediction = predict(model, image_tensor, class_labels)

    # Display the result in the center of the screen
    st.markdown(f"<h2 style='text-align: center;'>Prediction is: {prediction}</h2>", unsafe_allow_html=True)