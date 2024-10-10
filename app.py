import streamlit as st
import torch
from PIL import Image
from torchvision import models,transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_labels = {0: 'Apple___alternaria_leaf_spot',
 1: 'Apple___brown_spot',
 2: 'Apple___gray_spot',
 3: 'Apple___healthy',
 4: 'Apple___rust',
 5: 'Cherry___healthy',
 6: 'Cherry___powdery_mildew',
 7: 'Potato___early_blight',
 8: 'Potato___healthy',
 9: 'Potato___late_blight',
 10: 'Squash___powdery_mildew',
 11: 'Tomato___bacterial_spot',
 12: 'Tomato___early_blight',
 13: 'Tomato___healthy',
 14: 'Tomato___late_blight',
 15: 'Tomato___septoria_leaf_spot'}

model = models.efficientnet_b3(pretrained = False)
# model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 16)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3, inplace=True), 
    torch.nn.Linear(in_features=model.classifier[1].in_features, 
                    out_features=len(class_labels),
                    bias=True)
).to(device)

model.load_state_dict(torch.load('01_Model_PlantVillage_0.pth',map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])


# Title and description
st.title("Neem Leaf Disease Detection Web App")
st.write("Upload an image for prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:

    # Open and preprocess the image
    img = Image.open(uploaded_file)
    st.image(img, caption='The image is successfully detected.', use_column_width=True)
    img_tensor = transform(img).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    st.write(f'The predicted image is: {class_labels[predicted.item()]}')




