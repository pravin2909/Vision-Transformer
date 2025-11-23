# =============================================
# 1️⃣ Install required packages
# =============================================


# =============================================
# 2️⃣ Import librar# =============================================
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import os

# =============================================
# 3️⃣ Set device
# =============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================
# 4️⃣ Initialize ViT model architecture
# =============================================
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=10  # 10 classes in your dataset
)

# Load your fine-tuned weights
model.load_state_dict(torch.load("vit_skin_disease_model.pth", map_location=device))
model.to(device)
model.eval()

# Initialize feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# =============================================
# 5️⃣ Define class labels
# =============================================
labels = [
    "Eczema", "Warts", "Melanoma", "Atopic Dermatitis", 
    "Basal Cell Carcinoma", "Melanocytic Nevi", 
    "Benign Keratosis-like Lesions", "Psoriasis", 
    "Seborrheic Keratoses", "Fungal Infections"
]

# =============================================
# 6️⃣ Define prediction function
# =============================================
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_idx = logits.argmax(-1).item()
    
    return labels[pred_idx]

# =============================================
# 7️⃣ Run inference on a single image
# =============================================
image_path = "ezema.jpg"  # replace with your image path
predicted_class = predict(image_path)
print(f"Predicted class for '{image_path}': {predicted_class}")

# =============================================
# 8️⃣ Optional: Run inference on all images in a folder
# =============================================
folder_path = "images_folder"  # replace with your folder path
for img_file in os.listdir(folder_path):
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(folder_path, img_file)
        pred = predict(img_path)
        print(f"{img_file} --> {pred}")
