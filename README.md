# Vision-Transformer

This repository contains three Vision Transformer (ViT) based medical imaging models designed to detect diseases from different parts of the human body:

- **ViT-Skin** – analyzes skin images to classify skin conditions.
- **ViT-Eye** – analyzes retinal images to detect eye diseases.
- **ViT-Tooth** – analyzes dental X-rays to detect tooth and gum issues.

## Project Overview

The goal of this project is to build a multimodal medical AI system using Vision Transformers, where each model specializes in one type of medical image:

- **Skin model (ViT-Skin)**: Works on skin photos (dermoscopy or normal camera images) to classify lesions and skin conditions.
- **Eye model (ViT-Eye)**: Works on retinal fundus or OCT images to detect retinal diseases.
- **Tooth model (ViT-Tooth)**: Works on dental X-ray images to detect dental problems.

## Models

### 1. ViT-Skin (Dermatology)

- **Input**: Skin images (lesions, moles, rashes, acne, etc.).
- **Task**: Classify different skin conditions and help in early detection of diseases.
- **Possible outputs** (examples):
  - Benign vs malignant lesion
  - Dermatitis / Eczema
  - Acne severity grade
  - Infection or inflammation probability

### 2. ViT-Eye (Ophthalmology)

- **Input**: Retinal fundus images or OCT scans.
- **Task**: Detect and grade eye diseases.
- **Possible outputs** (examples):
  - Diabetic Retinopathy stage
  - Glaucoma risk
  - Macular degeneration severity
  - Other retinal abnormalities

### 3. ViT-Tooth (Dental Imaging)

- **Input**: Dental X-ray images (panoramic or periapical).
- **Task**: Detect dental and gum-related problems.
- **Possible outputs** (examples):
  - Cavity (caries) probability
  - Root infection / apical lesion
  - Gum disease signs
  - Alignment or structural issues

## Tech Stack

- **Model architecture**: Vision Transformer (ViT)
- **Language**: Python
- **Frameworks**: (e.g., PyTorch / TensorFlow – update based on your implementation)
- **Data**: Medical imaging datasets for skin, eye, and dental images (public or custom)

## How to Run

Update this section based on your code structure. Example template:

```bash
# 1. Create and activate environment
conda create -n vit-med python=3.10 -y
conda activate vit-med

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train or run inference (examples)
python train_skin.py     # Train ViT-Skin
python train_eye.py      # Train ViT-Eye
python train_tooth.py    # Train ViT-Tooth

python infer_skin.py --image path/to/skin_image.jpg
python infer_eye.py  --image path/to/eye_image.png
python infer_tooth.py --image path/to/dental_xray.png
```

## Folder Structure (example)

Update this to match your repository layout.

```text
Vision-Transformer/
├── data/
│   ├── skin/
│   ├── eye/
│   └── tooth/
├── models/
│   ├── vit_skin.py
│   ├── vit_eye.py
│   └── vit_tooth.py
├── train_skin.py
├── train_eye.py
├── train_tooth.py
├── infer_skin.py
├── infer_eye.py
├── infer_tooth.py
├── requirements.txt
└── README.md
```

## Future Work

- Add more disease classes for each body part.
- Improve model performance with better datasets and augmentation.
- Build a simple web/app interface where users can upload an image and get predictions from the correct model (skin, eye, or tooth).

---
