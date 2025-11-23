# vit_eye_disease_inference.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import json
import os
import matplotlib.pyplot as plt

# ========= ARCHITECTURE (EXACTLY AS TRAINING) ========= #

class ShiftedPatchTokenization(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.shift_size = patch_size // 2
        self.embed_dim = embed_dim

        # Projection layer
        self.proj = nn.Linear(5 * in_chans * patch_size ** 2, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.shift_size
        x_padded = F.pad(x, (p, p, p, p))

        # Extract shifted images
        orig = x_padded[:, :, p:-p, p:-p]
        left = x_padded[:, :, p:-p, 0:-2*p]
        right = x_padded[:, :, p:-p, 2*p:]
        up = x_padded[:, :, 0:-2*p, p:-p]
        down = x_padded[:, :, 2*p:, p:-p]

        # Concatenate shifted images
        x_cat = torch.cat([orig, left, right, up, down], dim=1)

        # Extract patches
        patches = F.unfold(x_cat, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.permute(0, 2, 1)
        patches = patches.view(B, self.num_patches, -1)

        # Project to embedding dimension
        tokens = self.proj(patches)
        spatial = patches.view(B, self.grid_size, self.grid_size, -1)

        return tokens, spatial


class PatchEncoder(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed


class MultiHeadAttentionLSA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn * self.temperature

        # LSA specific: subtract mean and apply softmax
        attn = attn - attn.mean(dim=-1, keepdim=True)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        # Attention block
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionLSA(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        # MLP block
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention with residual
        x_attn = self.attn(self.norm1(x))
        x = x + self.dropout1(x_attn)

        # MLP with residual
        x_mlp = self.mlp(self.norm2(x))
        x = x + x_mlp
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=72, patch_size=6, in_channels=3, num_classes=4,
                 embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.num_patches = (img_size // patch_size) ** 2

        # Tokenization and encoding
        self.shifted_patch_tokenization = ShiftedPatchTokenization(
            img_size, patch_size, in_channels, embed_dim
        )
        self.patch_encoder = PatchEncoder(self.num_patches, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head (same as training)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * self.num_patches, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # Tokenization and positional encoding
        tokens, _ = self.shifted_patch_tokenization(x)
        tokens = self.patch_encoder(tokens)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Classification
        tokens = self.norm(tokens)
        return self.head(tokens)

# ========= PREDICTOR WRAPPER ========= #

class EyeDiseasePredictor:
    def __init__(self, model_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)

        self.classes = checkpoint['classes']

        # Recreate model with same hyperparams
        self.model = VisionTransformer(
            img_size=72,
            patch_size=6,
            in_channels=3,
            num_classes=len(self.classes),
            embed_dim=64,
            depth=4,
            num_heads=4,
            mlp_dim=128,
            dropout=0.1
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Same transforms as training/validation
        self.transform = transforms.Compose([
            transforms.Resize((72, 72)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        print(f"Model loaded on {self.device}. Classes: {self.classes}")

    def predict_image(self, image_path):
        """Predict a single image"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        prediction = self.classes[predicted_idx.item()]
        confidence = confidence.item()

        class_probs = {
            self.classes[i]: float(probabilities[0][i])
            for i in range(len(self.classes))
        }

        return {
            'predicted_class': prediction,
            'confidence': confidence,
            'all_probabilities': class_probs,
            'top_prediction': f"{prediction} ({confidence:.2%})"
        }

    def predict_batch(self, image_paths):
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        return results

# ========= MAIN ========= #

def main():
    parser = argparse.ArgumentParser(description='Eye Disease Classification Inference')
    parser.add_argument('--model_path', type=str, default='vit_eye_disease.pth',
                        help='Path to trained model weights')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to single image or directory of images')
    parser.add_argument('--output', type=str, default='predictions.json',
                        help='Output file for batch predictions (JSON)')
    args = parser.parse_args()

    predictor = EyeDiseasePredictor(args.model_path)

    # ---------- Single image ---------- #
    if os.path.isfile(args.image_path):
        result = predictor.predict_image(args.image_path)

        # Show image with prediction
        img = Image.open(args.image_path).convert('RGB')
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"{result['predicted_class']} ({result['confidence']:.2%})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Console output
        print("\n=== Prediction Result ===")
        print(f"Image: {args.image_path}")
        print(f"Predicted: {result['top_prediction']}")
        print("\nAll probabilities:")
        for disease, prob in result['all_probabilities'].items():
            print(f"  {disease}: {prob:.2%}")

    # ---------- Directory (batch) ---------- #
    elif os.path.isdir(args.image_path):
        image_files = [
            os.path.join(args.image_path, f)
            for f in os.listdir(args.image_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]

        if not image_files:
            print(f"No image files found in {args.image_path}")
            return

        print(f"Found {len(image_files)} images. Processing...")
        results = predictor.predict_batch(image_files)

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nBatch prediction completed. Results saved to {args.output}")

        successful = [r for r in results if 'predicted_class' in r]
        print(f"Successfully processed: {len(successful)}/{len(image_files)}")

        if successful:
            print("\nTop predictions summary:")
            for result in successful[:5]:
                print(f"  {os.path.basename(result['image_path'])}: {result['top_prediction']}")
            if len(successful) > 5:
                print(f"  ... and {len(successful) - 5} more")

    else:
        print(f"Error: {args.image_path} is not a valid file or directory")


if __name__ == '__main__':
    main()
