# vit_skin_inference_updated.py
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse

# ---------------------------
# Model definition (must match training)
# ---------------------------
class ShiftedPatchTokenization(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.shift_size = patch_size // 2
        self.embed_dim = embed_dim
        self.proj = nn.Linear(5 * in_chans * patch_size ** 2, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.shift_size
        x_padded = F.pad(x, (p, p, p, p))
        orig = x_padded[:, :, p:-p, p:-p]
        left = x_padded[:, :, p:-p, 0:-2*p]
        right = x_padded[:, :, p:-p, 2*p:]
        up = x_padded[:, :, 0:-2*p, p:-p]
        down = x_padded[:, :, 2*p:, p:-p]
        x_cat = torch.cat([orig, left, right, up, down], dim=1)
        patches = F.unfold(x_cat, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.permute(0, 2, 1).contiguous()
        patches = patches.view(B, self.num_patches, -1)
        tokens = self.proj(patches)
        return tokens, patches

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
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn * self.temperature
        attn = attn - attn.mean(dim=-1, keepdim=True)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionLSA(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=72, patch_size=6, in_channels=3, num_classes=10,
                 embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.shifted_patch_tokenization = ShiftedPatchTokenization(img_size, patch_size, in_channels, embed_dim)
        self.patch_encoder = PatchEncoder(self.num_patches, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
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
        tokens, _ = self.shifted_patch_tokenization(x)
        tokens = self.patch_encoder(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens)

# ---------------------------
# Load checkpoint and class mapping
# ---------------------------
def load_checkpoint(model_path, device):
    ckpt = torch.load(model_path, map_location=device)

    num_classes = None
    if 'classes' in ckpt:
        num_classes = len(ckpt['classes'])
    elif 'clean_class_names' in ckpt:
        num_classes = len(ckpt['clean_class_names'])
    else:
        num_classes = 10

    img_size = ckpt.get('img_size', 72)
    patch_size = ckpt.get('patch_size', 6)
    model = VisionTransformer(img_size=img_size, patch_size=patch_size, num_classes=num_classes)
    state_key = 'model_state_dict' if 'model_state_dict' in ckpt else 'model_state'
    model.load_state_dict(ckpt.get(state_key, ckpt), strict=False)
    model.to(device).eval()

    # Read class mapping
    if 'class_mapping' in ckpt:
        class_mapping = ckpt['class_mapping']
        # Ensure ordered list
        class_names = [class_mapping.get(cls, cls) for cls in ckpt['classes']]
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]

    return model, class_names

# ---------------------------
# Transforms and prediction
# ---------------------------
_transform = transforms.Compose([
    transforms.Resize((72, 72)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_image(model, img_path, device, class_names, topk=3):
    img = Image.open(img_path).convert('RGB')
    x = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().squeeze(0)
    topk_probs, topk_idx = torch.topk(probs, k=min(topk, probs.numel()))
    return [(class_names[i], float(p)) for i,p in zip(topk_idx.tolist(), topk_probs.tolist())]

# ---------------------------
# CLI / main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="ViT Skin Disease Inference")
    parser.add_argument('--model', '-m', required=True, help="Path to trained .pth checkpoint")
    parser.add_argument('--image', '-i', help="Single image path")
    parser.add_argument('--folder', '-f', help="Folder path for batch inference")
    parser.add_argument('--topk', type=int, default=3, help="Top-K predictions")
    parser.add_argument('--out_csv', default='predictions_skin.csv', help="CSV output for batch")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_checkpoint(args.model, device)
    print(f"✅ Model loaded on {device}. Classes: {class_names}")

    if args.image:
        if not os.path.exists(args.image):
            raise SystemExit(f"Image not found: {args.image}")
        results = predict_image(model, args.image, device, class_names, topk=args.topk)
        print(f"\nPredictions for {args.image}:")
        for name, p in results:
            print(f"  {name}: {p*100:.2f}%")
    elif args.folder:
        if not os.path.isdir(args.folder):
            raise SystemExit(f"Folder not found: {args.folder}")
        image_files = [os.path.join(args.folder,f) for f in sorted(os.listdir(args.folder))
                       if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
        if not image_files:
            raise SystemExit("No images found in folder.")
        rows = []
        for img_path in image_files:
            results = predict_image(model, img_path, device, class_names, topk=args.topk)
            top_name, top_p = results[0]
            rows.append({'image': os.path.basename(img_path), 'predicted': top_name, 'confidence': float(top_p)})
            print(f"{os.path.basename(img_path)} -> {top_name} ({top_p*100:.2f}%)")
        with open(args.out_csv, 'w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=['image','predicted','confidence'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✅ Batch predictions saved to {args.out_csv}")
    else:
        print("Provide --image <path> or --folder <path> to run inference.")

import matplotlib.pyplot as plt

def predict_image(model, img_path, device, class_names, topk=3):
    img = Image.open(img_path).convert('RGB')
    x = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().squeeze(0)
    topk_probs, topk_idx = torch.topk(probs, k=min(topk, probs.numel()))
    results = [(class_names[i], float(p)) for i,p in zip(topk_idx.tolist(), topk_probs.tolist())]
    
    # Display image with top prediction
    top_name, top_score = results[0]
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {top_name}  ({top_score*100:.2f}%)")
    plt.show()

    return results


if __name__ == "__main__":
    main()
