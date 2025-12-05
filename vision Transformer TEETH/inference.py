# dental_inference_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os

# ==============================
# CLASS MAPPING
# ==============================
CLASS_MAPPING = {
    'CaS': 'Canker sores',
    'CoS': 'Cold sores',
    'Gum': 'Gingivostomatitis',
    'MC': 'Mouth cancer',
    'OC': 'Oral cancer',
    'OLP': 'Oral lichen planus',
    'OT': 'Oral thrush'
}

# ==============================
# MODEL ARCHITECTURE
# ==============================
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
        patches = patches.permute(0, 2, 1)
        patches = patches.view(B, self.num_patches, -1)
        tokens = self.proj(patches)
        return tokens

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
        self.temperature = nn.Parameter(torch.ones(num_heads,1,1))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn * self.temperature
        attn = attn - attn.mean(dim=-1, keepdim=True)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
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
    def __init__(self, img_size=72, patch_size=6, embed_dim=64, depth=4, num_heads=4, mlp_dim=128, num_classes=7, dropout=0.1):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # ✅ Use same names as in trained model
        self.shifted_patch_tokenization = ShiftedPatchTokenization(img_size, patch_size, 3, embed_dim)
        self.patch_encoder = PatchEncoder(self.num_patches, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*self.num_patches, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024,num_classes)
        )

    def forward(self, x):
        tokens = self.shifted_patch_tokenization(x)
        tokens = self.patch_encoder(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens)

# ==============================
# INFERENCE FUNCTION
# ==============================
def predict(model_path, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisionTransformer()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((72,72)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred_idx = output.argmax(dim=1).item()
        pred_class = list(CLASS_MAPPING.values())[pred_idx]
        confidence = probs[0, pred_idx].item()

    print(f"\nPredicted Disease: {pred_class}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nAll Class Probabilities:")
    for cls, prob in zip(CLASS_MAPPING.values(), probs[0]):
        print(f"  {cls}: {prob.item()*100:.2f}%")

# ==============================
# COMMAND LINE
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to trained model .pth')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print("❌ Model path not found!")
        exit(1)
    if not os.path.exists(args.image):
        print("❌ Image path not found!")
        exit(1)

    predict(args.model, args.image)
