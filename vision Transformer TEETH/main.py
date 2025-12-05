# dental_disease_vit.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
from PIL import Image
import argparse

# ==============================
# CONFIGURATION
# ==============================
class Config:
    def __init__(self):
        self.data_dir = 'Teeth_Dataset'  # Change this to your dataset path
        self.batch_size = 8
        self.epochs = 20
        self.lr = 3e-4
        self.weight_decay = 1e-2
        self.num_workers = 4
        self.output_dir = 'checkpoints'
        self.model_name = 'vit_dental_disease.pth'
        self.checkpoint_name = 'checkpoint.pth'  # For resuming training
        self.val_split = 0.2
        self.test_split = 0.1
        self.img_size = 72
        self.patch_size = 6
        self.embed_dim = 64
        self.depth = 4
        self.num_heads = 4
        self.mlp_dim = 128
        self.dropout = 0.1

# Dental disease classes
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
# VISION TRANSFORMER ARCHITECTURE
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
        return tokens, patches.view(B, self.grid_size, self.grid_size, -1)

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
        x_attn = self.attn(self.norm1(x))
        x = x + self.dropout1(x_attn)
        x_mlp = self.mlp(self.norm2(x))
        x = x + x_mlp
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        img_size = config.img_size
        patch_size = config.patch_size
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        
        self.num_patches = (img_size // patch_size) ** 2

        # Tokenization and encoding
        self.shifted_patch_tokenization = ShiftedPatchTokenization(
            img_size, patch_size, 3, config.embed_dim
        )
        self.patch_encoder = PatchEncoder(self.num_patches, config.embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.mlp_dim, config.dropout)
            for _ in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim * self.num_patches, 2048),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        tokens, _ = self.shifted_patch_tokenization(x)
        tokens = self.patch_encoder(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens)

# ==============================
# DATA UTILITIES
# ==============================
def setup_directories():
    """Create necessary directories"""
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)

def verify_dataset_structure(data_dir):
    """Verify the dataset structure"""
    print("🔍 Verifying dataset structure...")
    
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory not found: {data_dir}")
        return False
    
    splits = ['Training', 'Validation', 'Testing']
    for split in splits:
        split_path = os.path.join(data_dir, split)
        if os.path.exists(split_path):
            classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            print(f"\n📁 {split}: {len(classes)} classes found")
            for class_name in classes:
                class_path = os.path.join(split_path, class_name)
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                clean_name = CLASS_MAPPING.get(class_name, class_name)
                print(f"   {class_name} ({clean_name}): {len(images)} images")
        else:
            print(f"⚠️  {split} directory not found, will create split from Training data")
    
    return True

def get_data_loaders(config):
    """Create data loaders for training, validation and testing"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Simple transform for validation and testing
    val_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Check if we have separate validation and testing splits
    train_path = os.path.join(config.data_dir, 'Training')
    val_path = os.path.join(config.data_dir, 'Validation')
    test_path = os.path.join(config.data_dir, 'Testing')
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        # Use existing splits
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
        test_dataset = datasets.ImageFolder(test_path, transform=val_transform)
        
        # Get classes from training dataset
        classes = train_dataset.classes
        
        print("✅ Using existing dataset splits")
        
    else:
        # Create splits from training data
        print("📊 Creating train/val/test splits from Training directory...")
        full_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        
        # Get classes before splitting
        classes = full_dataset.classes
        
        # Calculate split sizes
        total_size = len(full_dataset)
        test_size = int(total_size * config.test_split)
        val_size = int(total_size * config.val_split)
        train_size = total_size - val_size - test_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # Apply val transform to validation and test sets
        val_dataset.dataset.transform = val_transform
        test_dataset.dataset.transform = val_transform
        
        print("✅ Created splits from Training directory")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, classes

# ==============================
# TRAINING UTILITIES
# ==============================
def build_model(config, num_classes):
    return VisionTransformer(config, num_classes)

def train_epoch(model, loader, criterion, optimizer, device, epoch, writer):
    model.train()
    running_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (imgs, lbls) in enumerate(pbar):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/total:.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
        
        # Log to tensorboard
        if writer and batch_idx % 10 == 0:
            writer.add_scalar('Train/Loss_batch', loss.item(), epoch * len(loader) + batch_idx)
            writer.add_scalar('Train/Acc_batch', 100. * (preds == lbls).float().mean(), 
                            epoch * len(loader) + batch_idx)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    if writer:
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Acc', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device, epoch, writer):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=f'Epoch {epoch} [Val]'):
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss = criterion(out, lbls)
            
            running_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    if writer:
        writer.add_scalar('Val/Loss', epoch_loss, epoch)
        writer.add_scalar('Val/Acc', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

def test_model(model, loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc='Testing'):
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

# ==============================
# VISUALIZATION UTILITIES
# ==============================
def plot_confusion_matrix(y_true, y_pred, classes, class_mapping, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # Map class names
    clean_classes = [class_mapping.get(cls, cls) for cls in classes]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clean_classes, 
                yticklabels=clean_classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_training_metrics(train_losses, val_losses, train_accs, val_accs, save_path):
    """Save training metrics to JSON"""
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==============================
# INFERENCE CLASS
# ==============================
class DentalDiseasePredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load class info
        self.classes = self.checkpoint['classes']
        self.class_mapping = self.checkpoint['class_mapping']
        self.clean_classes = [self.class_mapping.get(cls, cls) for cls in self.classes]
        
        # Create model
        config = Config()
        self.model = VisionTransformer(config, len(self.classes))
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((72, 72)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded from {model_path}")
        print(f"🎯 Classes: {self.clean_classes}")
    
    def predict(self, image_path):
        """Predict disease from image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': self.clean_classes[predicted_class],
            'confidence': confidence,
            'all_probabilities': {
                cls: prob.item() for cls, prob in zip(self.clean_classes, probabilities[0])
            }
        }
    
    def predict_batch(self, image_paths):
        """Predict multiple images"""
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result['image_path'] = image_path
            results.append(result)
        return results

# ==============================
# CHECKPOINT MANAGEMENT
# ==============================
def save_checkpoint(config, model, optimizer, scheduler, epoch, best_val_acc, 
                   train_losses, val_losses, train_accs, val_accs, classes):
    """Save training checkpoint"""
    checkpoint_path = os.path.join(config.output_dir, config.checkpoint_name)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'classes': classes,
        'class_mapping': CLASS_MAPPING,
        'config': config.__dict__,
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'last_train_loss': train_losses[-1] if train_losses else 0,
        'last_val_loss': val_losses[-1] if val_losses else 0,
        'last_train_acc': train_accs[-1] if train_accs else 0,
        'last_val_acc': val_accs[-1] if val_accs else 0
    }, checkpoint_path)
    
    return checkpoint_path

def load_checkpoint(config, device):
    """Load training checkpoint if exists"""
    checkpoint_path = os.path.join(config.output_dir, config.checkpoint_name)
    
    if os.path.exists(checkpoint_path):
        print(f"🔄 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        return checkpoint
    else:
        print("🆕 No checkpoint found, starting from scratch")
        return None

def save_best_model(config, model, optimizer, scheduler, epoch, best_val_acc, 
                   train_loss, val_loss, train_acc, val_acc, classes):
    """Save the best model"""
    model_path = os.path.join(config.output_dir, config.model_name)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'classes': classes,
        'class_mapping': CLASS_MAPPING,
        'config': config.__dict__,
        'best_val_acc': best_val_acc,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }, model_path)
    
    return model_path

# ==============================
# MAIN TRAINING FUNCTION
# ==============================
def train_model(resume_training=False):
    """Main training function with resume capability"""
    config = Config()
    setup_directories()
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    if device.type == 'cuda':
        print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
    
    # Verify dataset
    if not verify_dataset_structure(config.data_dir):
        print("❌ Please check your dataset structure and try again.")
        return
    
    # Get data loaders
    train_loader, val_loader, test_loader, classes = get_data_loaders(config)
    
    print(f"\n📊 Dataset Info:")
    print(f"   Classes: {len(classes)} - {classes}")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    print(f"   Testing samples: {len(test_loader.dataset)}")
    
    # Build model
    model = build_model(config, len(classes)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔨 Model Parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Tensorboard writer
    writer = SummaryWriter('logs/dental_vit')
    
    # Training state
    start_epoch = 1
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Load checkpoint if resuming
    if resume_training:
        checkpoint = load_checkpoint(config, device)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            train_accs = checkpoint['train_accs']
            val_accs = checkpoint['val_accs']
            
            print(f"🔄 Resuming from epoch {start_epoch}")
            print(f"📈 Previous best validation accuracy: {best_val_acc:.4f}")
    
    # Training loop
    print(f"\n🚀 Starting training for {config.epochs} epochs...")
    print(f"📅 Starting from epoch: {start_epoch}")
    
    try:
        for epoch in range(start_epoch, config.epochs + 1):
            print(f"\n" + "="*50)
            print(f"Epoch {epoch}/{config.epochs}")
            print("="*50)
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, writer
            )
            
            # Validate
            val_loss, val_acc = validate_epoch(
                model, val_loader, criterion, device, epoch, writer
            )
            
            # Update scheduler
            scheduler.step()
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f"\n📈 Epoch {epoch} Results:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"   Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
            print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint after every epoch
            checkpoint_path = save_checkpoint(
                config, model, optimizer, scheduler, epoch, best_val_acc,
                train_losses, val_losses, train_accs, val_accs, classes
            )
            print(f"💾 Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = save_best_model(
                    config, model, optimizer, scheduler, epoch, best_val_acc,
                    train_loss, val_loss, train_acc, val_acc, classes
                )
                print(f"🏆 Saved best model with validation accuracy: {best_val_acc:.4f}")
        
        # Close tensorboard writer
        writer.close()
        
        print(f"\n🎉 Training completed!")
        print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
        
        # Load best model for testing
        best_model_path = os.path.join(config.output_dir, config.model_name)
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded best model for testing")
        
        # Test the model
        print("\n🧪 Testing model...")
        y_pred, y_true, y_probs = test_model(model, test_loader, device, classes)
        
        # Calculate metrics
        test_accuracy = (y_pred == y_true).mean()
        print(f"\n📊 Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        clean_classes = [CLASS_MAPPING.get(cls, cls) for cls in classes]
        print(f"\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=clean_classes))
        
        # Save results
        print("\n💾 Saving results...")
        
        # Plot confusion matrix
        plot_confusion_matrix(y_true, y_pred, classes, CLASS_MAPPING, 
                            'results/confusion_matrix.png')
        
        # Plot training curves
        plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                            'results/training_curves.png')
        
        # Save metrics
        save_training_metrics(train_losses, val_losses, train_accs, val_accs,
                            'results/training_metrics.json')
        
        # Save test predictions
        test_results = {
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'y_probs': y_probs.tolist(),
            'classes': classes,
            'class_mapping': CLASS_MAPPING,
            'test_accuracy': float(test_accuracy)
        }
        
        with open('results/test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print("✅ All results saved in 'results/' directory")
        print("🎉 Training and evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user at epoch {epoch}")
        print("💾 Saving final checkpoint before exit...")
        
        # Save final checkpoint
        checkpoint_path = save_checkpoint(
            config, model, optimizer, scheduler, epoch, best_val_acc,
            train_losses, val_losses, train_accs, val_accs, classes
        )
        print(f"✅ Checkpoint saved: {checkpoint_path}")
        print(f"📊 Training progress saved. You can resume later with: --resume")
        writer.close()

# ==============================
# COMMAND LINE INTERFACE
# ==============================
def main():
    parser = argparse.ArgumentParser(description='Dental Disease Classification with Vision Transformer')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'predict-batch', 'resume'], 
                       default='train', help='Mode: train, predict, predict-batch, or resume')
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    parser.add_argument('--folder', type=str, help='Path to folder for batch prediction')
    parser.add_argument('--model', type=str, default='checkpoints/vit_dental_disease.pth', 
                       help='Path to trained model for inference')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume training from last checkpoint')
    
    args = parser.parse_args()
    
    if args.mode == 'train' or args.mode == 'resume' or args.resume:
        train_model(resume_training=(args.mode == 'resume' or args.resume))
    
    elif args.mode == 'predict':
        if not args.image:
            print("❌ Please provide --image path for prediction")
            return
        
        predictor = DentalDiseasePredictor(args.model)
        result = predictor.predict(args.image)
        
        print(f"\n🔍 Prediction for {args.image}:")
        print(f"   Disease: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"\n📊 All probabilities:")
        for disease, prob in result['all_probabilities'].items():
            print(f"   {disease}: {prob:.4f}")
    
    elif args.mode == 'predict-batch':
        if not args.folder:
            print("❌ Please provide --folder path for batch prediction")
            return
        
        predictor = DentalDiseasePredictor(args.model)
        
        # Get all images in folder
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_paths = [os.path.join(args.folder, f) for f in os.listdir(args.folder) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_paths:
            print("❌ No images found in the specified folder")
            return
        
        results = predictor.predict_batch(image_paths)
        
        print(f"\n📊 Batch Prediction Results ({len(results)} images):")
        for result in results:
            print(f"   {os.path.basename(result['image_path'])}: {result['predicted_class']} "
                  f"(conf: {result['confidence']:.4f})")
        
        # Save batch results
        with open('results/batch_predictions.json', 'w') as f:
            json.dump([{
                'image_path': r['image_path'],
                'predicted_class': r['predicted_class'],
                'confidence': r['confidence']
            } for r in results], f, indent=2)
        
        print("✅ Batch results saved to 'results/batch_predictions.json'")

if __name__ == '__main__':
    # Create required directories
    setup_directories()
    
    # Check if dependencies are installed
    try:
        import torch
        import torchvision
        import matplotlib
        import seaborn
        import sklearn
        print("✅ All dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install torch torchvision matplotlib seaborn scikit-learn tqdm pillow")
        exit(1)
    
    # Run the main function
    main()