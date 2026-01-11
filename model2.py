#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
MODEL 2 FULLY FIXED: All Issues Resolved
=============================================================================
FIXES:
1. Decoder call fixed (features as list)
2. Sobel filters moved to correct device (CUDA compatibility)
3. GaussNoise parameter fixed
=============================================================================
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
import json
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy import ndimage


# =============================================================================
# DEVICE SETUP
# =============================================================================
def get_device():
    """Get the best available device for training"""
    if torch.backends.mps.is_available():
        print("🍎 Using Apple Metal Performance Shaders (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("🎮 Using CUDA")
        return torch.device("cuda")
    else:
        print("💻 Using CPU")
        return torch.device("cpu")


# =============================================================================
# DATASET WITH Z-SCORE NORMALIZATION
# =============================================================================
class HeightDatasetZScore(Dataset):
    """Dataset with Z-Score (Standard) Normalization"""
    
    def __init__(self, data_list, height_stats, transform=None):
        self.data_list = data_list
        self.transform = transform
        
        # Height statistics
        self.height_max = height_stats['max']
        self.height_min = height_stats['min']
        self.height_mean = height_stats['mean']
        self.height_std = height_stats['std'] + 1e-8
        
        print(f"📊 Dataset: {len(self.data_list)} samples")
        print(f"   Height range: {self.height_min:.2f}m - {self.height_max:.2f}m")
        print(f"   Z-Score params: mean={self.height_mean:.2f}, std={self.height_std:.2f}")
    
    def normalize_height(self, height):
        """Z-Score normalization"""
        return (height - self.height_mean) / self.height_std
    
    def denormalize_height(self, height_norm):
        """Inverse Z-Score"""
        return height_norm * self.height_std + self.height_mean
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        
        # Load image
        image = cv2.imread(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load building mask
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0
        
        # Load height map
        height_map = np.load(sample['height'])
        
        # Apply Z-Score normalization
        height_map = self.normalize_height(height_map)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=image,
                masks=[mask, height_map]
            )
            image = transformed['image']
            mask = transformed['masks'][0]
            height_map = transformed['masks'][1]
        
        # Convert to tensors
        mask = torch.from_numpy(mask).unsqueeze(0) if isinstance(mask, np.ndarray) else mask.unsqueeze(0)
        height_map = torch.from_numpy(height_map).unsqueeze(0) if isinstance(height_map, np.ndarray) else height_map.unsqueeze(0)
        
        return image, height_map, mask


# =============================================================================
# AUGMENTATIONS
# =============================================================================
def get_training_transforms():
    """Training augmentations"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        ], p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3),
            A.GaussNoise(),  # Fixed: use default parameters
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_validation_transforms():
    """Validation transforms"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# =============================================================================
# ATTENTION MODULE FOR DETAIL PRESERVATION
# =============================================================================
class DetailAttention(nn.Module):
    """Spatial attention mechanism to preserve fine details"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.conv(x)
        return x * attention


# =============================================================================
# IMPROVED DUAL-HEAD MODEL - FULLY FIXED
# =============================================================================
class ImprovedDualHeadModel(nn.Module):
    """
    Improved Dual-Head Model - ALL FIXES APPLIED
    """
    
    def __init__(self, encoder_name="tu-hrnet_w18", encoder_weights="imagenet"):
        super().__init__()
        
        # Base U-Net model
        base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=16,
            activation=None
        )
        
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.segmentation_head = base_model.segmentation_head
        
        # Height head with detail attention
        self.height_head = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            DetailAttention(8),
            nn.Conv2d(8, 1, 1)
        )
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1)
        )
    
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Decode - FIXED: Pass features as list
        decoder_output = self.decoder(features)
        base_features = self.segmentation_head(decoder_output)
        
        # Get height and segmentation predictions
        height_raw = self.height_head(base_features)
        seg_logits = self.seg_head(base_features)
        seg_prob = torch.sigmoid(seg_logits)
        
        # Soft masking
        seg_weight = seg_prob * 0.7 + 0.3
        height_masked = height_raw * seg_weight
        
        # Feature fusion
        fused_features = torch.cat([height_raw, height_masked], dim=1)
        height_fused = self.fusion(fused_features)
        
        return height_fused, height_raw, seg_prob, seg_logits


# =============================================================================
# ENHANCED LOSS FUNCTION - FULLY FIXED
# =============================================================================
class EnhancedLoss(nn.Module):
    """Enhanced loss with device-aware Sobel filters"""
    
    def __init__(self, l1_weight=1.0, grad_weight=0.8, seg_weight=0.5, 
                 bg_weight=0.3, percep_weight=0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight
        self.seg_weight = seg_weight
        self.bg_weight = bg_weight
        self.percep_weight = percep_weight
        
        # FIXED: Register buffers (will automatically move to correct device)
        self.register_buffer('sobel_x', 
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer('sobel_y',
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
    
    def compute_gradient(self, x):
        """Compute image gradients"""
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    
    def multi_scale_gradient_loss(self, pred, target, mask):
        """Gradient loss at multiple scales"""
        scales = [1.0, 0.5, 0.25]
        total_loss = 0
        
        for scale in scales:
            if scale < 1.0:
                h, w = int(pred.shape[2] * scale), int(pred.shape[3] * scale)
                pred_scaled = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, size=(h, w), mode='bilinear', align_corners=False)
                mask_scaled = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
            else:
                pred_scaled, target_scaled, mask_scaled = pred, target, mask
            
            pred_grad = self.compute_gradient(pred_scaled)
            target_grad = self.compute_gradient(target_scaled)
            
            grad_diff = torch.abs(pred_grad - target_grad) * mask_scaled
            total_loss += grad_diff.mean() * scale
        
        return total_loss / len(scales)
    
    def forward(self, pred_fused, pred_raw, seg_prob, seg_logits, target, mask):
        # L1 loss
        l1_fused = F.l1_loss(pred_fused * mask, target * mask)
        l1_raw = F.l1_loss(pred_raw * mask, target * mask)
        l1_loss = (l1_fused + l1_raw) / 2
        
        # Multi-scale gradient loss
        grad_loss = self.multi_scale_gradient_loss(pred_fused, target, mask)
        
        # Segmentation loss
        seg_loss = F.binary_cross_entropy(seg_prob, mask)
        
        # Background penalty
        bg_penalty = torch.abs(pred_fused * (1 - mask)).mean()
        
        # Perceptual loss
        percep_loss = F.mse_loss(pred_fused * mask, target * mask)
        
        # Total loss
        total_loss = (
            self.l1_weight * l1_loss +
            self.grad_weight * grad_loss +
            self.seg_weight * seg_loss +
            self.bg_weight * bg_penalty +
            self.percep_weight * percep_loss
        )
        
        return total_loss, {
            'l1': l1_loss.item(),
            'grad': grad_loss.item(),
            'seg': seg_loss.item(),
            'bg': bg_penalty.item(),
            'percep': percep_loss.item()
        }


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(pred, target, mask):
    """Compute evaluation metrics"""
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    
    pred_masked = pred[mask > 0.5]
    target_masked = target[mask > 0.5]
    
    if len(pred_masked) == 0:
        return {'mae': 0, 'rmse': 0, 'r2': 0}
    
    mae = np.abs(pred_masked - target_masked).mean()
    rmse = np.sqrt(((pred_masked - target_masked) ** 2).mean())
    
    try:
        r2 = r2_score(target_masked, pred_masked)
    except:
        r2 = 0
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# =============================================================================
# TRAINING & VALIDATION
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device, epoch, height_mean, height_std):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds, all_targets, all_masks = [], [], []
    loss_components_sum = {'l1': 0, 'grad': 0, 'seg': 0, 'bg': 0, 'percep': 0}
    
    pbar = tqdm(loader, desc=f'Train Epoch {epoch}')
    for images, heights, masks in pbar:
        images = images.to(device)
        heights = heights.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        pred_fused, pred_raw, seg_prob, seg_logits = model(images)
        loss, loss_components = criterion(pred_fused, pred_raw, seg_prob, seg_logits, heights, masks)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in loss_components.items():
            loss_components_sum[k] += v
        
        pred_denorm = pred_fused * height_std + height_mean
        target_denorm = heights * height_std + height_mean
        
        all_preds.append(pred_denorm.detach())
        all_targets.append(target_denorm.detach())
        all_masks.append(masks.detach())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    metrics = compute_metrics(all_preds, all_targets, all_masks)
    
    avg_loss = total_loss / len(loader)
    avg_loss_components = {k: v / len(loader) for k, v in loss_components_sum.items()}
    
    return avg_loss, metrics, avg_loss_components


def validate_epoch(model, loader, criterion, device, epoch, height_mean, height_std):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_preds, all_targets, all_masks = [], [], []
    loss_components_sum = {'l1': 0, 'grad': 0, 'seg': 0, 'bg': 0, 'percep': 0}
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Val Epoch {epoch}')
        for images, heights, masks in pbar:
            images = images.to(device)
            heights = heights.to(device)
            masks = masks.to(device)
            
            pred_fused, pred_raw, seg_prob, seg_logits = model(images)
            loss, loss_components = criterion(pred_fused, pred_raw, seg_prob, seg_logits, heights, masks)
            
            total_loss += loss.item()
            for k, v in loss_components.items():
                loss_components_sum[k] += v
            
            pred_denorm = pred_fused * height_std + height_mean
            target_denorm = heights * height_std + height_mean
            
            all_preds.append(pred_denorm)
            all_targets.append(target_denorm)
            all_masks.append(masks)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    metrics = compute_metrics(all_preds, all_targets, all_masks)
    
    avg_loss = total_loss / len(loader)
    avg_loss_components = {k: v / len(loader) for k, v in loss_components_sum.items()}
    
    return avg_loss, metrics, avg_loss_components


# =============================================================================
# METRICS TRACKER
# =============================================================================
class MetricsTracker:
    """Track and visualize training metrics"""
    
    def __init__(self):
        self.history = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'val_mae': [], 'val_rmse': [], 'val_r2': [], 'lr': [],
            'train_l1': [], 'train_grad': [], 'train_seg': [], 'train_bg': []
        }
    
    def update(self, epoch, train_loss, train_loss_components, val_loss, val_metrics, lr):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_mae'].append(val_metrics['mae'])
        self.history['val_rmse'].append(val_metrics['rmse'])
        self.history['val_r2'].append(val_metrics['r2'])
        self.history['lr'].append(lr)
        self.history['train_l1'].append(train_loss_components['l1'])
        self.history['train_grad'].append(train_loss_components['grad'])
        self.history['train_seg'].append(train_loss_components['seg'])
        self.history['train_bg'].append(train_loss_components['bg'])
    
    def plot_curves(self, save_path):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], label='Train', marker='o')
        axes[0, 0].plot(self.history['epoch'], self.history['val_loss'], label='Val', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.history['epoch'], self.history['val_mae'], marker='o', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (m)')
        axes[0, 1].set_title('Validation MAE')
        axes[0, 1].grid(True)
        
        axes[0, 2].plot(self.history['epoch'], self.history['val_rmse'], marker='s', color='orange')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('RMSE (m)')
        axes[0, 2].set_title('Validation RMSE')
        axes[0, 2].grid(True)
        
        axes[1, 0].plot(self.history['epoch'], self.history['val_r2'], marker='^', color='purple')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_title('Validation R²')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.history['epoch'], self.history['train_l1'], label='L1', marker='o')
        axes[1, 1].plot(self.history['epoch'], self.history['train_grad'], label='Grad', marker='s')
        axes[1, 1].plot(self.history['epoch'], self.history['train_seg'], label='Seg', marker='^')
        axes[1, 1].plot(self.history['epoch'], self.history['train_bg'], label='BG', marker='d')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Component')
        axes[1, 1].set_title('Loss Components')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        axes[1, 2].plot(self.history['epoch'], self.history['lr'], marker='o', color='red')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Training curves saved: {save_path}")
    
    def save_csv(self, save_path):
        import pandas as pd
        df = pd.DataFrame(self.history)
        df.to_csv(save_path, index=False)
        print(f"💾 Metrics saved: {save_path}")


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_predictions(model, loader, device, height_mean, height_std, save_path, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 4))
    
    with torch.no_grad():
        for i, (images, heights, masks) in enumerate(loader):
            if i >= num_samples:
                break
            
            images = images.to(device)
            heights = heights.to(device)
            masks = masks.to(device)
            
            pred_fused, _, _, _ = model(images)
            
            pred = pred_fused[0, 0].cpu().numpy() * height_std + height_mean
            target = heights[0, 0].cpu().numpy() * height_std + height_mean
            mask = masks[0, 0].cpu().numpy()
            
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(target * mask, cmap='jet', vmin=0, vmax=50)
            axes[i, 1].set_title(f'Ground Truth (Max: {target.max():.1f}m)')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred * mask, cmap='jet', vmin=0, vmax=50)
            axes[i, 2].set_title(f'Prediction (Max: {pred.max():.1f}m)')
            axes[i, 2].axis('off')
            
            error = np.abs(pred - target) * mask
            axes[i, 3].imshow(error, cmap='hot', vmin=0, vmax=10)
            axes[i, 3].set_title(f'Error (MAE: {error[mask>0.5].mean():.2f}m)')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Visualization saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Model 2 Fully Fixed')
    parser.add_argument('--data_dir', default='dfc2023_height_dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--save_dir', default='./checkpoints_model2_fixed')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', default='height-estimation-model2')
    parser.add_argument('--l1_weight', type=float, default=1.0)
    parser.add_argument('--grad_weight', type=float, default=0.8)
    parser.add_argument('--seg_weight', type=float, default=0.5)
    parser.add_argument('--bg_weight', type=float, default=0.3)
    parser.add_argument('--percep_weight', type=float, default=0.2)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MODEL 2 FULLY FIXED: All Issues Resolved")
    print("=" * 80)
    print("\n📋 Configuration:")
    print(f"   Architecture: Improved Dual-Head U-Net with HRNet-W18")
    print(f"   ALL FIXES APPLIED:")
    print(f"   ✅ Decoder call fixed (features as list)")
    print(f"   ✅ Sobel filters device-aware (CUDA compatible)")
    print(f"   ✅ GaussNoise fixed")
    print()
    
    if args.use_wandb:
        wandb.init(project=args.project_name, config=args, name="model2_fully_fixed")
    
    device = get_device()
    data_dir = Path(args.data_dir)
    
    with open(data_dir / 'height_statistics.json', 'r') as f:
        height_stats = json.load(f)
    
    height_mean = height_stats['mean']
    height_std = height_stats['std'] + 1e-8
    
    print(f"\n📊 Height Statistics:")
    print(f"   Min: {height_stats['min']:.2f}m")
    print(f"   Max: {height_stats['max']:.2f}m")
    print(f"   Mean: {height_mean:.2f}m")
    print(f"   Std: {height_std:.2f}m")
    
    with open(data_dir / 'train_data.json', 'r') as f:
        train_data = json.load(f)
    with open(data_dir / 'val_data.json', 'r') as f:
        val_data = json.load(f)
    
    print(f"\n📂 Dataset:")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val: {len(val_data)} samples")
    
    train_dataset = HeightDatasetZScore(train_data, height_stats, get_training_transforms())
    val_dataset = HeightDatasetZScore(val_data, height_stats, get_validation_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = ImprovedDualHeadModel(encoder_name="tu-hrnet_w18", encoder_weights="imagenet")
    model.to(device)
    
    print(f"\n🏗️  Model: Improved Dual-Head U-Net with HRNet-W18")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = EnhancedLoss(
        l1_weight=args.l1_weight, grad_weight=args.grad_weight,
        seg_weight=args.seg_weight, bg_weight=args.bg_weight,
        percep_weight=args.percep_weight
    )
    criterion.to(device)  # Move criterion to device
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    best_mae = float('inf')
    patience_counter = 0
    metrics_tracker = MetricsTracker()
    
    print(f"\n🚀 Starting training...")
    
    for epoch in range(args.epochs):
        print(f'\n{"=" * 70}')
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print(f'{"=" * 70}')
        
        train_loss, train_metrics, train_loss_components = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch + 1, height_mean, height_std
        )
        val_loss, val_metrics, val_loss_components = validate_epoch(
            model, val_loader, criterion, device, 
            epoch + 1, height_mean, height_std
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        metrics_tracker.update(
            epoch + 1, train_loss, train_loss_components, 
            val_loss, val_metrics, current_lr
        )
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                'train_mae': train_metrics['mae'], 'val_mae': val_metrics['mae'],
                'val_r2': val_metrics['r2'], 'lr': current_lr,
                **{f'train_{k}': v for k, v in train_loss_components.items()}
            })
        
        print(f'\n📈 Training: Loss={train_loss:.4f}, MAE={train_metrics["mae"]:.2f}m')
        print(f'📉 Validation: Loss={val_loss:.4f}, MAE={val_metrics["mae"]:.2f}m, R²={val_metrics["r2"]:.4f}')
        
        if val_metrics['mae'] < best_mae and val_metrics['mae'] > 0:
            best_mae = val_metrics['mae']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_mae': best_mae,
                'val_metrics': val_metrics,
                'height_stats': height_stats,
            }, save_dir / 'best_model.pth')
            
            print(f'\n✅ New best model! MAE: {best_mae:.2f}m')
            visualize_predictions(model, val_loader, device, height_mean, height_std, 
                                save_dir / f'pred_epoch_{epoch+1}.png')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            metrics_tracker.plot_curves(save_dir / 'training_curves.png')
            metrics_tracker.save_csv(save_dir / 'metrics_history.csv')
        
        if patience_counter >= args.patience:
            print(f'\n⏹️  Early stopping')
            break
    
    metrics_tracker.plot_curves(save_dir / 'training_curves_final.png')
    metrics_tracker.save_csv(save_dir / 'metrics_history_final.csv')
    
    print(f'\n{"=" * 70}')
    print(f'🎉 Training completed! Best MAE: {best_mae:.2f}m')
    print(f'{"=" * 70}')
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()