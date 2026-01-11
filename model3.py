#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
MODEL 3 IMPROVED: EfficientNet-B4 U-Net + BerHu Loss + Percentile Normalization
=============================================================================
IMPROVEMENTS:
- Automatic mask polarity detection and correction
- Mask binarization
- Comprehensive epoch logging
- Training progress dashboard visualization
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


def get_device():
    if torch.backends.mps.is_available():
        print("🍎 Using Apple MPS")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("🎮 Using CUDA")
        return torch.device("cuda")
    else:
        print("💻 Using CPU")
        return torch.device("cpu")


# =============================================================================
# DATASET WITH MASK POLARITY CHECK
# =============================================================================
class HeightDatasetPercentile(Dataset):
    """Dataset with Percentile-based Normalization and Mask Polarity Check"""
    
    def __init__(self, data_list, height_stats, transform=None, check_mask_polarity=False):
        self.data_list = data_list
        self.transform = transform
        self.height_max = height_stats['max']
        self.height_min = height_stats['min']
        self.height_mean = height_stats['mean']
        self.height_std = height_stats['std']
        self.mask_inverted = False
        self.check_mask_polarity = check_mask_polarity
        
        self.p1 = max(0, self.height_mean - 2 * self.height_std)
        self.p99 = min(self.height_max, self.height_mean + 2 * self.height_std)
        if self.p99 <= self.p1:
            self.p1, self.p99 = 0, self.height_max
        self.range = self.p99 - self.p1 + 1e-8
        
        print(f"📊 Dataset: {len(self.data_list)} samples, p1={self.p1:.2f}m, p99={self.p99:.2f}m")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        image = cv2.cvtColor(cv2.imread(sample['image']), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        height_map = np.load(sample['height'])
        
        # Apply mask inversion if detected
        if self.mask_inverted:
            mask = 1.0 - mask
        
        # Binarize mask
        mask = (mask > 0.5).astype(np.float32)
        
        height_map = np.clip((height_map - self.p1) / self.range, 0, 1)
        
        if self.transform:
            transformed = self.transform(image=image, masks=[mask, height_map])
            image = transformed['image']
            mask, height_map = transformed['masks']
        
        mask = torch.from_numpy(mask).unsqueeze(0) if isinstance(mask, np.ndarray) else mask.unsqueeze(0)
        height_map = torch.from_numpy(height_map).unsqueeze(0) if isinstance(height_map, np.ndarray) else height_map.unsqueeze(0)
        return image, height_map, mask


def check_and_fix_mask_polarity(dataset, num_samples=100):
    """
    Check mask polarity on first batch and invert if necessary.
    
    Args:
        dataset: Dataset to check
        num_samples: Number of samples to check
        
    Returns:
        bool: True if mask was inverted, False otherwise
    """
    building_ratios = []
    check_size = min(num_samples, len(dataset))
    
    print(f"\n🔍 Checking mask polarity on {check_size} samples...")
    
    for idx in range(check_size):
        sample = dataset.data_list[idx]
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        building_ratio = np.mean(mask > 0.5)
        building_ratios.append(building_ratio)
    
    avg_building_ratio = np.mean(building_ratios)
    print(f"   Average building ratio: {avg_building_ratio:.4f}")
    
    # Check if mask is inverted
    if avg_building_ratio > 0.60:
        print(f"⚠️  WARNING: Building ratio is abnormally high ({avg_building_ratio:.2%})")
        print(f"   Mask appears to be INVERTED (background=1, building=0)")
        print(f"   Automatically inverting all masks: mask = 1 - mask")
        dataset.mask_inverted = True
        return True
    elif avg_building_ratio < 0.01:
        print(f"⚠️  WARNING: Building ratio is abnormally low ({avg_building_ratio:.2%})")
        print(f"   Mask appears to be INVERTED (background=1, building=0)")
        print(f"   Automatically inverting all masks: mask = 1 - mask")
        dataset.mask_inverted = True
        return True
    else:
        print(f"✅ Mask polarity appears correct (building ratio: {avg_building_ratio:.2%})")
        return False


def get_training_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.OneOf([A.HueSaturationValue(10, 15, 10), A.RandomBrightnessContrast(0.2, 0.2)], p=0.5),
        A.OneOf([A.Blur(blur_limit=3), A.GaussNoise(var_limit=(10.0, 50.0))], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_validation_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# =============================================================================
# DUAL-HEAD MODEL
# =============================================================================
class DualHeadEfficientNet(nn.Module):
    """
    Dual-Head Model: Height Regression + Building Segmentation
    
    Final output = sigmoid(Height) * sigmoid(Segmentation)
    
    Bu sayede:
    1. Bina olmayan alanlar otomatik 0 oluyor
    2. Model bina sınırlarını öğreniyor
    3. False positive azalıyor
    """
    
    def __init__(self, encoder_name="efficientnet-b4", encoder_weights="imagenet"):
        super().__init__()
        
        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=2,  # 2 channels: height + segmentation
            activation=None
        )
    
    def forward(self, x):
        out = self.encoder(x)
        
        # Split outputs
        height_raw = out[:, 0:1, :, :]   # Height regression (raw logits)
        seg_logits = out[:, 1:2, :, :]   # Segmentation logits
        
        # Apply sigmoid to both
        height_sigmoid = torch.sigmoid(height_raw)  # [0, 1] range
        seg_prob = torch.sigmoid(seg_logits)        # Building probability
        
        # Final masked height
        height_masked = height_sigmoid * seg_prob
        
        return height_masked, height_sigmoid, seg_prob, seg_logits


# =============================================================================
# LOSSES
# =============================================================================
class BerHuLoss(nn.Module):
    """BerHu (Reverse Huber) Loss: L1 for small errors, L2 for large errors"""
    
    def __init__(self, threshold_ratio=0.2):
        super().__init__()
        self.threshold_ratio = threshold_ratio
    
    def forward(self, pred, target, mask):
        diff = torch.abs(pred - target) * mask
        n = mask.sum()
        if n < 1.0:
            return torch.zeros(1, device=pred.device, requires_grad=True).mean()
        
        c = torch.clamp(self.threshold_ratio * diff.max().detach(), min=1e-4)
        l1_mask = (diff <= c).float()
        l1_loss = diff * l1_mask
        l2_loss = ((diff ** 2 + c ** 2) / (2 * c)) * (1 - l1_mask)
        
        return ((l1_loss + l2_loss) * mask).sum() / (n + 1e-8)


class CombinedLossWithBGSuppression(nn.Module):
    """
    Combined Loss: BerHu + Smoothness + Segmentation + Background Suppression
    
    Adaptive Error Handling:
    - BerHu: Outlier'lara robust
    - Smoothness: Edge-aware smoothing
    - Segmentation: Bina sınırlarını öğrenme
    - Background Suppression: Bina olmayan alanlarda 0 tahmin
    """
    
    def __init__(self, berhu_threshold=0.2, smoothness_weight=0.1, 
                 seg_weight=0.5, bg_weight=0.3):
        super().__init__()
        self.berhu = BerHuLoss(berhu_threshold)
        self.smoothness_weight = smoothness_weight
        self.seg_weight = seg_weight
        self.bg_weight = bg_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, height_masked, height_sigmoid, seg_prob, seg_logits, target, mask):
        """
        Args:
            height_masked: Final masked height prediction
            height_sigmoid: Raw height prediction (before masking)
            seg_prob: Segmentation probability (after sigmoid)
            seg_logits: Segmentation logits (before sigmoid)
            target: Ground truth height
            mask: Ground truth building mask
        """
        
        # 1. BerHu Loss (on building regions)
        berhu_loss = self.berhu(height_masked, target, mask)
        
        # 2. Smoothness Loss (edge-aware)
        grad_x = torch.abs(height_masked[:, :, :-1, :] - height_masked[:, :, 1:, :])
        grad_y = torch.abs(height_masked[:, :, :, :-1] - height_masked[:, :, :, 1:])
        
        # Edge weights from target (avoid smoothing at real edges)
        target_grad_x = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        target_grad_y = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        edge_weight_x = torch.exp(-10 * target_grad_x)
        edge_weight_y = torch.exp(-10 * target_grad_y)
        
        smoothness_loss = (grad_x * edge_weight_x).mean() + (grad_y * edge_weight_y).mean()
        
        # 3. Segmentation Loss (BCE)
        seg_loss = (self.bce(seg_logits, mask) * mask).sum() / (mask.sum() + 1e-8)
        
        # 4. Background Suppression Loss (penalize non-zero predictions on background)
        bg_mask = 1.0 - mask
        bg_loss = (height_masked * bg_mask).abs().mean()
        
        # Combined loss
        total_loss = (berhu_loss + 
                     self.smoothness_weight * smoothness_loss + 
                     self.seg_weight * seg_loss + 
                     self.bg_weight * bg_loss)
        
        loss_components = {
            'berhu': berhu_loss.item(),
            'smooth': smoothness_loss.item(),
            'seg': seg_loss.item(),
            'bg': bg_loss.item(),
            'building': berhu_loss.item(),  # For logging
            'background': bg_loss.item()    # For logging
        }
        
        return total_loss, loss_components


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(pred, target, mask, p1, range_val):
    """Compute evaluation metrics"""
    with torch.no_grad():
        # Convert to meters
        pred_m = pred * range_val + p1
        target_m = target * range_val + p1
        
        # Only compute on building regions
        building_pixels = mask > 0.5
        n = building_pixels.sum()
        
        if n < 1:
            return {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'delta1': 0.0, 'delta2': 0.0, 'delta3': 0.0}
        
        pred_flat = pred_m[building_pixels]
        target_flat = target_m[building_pixels]
        
        # Metrics
        mae = torch.abs(pred_flat - target_flat).mean().item()
        rmse = torch.sqrt(((pred_flat - target_flat) ** 2).mean()).item()
        
        # R²
        ss_res = ((pred_flat - target_flat) ** 2).sum()
        ss_tot = ((target_flat - target_flat.mean()) ** 2).sum()
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
        
        # Delta thresholds
        thresh = torch.maximum(pred_flat / (target_flat + 1e-8), target_flat / (pred_flat + 1e-8))
        delta1 = (thresh < 1.25).float().mean().item() * 100
        delta2 = (thresh < 1.25 ** 2).float().mean().item() * 100
        delta3 = (thresh < 1.25 ** 3).float().mean().item() * 100
        
        return {
            'mae': mae, 
            'rmse': rmse, 
            'r2': r2, 
            'delta1': delta1, 
            'delta2': delta2, 
            'delta3': delta3
        }


# =============================================================================
# TRAINING & VALIDATION
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device, epoch, p1, range_val):
    model.train()
    total_loss = 0
    all_preds, all_targets, all_masks = [], [], []
    loss_components_sum = {'berhu': 0, 'smooth': 0, 'seg': 0, 'bg': 0, 'building': 0, 'background': 0}
    
    pbar = tqdm(loader, desc=f'Train Epoch {epoch}')
    for images, target_heights, masks in pbar:
        images = images.to(device)
        target_heights = target_heights.to(device)
        masks = masks.to(device)
        
        # Forward
        height_masked, height_sigmoid, seg_prob, seg_logits = model(images)
        loss, loss_comp = criterion(height_masked, height_sigmoid, seg_prob, seg_logits, target_heights, masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in loss_comp.items():
            loss_components_sum[k] += v
        
        all_preds.append(height_masked.detach())
        all_targets.append(target_heights.detach())
        all_masks.append(masks.detach())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    metrics = compute_metrics(all_preds, all_targets, all_masks, p1, range_val)
    
    avg_loss = total_loss / len(loader)
    avg_loss_comp = {k: v / len(loader) for k, v in loss_components_sum.items()}
    
    return avg_loss, metrics, avg_loss_comp


def validate_epoch(model, loader, criterion, device, epoch, p1, range_val):
    model.eval()
    total_loss = 0
    all_preds, all_targets, all_masks = [], [], []
    loss_components_sum = {'berhu': 0, 'smooth': 0, 'seg': 0, 'bg': 0, 'building': 0, 'background': 0}
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Val Epoch {epoch}')
        for images, target_heights, masks in pbar:
            images = images.to(device)
            target_heights = target_heights.to(device)
            masks = masks.to(device)
            
            height_masked, height_sigmoid, seg_prob, seg_logits = model(images)
            loss, loss_comp = criterion(height_masked, height_sigmoid, seg_prob, seg_logits, target_heights, masks)
            
            total_loss += loss.item()
            for k, v in loss_comp.items():
                loss_components_sum[k] += v
            
            all_preds.append(height_masked)
            all_targets.append(target_heights)
            all_masks.append(masks)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    metrics = compute_metrics(all_preds, all_targets, all_masks, p1, range_val)
    
    avg_loss = total_loss / len(loader)
    avg_loss_comp = {k: v / len(loader) for k, v in loss_components_sum.items()}
    
    return avg_loss, metrics, avg_loss_comp


# =============================================================================
# TRAINING PROGRESS DASHBOARD
# =============================================================================
class TrainingLogger:
    """Logger to track training progress and generate dashboards"""
    
    def __init__(self):
        self.history = {
            'epoch': [],
            'train_total_loss': [],
            'val_total_loss': [],
            'train_building_loss': [],
            'train_background_loss': [],
            'val_mae_meters': [],
            'val_rmse_meters': [],
            'val_r2': [],
            'learning_rate': []
        }
        self.best_epoch = {'mae': 0, 'rmse': 0, 'r2': 0}
        self.best_values = {'mae': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')}
    
    def log_epoch(self, epoch, train_loss, val_loss, train_loss_comp, val_metrics, lr):
        """Log metrics for one epoch"""
        self.history['epoch'].append(epoch)
        self.history['train_total_loss'].append(train_loss)
        self.history['val_total_loss'].append(val_loss)
        self.history['train_building_loss'].append(train_loss_comp['building'])
        self.history['train_background_loss'].append(train_loss_comp['background'])
        self.history['val_mae_meters'].append(val_metrics['mae'])
        self.history['val_rmse_meters'].append(val_metrics['rmse'])
        self.history['val_r2'].append(val_metrics['r2'])
        self.history['learning_rate'].append(lr)
        
        # Update best epochs
        if val_metrics['mae'] < self.best_values['mae']:
            self.best_values['mae'] = val_metrics['mae']
            self.best_epoch['mae'] = epoch
        if val_metrics['rmse'] < self.best_values['rmse']:
            self.best_values['rmse'] = val_metrics['rmse']
            self.best_epoch['rmse'] = epoch
        if val_metrics['r2'] > self.best_values['r2']:
            self.best_values['r2'] = val_metrics['r2']
            self.best_epoch['r2'] = epoch
    
    def print_epoch_summary(self, epoch):
        """Print epoch summary"""
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*70}")
        print(f"📊 Total Loss:")
        print(f"   Train: {self.history['train_total_loss'][-1]:.4f}")
        print(f"   Val:   {self.history['val_total_loss'][-1]:.4f}")
        print(f"\n📊 Training Loss Components:")
        print(f"   Building:   {self.history['train_building_loss'][-1]:.4f}")
        print(f"   Background: {self.history['train_background_loss'][-1]:.4f}")
        print(f"\n📊 Validation Metrics:")
        print(f"   MAE:  {self.history['val_mae_meters'][-1]:.2f}m")
        print(f"   RMSE: {self.history['val_rmse_meters'][-1]:.2f}m")
        print(f"   R²:   {self.history['val_r2'][-1]:.4f}")
        print(f"\n📊 Learning Rate: {self.history['learning_rate'][-1]:.6f}")
        print(f"{'='*70}")
    
    def plot_training_progress(self, save_dir, epoch):
        """Generate 2x3 training progress dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        epochs = self.history['epoch']
        
        # 1. Total Loss (Train vs Val)
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_total_loss'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, self.history['val_total_loss'], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss (Train vs Val)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Training Loss Components (Building vs Background)
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_building_loss'], 'g-', label='Building', linewidth=2)
        ax.plot(epochs, self.history['train_background_loss'], 'orange', label='Background', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Validation MAE
        ax = axes[0, 2]
        ax.plot(epochs, self.history['val_mae_meters'], 'b-', linewidth=2)
        best_mae_epoch = self.best_epoch['mae']
        best_mae_value = self.best_values['mae']
        if best_mae_epoch in epochs:
            idx = epochs.index(best_mae_epoch)
            ax.plot(best_mae_epoch, self.history['val_mae_meters'][idx], 'go', markersize=10, label=f'Best: {best_mae_value:.2f}m')
            ax.axhline(y=best_mae_value, color='g', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE (meters)')
        ax.set_title('Validation MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Validation RMSE
        ax = axes[1, 0]
        ax.plot(epochs, self.history['val_rmse_meters'], 'r-', linewidth=2)
        best_rmse_epoch = self.best_epoch['rmse']
        best_rmse_value = self.best_values['rmse']
        if best_rmse_epoch in epochs:
            idx = epochs.index(best_rmse_epoch)
            ax.plot(best_rmse_epoch, self.history['val_rmse_meters'][idx], 'go', markersize=10, label=f'Best: {best_rmse_value:.2f}m')
            ax.axhline(y=best_rmse_value, color='g', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE (meters)')
        ax.set_title('Validation RMSE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Validation R²
        ax = axes[1, 1]
        ax.plot(epochs, self.history['val_r2'], 'purple', linewidth=2)
        best_r2_epoch = self.best_epoch['r2']
        best_r2_value = self.best_values['r2']
        if best_r2_epoch in epochs:
            idx = epochs.index(best_r2_epoch)
            ax.plot(best_r2_epoch, self.history['val_r2'][idx], 'go', markersize=10, label=f'Best: {best_r2_value:.4f}')
            ax.axhline(y=best_r2_value, color='g', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('R²')
        ax.set_title('Validation R²')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Learning Rate Schedule
        ax = axes[1, 2]
        ax.plot(epochs, self.history['learning_rate'], 'orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save with epoch number and overwrite latest
        save_path_epoch = save_dir / f'training_progress_epoch_{epoch:03d}.png'
        save_path_latest = save_dir / 'training_progress_latest.png'
        
        plt.savefig(save_path_epoch, dpi=150, bbox_inches='tight')
        plt.savefig(save_path_latest, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training progress dashboard saved:")
        print(f"   {save_path_epoch}")
        print(f"   {save_path_latest}")


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_predictions(model, loader, device, p1, range_val, save_path, num_samples=3):
    """Visualize predictions vs ground truth"""
    model.eval()
    
    images, target_heights, masks = next(iter(loader))
    num_samples = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        images = images.to(device)
        height_masked, height_sigmoid, seg_prob, seg_logits = model(images)
        
        for i in range(num_samples):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
            mask = masks[i, 0].numpy()
            target_h = np.clip(target_heights[i, 0].numpy() * range_val + p1, 0, None)
            pred_h = np.clip(height_masked[i, 0].cpu().numpy() * range_val + p1, 0, None)
            seg_pred = seg_prob[i, 0].cpu().numpy()
            
            v_max = max(target_h.max(), pred_h.max(), 1)
            error_map = np.abs(target_h - pred_h) * mask
            
            # Plot
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')
            
            im1 = axes[i, 1].imshow(target_h, cmap='viridis', vmin=0, vmax=v_max)
            axes[i, 1].set_title(f'GT Max: {target_h[mask>0].max():.1f}m' if mask.sum()>0 else 'GT')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
            
            im2 = axes[i, 2].imshow(pred_h, cmap='viridis', vmin=0, vmax=v_max)
            axes[i, 2].set_title(f'Pred Max: {pred_h[mask>0].max():.1f}m' if mask.sum()>0 else 'Pred')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
            
            im3 = axes[i, 3].imshow(error_map, cmap='hot', vmin=0, vmax=10)
            axes[i, 3].set_title(f'Error MAE: {error_map[mask>0].mean():.2f}m' if mask.sum()>0 else 'Error')
            axes[i, 3].axis('off')
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)
            
            # Segmentation prediction
            im4 = axes[i, 4].imshow(seg_pred, cmap='gray', vmin=0, vmax=1)
            axes[i, 4].set_title('Seg Prediction')
            axes[i, 4].axis('off')
            plt.colorbar(im4, ax=axes[i, 4], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Visualization saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Model 3 Improved: EfficientNet-B4 + BerHu + Percentile')
    parser.add_argument('--data_dir', default='dfc2023_height_dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_dir', default='./checkpoints_model3_improved')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', default='height-estimation-model3-improved')
    parser.add_argument('--berhu_threshold', type=float, default=0.2)
    parser.add_argument('--smoothness_weight', type=float, default=0.1)
    parser.add_argument('--seg_weight', type=float, default=0.5)
    parser.add_argument('--bg_weight', type=float, default=0.3)
    args = parser.parse_args()
    
    print("=" * 70)
    print("MODEL 3 IMPROVED: EfficientNet-B4 + BerHu + Percentile + BG Suppression")
    print("=" * 70)
    print(f"\n📋 Improvements:")
    print(f"   ✓ Automatic mask polarity detection and correction")
    print(f"   ✓ Mask binarization")
    print(f"   ✓ Comprehensive epoch logging")
    print(f"   ✓ Training progress dashboard visualization")
    
    if args.use_wandb:
        wandb.init(project=args.project_name, config=args, name="efficientnet_berhu_improved")
    
    device = get_device()
    data_dir = Path(args.data_dir)
    
    with open(data_dir / 'height_statistics.json', 'r') as f:
        height_stats = json.load(f)
    
    p1 = max(0, height_stats['mean'] - 2 * height_stats['std'])
    p99 = min(height_stats['max'], height_stats['mean'] + 2 * height_stats['std'])
    if p99 <= p1:
        p1, p99 = 0, height_stats['max']
    range_val = p99 - p1 + 1e-8
    
    print(f"\n📊 Height Stats: min={height_stats['min']:.2f}m, max={height_stats['max']:.2f}m")
    print(f"   Percentile: p1={p1:.2f}m, p99={p99:.2f}m")
    
    with open(data_dir / 'train_data.json', 'r') as f:
        train_data = json.load(f)
    with open(data_dir / 'val_data.json', 'r') as f:
        val_data = json.load(f)
    
    train_dataset = HeightDatasetPercentile(train_data, height_stats, get_training_transforms())
    val_dataset = HeightDatasetPercentile(val_data, height_stats, get_validation_transforms())
    
    # Check and fix mask polarity on training set
    mask_was_inverted = check_and_fix_mask_polarity(train_dataset, num_samples=100)
    if mask_was_inverted:
        # Apply same inversion to validation set
        val_dataset.mask_inverted = True
        print(f"   Applying mask inversion to validation set as well")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Dual-Head Model
    model = DualHeadEfficientNet(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet"
    )
    model.to(device)
    
    print(f"\n🏗️  Model: Dual-Head U-Net with EfficientNet-B4")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Output: Height (masked) + Segmentation")
    
    criterion = CombinedLossWithBGSuppression(
        berhu_threshold=args.berhu_threshold,
        smoothness_weight=args.smoothness_weight,
        seg_weight=args.seg_weight,
        bg_weight=args.bg_weight
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    best_mae, patience_counter = float('inf'), 0
    
    # Initialize training logger
    logger = TrainingLogger()
    
    print(f"\n🚀 Starting training...")
    
    for epoch in range(args.epochs):
        current_epoch = epoch + 1
        print(f'\n{"="*70}\nEpoch {current_epoch}/{args.epochs}\n{"="*70}')
        
        # Train and validate
        train_loss, train_m, train_loss_comp = train_epoch(
            model, train_loader, criterion, optimizer, device, current_epoch, p1, range_val
        )
        val_loss, val_m, val_loss_comp = validate_epoch(
            model, val_loader, criterion, device, current_epoch, p1, range_val
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch data
        logger.log_epoch(current_epoch, train_loss, val_loss, train_loss_comp, val_m, current_lr)
        
        # Print epoch summary
        logger.print_epoch_summary(current_epoch)
        
        # Generate training progress dashboard
        logger.plot_training_progress(save_dir, current_epoch)
        
        # Step scheduler
        scheduler.step()
        
        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log({
                'epoch': current_epoch,
                'train_total_loss': train_loss,
                'val_total_loss': val_loss,
                'train_building_loss': train_loss_comp['building'],
                'train_background_loss': train_loss_comp['background'],
                'val_mae_meters': val_m['mae'],
                'val_rmse_meters': val_m['rmse'],
                'val_r2': val_m['r2'],
                'learning_rate': current_lr
            })
        
        # Save best model
        if val_m['mae'] < best_mae and val_m['mae'] > 0:
            best_mae, patience_counter = val_m['mae'], 0
            torch.save({
                'epoch': current_epoch, 
                'model_state_dict': model.state_dict(),
                'best_mae': best_mae, 
                'val_metrics': val_m, 
                'height_stats': height_stats,
                'percentile_params': {'p1': p1, 'p99': p99, 'range': range_val},
                'mask_was_inverted': mask_was_inverted,
                'config': {
                    'model': 'DualHeadUnet', 
                    'encoder': 'efficientnet-b4', 
                    'loss': 'BerHu+Smooth+Seg+BG', 
                    'normalization': 'Percentile',
                    'seg_weight': args.seg_weight,
                    'bg_weight': args.bg_weight
                }
            }, save_dir / 'best_model.pth')
            print(f'\n✅ Best model saved! MAE: {best_mae:.2f}m')
            visualize_predictions(model, val_loader, device, p1, range_val, 
                                 save_dir / f'pred_epoch_{current_epoch:03d}.png')
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f'\n⏹️  Early stopping')
            break
    
    print(f'\n{"="*70}\n🎉 Training completed! Best MAE: {best_mae:.2f}m\n{"="*70}')
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()