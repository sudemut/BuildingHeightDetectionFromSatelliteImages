#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
MODEL 5 IMPROVED: SegFormer + Ordinal Regression + Advanced Features
=============================================================================
İYİLEŞTİRMELER:
- Daha büyük encoder (MIT-B4)
- Daha fazla ordinal bin (100)
- Focal Loss ile zor örneklere odaklanma
- Edge-aware loss ile kenar detayları
- Daha iyi visualization
- Multi-worker data loading (num_workers=4)
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
# DATASET WITH ORDINAL LABELS
# =============================================================================
class HeightDatasetOrdinal(Dataset):
    """Dataset for Ordinal Regression"""
    
    def __init__(self, data_list, height_stats, transform=None, num_bins=100):
        self.data_list = data_list
        self.transform = transform
        self.num_bins = num_bins
        
        self.height_max = height_stats['max']
        self.height_min = height_stats['min']
        self.height_mean = height_stats['mean']
        self.height_std = height_stats['std']
        
        # Create bin thresholds
        self.bin_thresholds = np.linspace(0, self.height_max, num_bins + 1)[1:]
        self.bin_centers = (np.concatenate([[0], self.bin_thresholds[:-1]]) + self.bin_thresholds) / 2
        
        print(f"📊 Dataset: {len(self.data_list)} samples")
        print(f"   Height range: {self.height_min:.2f}m - {self.height_max:.2f}m")
        print(f"   Ordinal bins: {num_bins} (bin width: {self.height_max/num_bins:.2f}m)")
    
    def height_to_ordinal(self, height_map):
        """Convert continuous height to ordinal labels"""
        ordinal = np.zeros((self.num_bins, *height_map.shape), dtype=np.float32)
        for i, threshold in enumerate(self.bin_thresholds):
            ordinal[i] = (height_map > threshold).astype(np.float32)
        return ordinal
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        
        image = cv2.cvtColor(cv2.imread(sample['image']), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        height_map = np.load(sample['height'])
        height_map = np.clip(height_map, 0, self.height_max)
        
        # Convert to ordinal labels
        ordinal_labels = self.height_to_ordinal(height_map)
        
        if self.transform:
            transformed = self.transform(image=image, masks=[mask, height_map])
            image = transformed['image']
            mask, height_map = transformed['masks']
            
            if isinstance(height_map, np.ndarray):
                ordinal_labels = self.height_to_ordinal(height_map)
            else:
                ordinal_labels = self.height_to_ordinal(height_map.numpy())
        
        mask = torch.from_numpy(mask).unsqueeze(0) if isinstance(mask, np.ndarray) else mask.unsqueeze(0)
        height_map = torch.from_numpy(height_map).unsqueeze(0) if isinstance(height_map, np.ndarray) else height_map.unsqueeze(0)
        ordinal_labels = torch.from_numpy(ordinal_labels) if isinstance(ordinal_labels, np.ndarray) else ordinal_labels
        
        return image, height_map, mask, ordinal_labels


def get_training_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.HueSaturationValue(15, 20, 15, p=1),
            A.RandomBrightnessContrast(0.3, 0.3, p=1),
        ], p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3, p=1),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_validation_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# =============================================================================
# IMPROVED SEGFORMER MODEL
# =============================================================================
class SegFormerOrdinalImproved(nn.Module):
    """
    SegFormer with Improved Ordinal Regression Head
    - Larger encoder (MIT-B4)
    - Multi-scale feature fusion
    - Better ordinal head
    """
    
    def __init__(self, num_bins=100, encoder_name="mit_b4"):
        super().__init__()
        self.num_bins = num_bins
        
        # Larger encoder for better features
        self.segformer = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=256,  # More channels for richer features
            activation=None
        )
        
        # Improved ordinal regression head
        self.ordinal_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_bins, 1),
        )
    
    def forward(self, x):
        features = self.segformer(x)
        ordinal_logits = self.ordinal_head(features)
        return ordinal_logits
    
    def predict_height(self, x, max_height):
        """Convert ordinal logits to continuous height prediction"""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        bin_width = max_height / self.num_bins
        height = probs.sum(dim=1, keepdim=True) * bin_width
        return height, probs


# =============================================================================
# IMPROVED LOSS WITH FOCAL LOSS AND EDGE-AWARE
# =============================================================================
class FocalBCEWithLogitsLoss(nn.Module):
    """Focal Loss for handling hard examples"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ImprovedOrdinalLoss(nn.Module):
    """
    Improved Ordinal Loss with:
    - Focal loss for hard examples
    - Edge-aware component
    - Better background suppression
    """
    
    def __init__(self, consistency_weight=0.1, bg_weight=0.2, edge_weight=0.15, use_focal=True):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.bg_weight = bg_weight
        self.edge_weight = edge_weight
        
        if use_focal:
            self.bce = FocalBCEWithLogitsLoss(alpha=0.25, gamma=2.0, reduction='none')
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def compute_edges(self, mask):
        """Compute edge map using Sobel operator"""
        # Pad mask for convolution
        mask_pad = F.pad(mask, (1, 1, 1, 1), mode='replicate')
        
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(mask_pad, sobel_x)
        grad_y = F.conv2d(mask_pad, sobel_y)
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        edges = (edges > 0.1).float()
        
        return edges
    
    def forward(self, ordinal_logits, ordinal_targets, building_mask):
        # 1. Main ordinal loss (building pixels only)
        mask_expanded = building_mask.expand_as(ordinal_logits)
        loss = self.bce(ordinal_logits, ordinal_targets)
        masked_loss = loss * mask_expanded
        n_building = mask_expanded.sum() + 1e-8
        ordinal_loss = masked_loss.sum() / n_building
        
        # 2. Consistency loss: P(ti) >= P(ti+1)
        probs = torch.sigmoid(ordinal_logits)
        diff = probs[:, :-1, :, :] - probs[:, 1:, :, :]
        violations = F.relu(-diff)
        mask_for_diff = building_mask.expand(-1, probs.shape[1] - 1, -1, -1)
        consistency_loss = (violations * mask_for_diff).sum() / (mask_for_diff.sum() + 1e-8)
        
        # 3. Background Suppression Loss (softer version)
        bg_mask = 1.0 - building_mask
        bg_mask_expanded = bg_mask.expand_as(probs)
        
        if bg_mask.sum() > 0:
            # Suppress background predictions but not too aggressively
            bg_probs = probs * bg_mask_expanded
            bg_loss = bg_probs.sum() / (bg_mask_expanded.sum() + 1e-8)
        else:
            bg_loss = torch.zeros(1, device=ordinal_logits.device).mean()
        
        # 4. Edge-aware loss (focus on building boundaries)
        if self.edge_weight > 0:
            edges = self.compute_edges(building_mask)
            edge_mask = edges.expand_as(ordinal_logits)
            edge_loss = (loss * edge_mask).sum() / (edge_mask.sum() + 1e-8)
        else:
            edge_loss = torch.zeros(1, device=ordinal_logits.device).mean()
        
        total = (ordinal_loss + 
                self.consistency_weight * consistency_loss + 
                self.bg_weight * bg_loss +
                self.edge_weight * edge_loss)
        
        return total, {
            'ordinal': ordinal_loss.item(),
            'consistency': consistency_loss.item(),
            'bg': bg_loss.item(),
            'edge': edge_loss.item()
        }


# =============================================================================
# METRICS
# =============================================================================
def ordinal_to_height_tensor(ordinal_probs, max_height, num_bins):
    """Convert ordinal probabilities to height"""
    bin_width = max_height / num_bins
    height = ordinal_probs.sum(dim=1, keepdim=True) * bin_width
    return height


def calculate_metrics(pred_height, target_height, mask):
    """Calculate metrics in meters"""
    mask_bool = mask > 0.5
    pred_b = pred_height[mask_bool]
    target_b = target_height[mask_bool]
    
    if len(pred_b) == 0:
        return {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0}
    
    rmse = torch.sqrt(torch.mean((pred_b - target_b) ** 2))
    mae = torch.mean(torch.abs(pred_b - target_b))
    ss_res = torch.sum((target_b - pred_b) ** 2)
    ss_tot = torch.sum((target_b - target_b.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    return {'rmse': rmse.item(), 'mae': mae.item(), 'r2': r2.item()}


def calculate_ordinal_accuracy(pred_logits, target_ordinal, mask):
    """Calculate ordinal classification accuracy"""
    pred_binary = (torch.sigmoid(pred_logits) > 0.5).float()
    mask_expanded = mask.expand_as(pred_binary)
    correct = ((pred_binary == target_ordinal) * mask_expanded).sum()
    total = mask_expanded.sum() + 1e-8
    return (correct / total).item()


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, max_height, num_bins):
    model.train()
    running_loss = 0.0
    loss_components = {'ordinal': 0, 'consistency': 0, 'bg': 0, 'edge': 0}
    metrics = {'rmse': 0, 'mae': 0, 'r2': 0, 'ord_acc': 0, 'num_batches': 0}
    
    pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    for images, target_heights, masks, ordinal_targets in pbar:
        images = images.to(device)
        target_heights = target_heights.to(device)
        masks = masks.to(device)
        ordinal_targets = ordinal_targets.to(device)
        
        if masks.sum() < 1.0:
            continue
        
        optimizer.zero_grad()
        ordinal_logits = model(images)
        
        if torch.isnan(ordinal_logits).any():
            continue
        
        loss, loss_dict = criterion(ordinal_logits, ordinal_targets, masks)
        
        if torch.isnan(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        for k in loss_components:
            loss_components[k] += loss_dict.get(k, 0)
        
        with torch.no_grad():
            ordinal_probs = torch.sigmoid(ordinal_logits)
            # Apply mask to predictions for metric calculation
            ordinal_probs = ordinal_probs * masks.expand_as(ordinal_probs)
            pred_heights = ordinal_to_height_tensor(ordinal_probs, max_height, num_bins)
            
            height_metrics = calculate_metrics(pred_heights, target_heights, masks)
            ord_acc = calculate_ordinal_accuracy(ordinal_logits, ordinal_targets, masks)
            
            for k in ['rmse', 'mae', 'r2']:
                metrics[k] += height_metrics[k]
            metrics['ord_acc'] += ord_acc
            metrics['num_batches'] += 1
        
        if metrics['num_batches'] > 0:
            pbar.set_postfix({
                'loss': f'{running_loss/metrics["num_batches"]:.4f}',
                'mae': f'{metrics["mae"]/metrics["num_batches"]:.2f}m',
                'edge': f'{loss_components["edge"]/metrics["num_batches"]:.3f}'
            })
    
    n = metrics['num_batches']
    if n == 0:
        return 0.0, {'rmse': 0, 'mae': 0, 'r2': 0, 'ord_acc': 0}, loss_components
    
    avg_loss_comp = {k: v/n for k, v in loss_components.items()}
    return running_loss/n, {k: metrics[k]/n for k in ['rmse', 'mae', 'r2', 'ord_acc']}, avg_loss_comp


def validate_epoch(model, dataloader, criterion, device, epoch, max_height, num_bins):
    model.eval()
    running_loss = 0.0
    loss_components = {'ordinal': 0, 'consistency': 0, 'bg': 0, 'edge': 0}
    metrics = {'rmse': 0, 'mae': 0, 'r2': 0, 'ord_acc': 0, 'num_batches': 0}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Validation Epoch {epoch}')
        for images, target_heights, masks, ordinal_targets in pbar:
            images = images.to(device)
            target_heights = target_heights.to(device)
            masks = masks.to(device)
            ordinal_targets = ordinal_targets.to(device)
            
            if masks.sum() < 1.0:
                continue
            
            ordinal_logits = model(images)
            
            if torch.isnan(ordinal_logits).any():
                continue
            
            loss, loss_dict = criterion(ordinal_logits, ordinal_targets, masks)
            
            if torch.isnan(loss):
                continue
            
            running_loss += loss.item()
            for k in loss_components:
                loss_components[k] += loss_dict.get(k, 0)
            
            ordinal_probs = torch.sigmoid(ordinal_logits)
            # Apply mask to predictions
            ordinal_probs = ordinal_probs * masks.expand_as(ordinal_probs)
            pred_heights = ordinal_to_height_tensor(ordinal_probs, max_height, num_bins)
            
            height_metrics = calculate_metrics(pred_heights, target_heights, masks)
            ord_acc = calculate_ordinal_accuracy(ordinal_logits, ordinal_targets, masks)
            
            for k in ['rmse', 'mae', 'r2']:
                metrics[k] += height_metrics[k]
            metrics['ord_acc'] += ord_acc
            metrics['num_batches'] += 1
            
            if metrics['num_batches'] > 0:
                pbar.set_postfix({
                    'loss': f'{running_loss/metrics["num_batches"]:.4f}',
                    'mae': f'{metrics["mae"]/metrics["num_batches"]:.2f}m'
                })
    
    n = metrics['num_batches']
    if n == 0:
        return 0.0, {'rmse': 0, 'mae': 0, 'r2': 0, 'ord_acc': 0}, loss_components
    
    avg_loss_comp = {k: v/n for k, v in loss_components.items()}
    return running_loss/n, {k: metrics[k]/n for k in ['rmse', 'mae', 'r2', 'ord_acc']}, avg_loss_comp


# =============================================================================
# IMPROVED VISUALIZATION
# =============================================================================
def visualize_predictions(model, dataloader, device, max_height, num_bins, save_path, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    valid_batch = next((b for b in dataloader if b[2].sum() > 0), None)
    if valid_batch is None:
        return
    
    images, target_heights, masks, ordinal_targets = valid_batch
    
    with torch.no_grad():
        images = images.to(device)
        ordinal_logits = model(images)
        ordinal_probs = torch.sigmoid(ordinal_logits)
        
        # Apply mask to predictions
        masks_dev = masks.to(device)
        ordinal_probs = ordinal_probs * masks_dev.expand_as(ordinal_probs)
        pred_heights = ordinal_to_height_tensor(ordinal_probs, max_height, num_bins)
        
        for i in range(min(num_samples, images.size(0))):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
            
            mask = masks[i, 0].numpy()
            target_h = target_heights[i, 0].numpy()
            pred_h = pred_heights[i, 0].cpu().numpy()
            
            # Uncertainty from ordinal probabilities
            probs = ordinal_probs[i].cpu().numpy()
            eps = 1e-7
            entropy = -np.sum(probs * np.log(probs + eps) + (1 - probs) * np.log(1 - probs + eps), axis=0)
            uncertainty = entropy / num_bins
            
            v_max = max(target_h.max(), pred_h.max(), 1)
            error_map = np.abs(target_h - pred_h) * mask
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')
            
            im1 = axes[i, 1].imshow(target_h, cmap='viridis', vmin=0, vmax=v_max)
            axes[i, 1].set_title(f'GT (Max: {target_h.max():.1f}m)')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
            
            im2 = axes[i, 2].imshow(pred_h, cmap='viridis', vmin=0, vmax=v_max)
            axes[i, 2].set_title(f'Pred (Max: {pred_h.max():.1f}m)')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
            
            im3 = axes[i, 3].imshow(error_map, cmap='hot', vmin=0, vmax=10)
            mae_val = error_map[mask>0].mean() if mask.sum()>0 else 0
            axes[i, 3].set_title(f'Error (MAE: {mae_val:.2f}m)')
            axes[i, 3].axis('off')
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)
            
            im4 = axes[i, 4].imshow(uncertainty * mask, cmap='plasma')
            axes[i, 4].set_title('Uncertainty (Entropy)')
            axes[i, 4].axis('off')
            plt.colorbar(im4, ax=axes[i, 4], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Visualization saved: {save_path}")


def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MAE
    axes[0, 1].plot(epochs, history['train_mae'], label='Train')
    axes[0, 1].plot(epochs, history['val_mae'], label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (m)')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # R²
    axes[0, 2].plot(epochs, history['val_r2'], label='Val R²')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('R²')
    axes[0, 2].set_title('R² Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Loss Components
    axes[1, 0].plot(epochs, history['train_ordinal'], label='Ordinal')
    axes[1, 0].plot(epochs, history['train_consistency'], label='Consistency')
    axes[1, 0].plot(epochs, history['train_bg'], label='Background')
    axes[1, 0].plot(epochs, history['train_edge'], label='Edge')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Ordinal Accuracy
    axes[1, 1].plot(epochs, history['train_ord_acc'], label='Train')
    axes[1, 1].plot(epochs, history['val_ord_acc'], label='Val')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Ordinal Classification Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Learning Rate
    if 'lr' in history:
        axes[1, 2].plot(epochs, history['lr'])
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].grid(True)
        axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Training history saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Model 5 Improved: SegFormer + Advanced Ordinal Regression')
    parser.add_argument('--data_dir', default='dfc2023_height_dataset')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--save_dir', default='./checkpoints_model5_improved')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', default='height-estimation-model5-improved')
    parser.add_argument('--num_bins', type=int, default=100)
    parser.add_argument('--consistency_weight', type=float, default=0.1)
    parser.add_argument('--bg_weight', type=float, default=0.2)
    parser.add_argument('--edge_weight', type=float, default=0.15)
    parser.add_argument('--use_focal', action='store_true', default=True)
    parser.add_argument('--encoder', default='mit_b4', choices=['mit_b2', 'mit_b3', 'mit_b4', 'mit_b5'])
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='Pin memory for faster data transfer')
    parser.add_argument('--persistent_workers', action='store_true', default=True, help='Keep workers alive between epochs')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MODEL 5 IMPROVED: SegFormer + Advanced Ordinal Regression")
    print("=" * 70)
    print(f"\n📋 Architecture: FPN with {args.encoder.upper()} backbone")
    print(f"   Loss: {'Focal' if args.use_focal else 'BCE'} Ordinal + Consistency + BG + Edge")
    print(f"   Weights: Cons={args.consistency_weight}, BG={args.bg_weight}, Edge={args.edge_weight}")
    print(f"   Ordinal Bins: {args.num_bins} (bin width: ~{70/args.num_bins:.2f}m)")
    print(f"   Data Loading: {args.num_workers} workers, pin_memory={args.pin_memory}")
    
    if args.use_wandb:
        wandb.init(project=args.project_name, config=args, name=f"segformer_{args.encoder}_improved")
    
    device = get_device()
    data_dir = Path(args.data_dir)
    
    with open(data_dir / 'height_statistics.json', 'r') as f:
        height_stats = json.load(f)
    
    max_height = height_stats['max']
    
    print(f"\n📊 Height Stats: min={height_stats['min']:.2f}m, max={max_height:.2f}m")
    print(f"   Effective bin width: {max_height / args.num_bins:.2f}m")
    
    with open(data_dir / 'train_data.json', 'r') as f:
        train_data = json.load(f)
    with open(data_dir / 'val_data.json', 'r') as f:
        val_data = json.load(f)
    
    train_dataset = HeightDatasetOrdinal(train_data, height_stats, get_training_transforms(), args.num_bins)
    val_dataset = HeightDatasetOrdinal(val_data, height_stats, get_validation_transforms(), args.num_bins)
    
    # DataLoaders with multi-worker support
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    model = SegFormerOrdinalImproved(num_bins=args.num_bins, encoder_name=args.encoder)
    model.to(device)
    print(f"\n🏗️  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = ImprovedOrdinalLoss(
        consistency_weight=args.consistency_weight,
        bg_weight=args.bg_weight,
        edge_weight=args.edge_weight,
        use_focal=args.use_focal
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    best_mae, patience_counter = float('inf'), 0
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': [],
        'train_r2': [], 'val_r2': [],
        'train_ord_acc': [], 'val_ord_acc': [],
        'train_ordinal': [], 'train_consistency': [], 'train_bg': [], 'train_edge': [],
        'lr': []
    }
    
    print(f"\n🚀 Starting training...")
    
    for epoch in range(args.epochs):
        print(f'\n{"="*30}\nEpoch {epoch+1}/{args.epochs}\n{"="*30}')
        
        train_loss, train_m, train_loss_comp = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1, max_height, args.num_bins
        )
        val_loss, val_m, val_loss_comp = validate_epoch(
            model, val_loader, criterion, device, epoch+1, max_height, args.num_bins
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_m['mae'])
        history['val_mae'].append(val_m['mae'])
        history['train_r2'].append(train_m['r2'])
        history['val_r2'].append(val_m['r2'])
        history['train_ord_acc'].append(train_m['ord_acc'])
        history['val_ord_acc'].append(val_m['ord_acc'])
        history['train_ordinal'].append(train_loss_comp['ordinal'])
        history['train_consistency'].append(train_loss_comp['consistency'])
        history['train_bg'].append(train_loss_comp['bg'])
        history['train_edge'].append(train_loss_comp['edge'])
        history['lr'].append(current_lr)
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mae': train_m['mae'],
                'val_mae': val_m['mae'],
                'val_r2': val_m['r2'],
                'val_ord_acc': val_m['ord_acc'],
                'train_ordinal': train_loss_comp['ordinal'],
                'train_edge': train_loss_comp['edge'],
                'lr': current_lr
            })
        
        print(f'\n📈 Train: Loss={train_loss:.4f}, MAE={train_m["mae"]:.2f}m, OrdAcc={train_m["ord_acc"]:.3f}')
        print(f'   Loss: Ord={train_loss_comp["ordinal"]:.3f}, Cons={train_loss_comp["consistency"]:.3f}, ' + 
              f'BG={train_loss_comp["bg"]:.3f}, Edge={train_loss_comp["edge"]:.3f}')
        print(f'📉 Val: Loss={val_loss:.4f}, MAE={val_m["mae"]:.2f}m, R²={val_m["r2"]:.4f}, OrdAcc={val_m["ord_acc"]:.3f}')
        
        if val_m['mae'] < best_mae and val_m['mae'] > 0:
            best_mae, patience_counter = val_m['mae'], 0
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'best_mae': best_mae,
                'val_metrics': val_m,
                'height_stats': height_stats,
                'config': {
                    'model': 'SegFormerOrdinalImproved',
                    'encoder': args.encoder,
                    'loss': f'{"Focal" if args.use_focal else "BCE"}Ordinal+Consistency+BG+Edge',
                    'num_bins': args.num_bins,
                    'consistency_weight': args.consistency_weight,
                    'bg_weight': args.bg_weight,
                    'edge_weight': args.edge_weight
                }
            }, save_dir / 'best_model.pth')
            print(f'\n✅ Best model saved! MAE: {best_mae:.2f}m')
            visualize_predictions(model, val_loader, device, max_height, args.num_bins,
                                save_dir / f'pred_epoch_{epoch+1}.png')
        else:
            patience_counter += 1
        
        # Plot history every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_training_history(history, save_dir / f'training_history_epoch_{epoch+1}.png')
        
        if patience_counter >= args.patience:
            print(f'\n⏹️  Early stopping')
            break
    
    # Final history plot
    plot_training_history(history, save_dir / 'training_history_final.png')
    
    # Load best model and visualize
    print(f'\n{"="*30}')
    print('📊 Creating final visualizations from best model...')
    checkpoint = torch.load(save_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    visualize_predictions(model, val_loader, device, max_height, args.num_bins,
                         save_dir / 'pred_best_model.png', num_samples=6)
    
    print(f'\n{"="*30}\n🎉 Training completed! Best MAE: {best_mae:.2f}m\n{"="*30}')
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()