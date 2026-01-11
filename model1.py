#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
MODEL 1 IMPROVED: DeepLabV3+ + Background Regularization
=============================================================================

IMPROVEMENTS:
-------------
1. Background Regularization Loss:
   - Penalizes non-zero predictions on background pixels
   - Prevents height bleeding into non-building areas
   - Numerically stable for log-scaled outputs

2. Enhanced Metrics Tracking:
   - Training loss (total, building, background components)
   - Validation MAE, RMSE
   - Learning curves visualization

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
# DATASET WITH LOG-SCALE NORMALIZATION
# =============================================================================
class HeightDatasetLogScale(Dataset):
    """Dataset with Log-Scale Normalization"""
    
    def __init__(self, data_list, height_stats, transform=None):
        self.data_list = data_list
        self.transform = transform
        
        # Height statistics
        self.height_max = height_stats['max']
        self.height_min = height_stats['min']
        self.height_mean = height_stats['mean']
        self.height_std = height_stats['std']
        
        # Log-scale normalization constant
        self.log_max = np.log(self.height_max + 1)
        
        print(f"📊 Dataset: {len(self.data_list)} samples")
        print(f"   Height range: {self.height_min:.2f}m - {self.height_max:.2f}m")
        print(f"   Log normalization constant: {self.log_max:.4f}")
    
    def normalize_height(self, height):
        """Log-scale normalization: [0, max] -> [0, 1]"""
        return np.log(height + 1) / self.log_max
    
    def denormalize_height(self, height_norm):
        """Inverse log-scale: [0, 1] -> [0, max]"""
        return np.exp(height_norm * self.log_max) - 1
    
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
        
        # Apply log-scale normalization
        height_map = np.clip(height_map, 0, self.height_max)
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
            A.GaussNoise(var_limit=(10.0, 50.0)),
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
# IMPROVED LOSS WITH BACKGROUND REGULARIZATION
# =============================================================================
class ScaleInvariantLossWithBackground(nn.Module):
    """
    Scale-Invariant Loss with Background Regularization
    
    PROBLEM:
    --------
    When training only on building pixels (mask=1), the model has no supervision
    on background pixels (mask=0). This leads to:
    - Non-zero height predictions on roads, trees, grass
    - Height "bleeding" into non-building areas
    - Poor visual quality and incorrect height estimation
    
    SOLUTION:
    ---------
    Combined loss: L = L_building + λ * L_background
    
    1. L_building: Scale-invariant loss on building pixels
       - Same as original (Eigen et al., 2014)
       - Optimizes height estimation where buildings exist
    
    2. L_background: Regularization on background pixels
       - Penalizes non-zero predictions on background
       - Soft constraint (not hard zero enforcement)
       - Uses MSE or L1 to push predictions toward zero
    
    WHY THIS WORKS:
    ---------------
    - Building loss: Learns accurate heights on buildings
    - Background loss: Suppresses spurious predictions on background
    - λ balances the two objectives
    - Numerically stable in log-space (predictions are in [0,1])
    
    PARAMETERS:
    -----------
    lambda_building: Weight for scale-invariant term (default: 0.5)
    lambda_background: Weight for background regularization (default: 0.1)
    background_loss_type: 'mse' or 'l1' (default: 'l1')
    epsilon: Small constant for numerical stability (default: 1e-6)
    """
    
    def __init__(self, 
                 lambda_building=0.5, 
                 lambda_background=0.1,
                 background_loss_type='l1',
                 epsilon=1e-6):
        super().__init__()
        self.lambda_building = lambda_building
        self.lambda_background = lambda_background
        self.background_loss_type = background_loss_type
        self.epsilon = epsilon
        
        print(f"\n🎯 Loss Configuration:")
        print(f"   Building loss: Scale-Invariant (λ={lambda_building})")
        print(f"   Background loss: {background_loss_type.upper()} (λ={lambda_background})")
        print(f"   Combined: L = L_building + {lambda_background} * L_background")
    
    def forward(self, pred, target, mask):
        """
        Compute combined loss
        
        Args:
            pred: Predicted height map [B, 1, H, W] in log-normalized space [0, 1]
            target: Target height map [B, 1, H, W] in log-normalized space [0, 1]
            mask: Building mask [B, 1, H, W], 1=building, 0=background
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with loss components for logging
        """
        # Ensure values are in valid range [0, 1]
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # =====================================================================
        # PART 1: BUILDING LOSS (Scale-Invariant)
        # =====================================================================
        # Convert to log-space: log(exp(x * log_max) - 1 + ε) ≈ x * log_max for x close to normalized values
        # Since pred and target are already in log-normalized space [0,1], 
        # we work directly with them for scale-invariant computation
        
        # Get building pixels only
        building_mask = (mask > 0.5).float()
        n_building = building_mask.sum() + self.epsilon
        
        if n_building > 1:
            # Compute difference in log-normalized space
            # This is equivalent to log-ratio since: log(a) - log(b) = log(a/b)
            diff = (pred - target) * building_mask
            
            # Scale-invariant loss components
            term1 = (diff ** 2).sum() / n_building
            term2 = (self.lambda_building / (n_building ** 2)) * (diff.sum() ** 2)
            
            loss_building = term1 - term2
        else:
            # No building pixels in batch
            loss_building = torch.tensor(0.0, device=pred.device)
        
        # =====================================================================
        # PART 2: BACKGROUND LOSS (Regularization)
        # =====================================================================
        # Goal: Push background predictions toward zero
        # Background mask: inverse of building mask
        background_mask = (mask <= 0.5).float()
        n_background = background_mask.sum() + self.epsilon
        
        if n_background > 0:
            # Target for background is zero (in normalized space)
            background_target = torch.zeros_like(pred)
            
            if self.background_loss_type == 'mse':
                # MSE: (pred - 0)^2
                loss_background = ((pred - background_target) ** 2 * background_mask).sum() / n_background
            elif self.background_loss_type == 'l1':
                # L1: |pred - 0|
                loss_background = (torch.abs(pred - background_target) * background_mask).sum() / n_background
            else:
                raise ValueError(f"Unknown background_loss_type: {self.background_loss_type}")
        else:
            loss_background = torch.tensor(0.0, device=pred.device)
        
        # =====================================================================
        # COMBINE LOSSES
        # =====================================================================
        total_loss = loss_building + self.lambda_background * loss_background
        
        # Return detailed breakdown for logging
        loss_dict = {
            'total': total_loss.item(),
            'building': loss_building.item(),
            'background': loss_background.item(),
            'n_building': n_building.item(),
            'n_background': n_background.item()
        }
        
        return total_loss, loss_dict


# =============================================================================
# METRICS TRACKER
# =============================================================================
class MetricsTracker:
    """
    Track and visualize training metrics over epochs
    """
    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss_total': [],
            'train_loss_building': [],
            'train_loss_background': [],
            'val_loss_total': [],
            'val_mae': [],
            'val_rmse': [],
            'val_r2': [],
            'learning_rate': []
        }
    
    def update(self, epoch, train_metrics, val_metrics, lr):
        """Update metrics for current epoch"""
        self.history['epoch'].append(epoch)
        self.history['train_loss_total'].append(train_metrics.get('loss', 0))
        self.history['train_loss_building'].append(train_metrics.get('loss_building', 0))
        self.history['train_loss_background'].append(train_metrics.get('loss_background', 0))
        self.history['val_loss_total'].append(val_metrics.get('loss', 0))
        self.history['val_mae'].append(val_metrics.get('mae', 0))
        self.history['val_rmse'].append(val_metrics.get('rmse', 0))
        self.history['val_r2'].append(val_metrics.get('r2', 0))
        self.history['learning_rate'].append(lr)
    
    def plot_curves(self, save_path):
        """
        Plot comprehensive training curves
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        epochs = self.history['epoch']
        
        # 1. Total Loss (Train + Val)
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss_total'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, self.history['val_loss_total'], 'r-', label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Loss Components (Building vs Background)
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_loss_building'], 'g-', label='Building', linewidth=2)
        ax.plot(epochs, self.history['train_loss_background'], 'm-', label='Background', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Validation MAE
        ax = axes[0, 2]
        ax.plot(epochs, self.history['val_mae'], 'r-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE (meters)')
        ax.set_title('Validation MAE')
        ax.grid(True, alpha=0.3)
        if self.history['val_mae']:
            best_mae = min(self.history['val_mae'])
            best_epoch = self.history['val_mae'].index(best_mae) + 1
            ax.axhline(y=best_mae, color='g', linestyle='--', alpha=0.5, label=f'Best: {best_mae:.2f}m')
            ax.scatter([best_epoch], [best_mae], color='g', s=100, zorder=5)
            ax.legend()
        
        # 4. Validation RMSE
        ax = axes[1, 0]
        ax.plot(epochs, self.history['val_rmse'], 'orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE (meters)')
        ax.set_title('Validation RMSE')
        ax.grid(True, alpha=0.3)
        if self.history['val_rmse']:
            best_rmse = min(self.history['val_rmse'])
            best_epoch = self.history['val_rmse'].index(best_rmse) + 1
            ax.axhline(y=best_rmse, color='g', linestyle='--', alpha=0.5, label=f'Best: {best_rmse:.2f}m')
            ax.scatter([best_epoch], [best_rmse], color='g', s=100, zorder=5)
            ax.legend()
        
        # 5. Validation R²
        ax = axes[1, 1]
        ax.plot(epochs, self.history['val_r2'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('R² Score')
        ax.set_title('Validation R²')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        if self.history['val_r2']:
            best_r2 = max(self.history['val_r2'])
            best_epoch = self.history['val_r2'].index(best_r2) + 1
            ax.axhline(y=best_r2, color='g', linestyle='--', alpha=0.5, label=f'Best: {best_r2:.4f}')
            ax.scatter([best_epoch], [best_r2], color='g', s=100, zorder=5)
            ax.legend()
        
        # 6. Learning Rate
        ax = axes[1, 2]
        ax.plot(epochs, self.history['learning_rate'], 'brown', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Training curves saved: {save_path}")
    
    def save_csv(self, save_path):
        """Save metrics history to CSV"""
        import pandas as pd
        df = pd.DataFrame(self.history)
        df.to_csv(save_path, index=False)
        print(f"💾 Metrics saved: {save_path}")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, log_max):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_building_loss = 0
    total_background_loss = 0
    n_batches = len(dataloader)
    
    # For MAE/RMSE computation
    all_preds = []
    all_targets = []
    all_masks = []
    
    pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    
    for images, target_heights, masks in pbar:
        images = images.to(device)
        target_heights = target_heights.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        pred_heights = torch.sigmoid(outputs)
        
        # Compute loss
        loss, loss_dict = criterion(pred_heights, target_heights, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss_dict['total']
        total_building_loss += loss_dict['building']
        total_background_loss += loss_dict['background']
        
        # Store for metrics
        all_preds.append(pred_heights.detach())
        all_targets.append(target_heights.detach())
        all_masks.append(masks.detach())
        
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'build': f"{loss_dict['building']:.4f}",
            'back': f"{loss_dict['background']:.4f}"
        })
    
    # Compute epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Denormalize for MAE/RMSE
    all_preds_m = torch.exp(all_preds * log_max) - 1
    all_targets_m = torch.exp(all_targets * log_max) - 1
    
    # Compute metrics on building pixels only
    building_mask = (all_masks > 0.5).float()
    n_building = building_mask.sum()
    
    if n_building > 0:
        mae = (torch.abs(all_preds_m - all_targets_m) * building_mask).sum() / n_building
        rmse = torch.sqrt(((all_preds_m - all_targets_m) ** 2 * building_mask).sum() / n_building)
    else:
        mae = torch.tensor(0.0)
        rmse = torch.tensor(0.0)
    
    metrics = {
        'loss': total_loss / n_batches,
        'loss_building': total_building_loss / n_batches,
        'loss_background': total_background_loss / n_batches,
        'mae': mae.item(),
        'rmse': rmse.item()
    }
    
    return metrics


def validate_epoch(model, dataloader, criterion, device, epoch, log_max):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0
    total_building_loss = 0
    total_background_loss = 0
    n_batches = len(dataloader)
    
    all_preds = []
    all_targets = []
    all_masks = []
    
    pbar = tqdm(dataloader, desc=f'Validation Epoch {epoch}')
    
    with torch.no_grad():
        for images, target_heights, masks in pbar:
            images = images.to(device)
            target_heights = target_heights.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            pred_heights = torch.sigmoid(outputs)
            
            loss, loss_dict = criterion(pred_heights, target_heights, masks)
            
            total_loss += loss_dict['total']
            total_building_loss += loss_dict['building']
            total_background_loss += loss_dict['background']
            
            all_preds.append(pred_heights)
            all_targets.append(target_heights)
            all_masks.append(masks)
            
            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
    
    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Denormalize
    all_preds_m = torch.exp(all_preds * log_max) - 1
    all_targets_m = torch.exp(all_targets * log_max) - 1
    
    building_mask = (all_masks > 0.5).float()
    n_building = building_mask.sum()
    
    if n_building > 0:
        mae = (torch.abs(all_preds_m - all_targets_m) * building_mask).sum() / n_building
        rmse = torch.sqrt(((all_preds_m - all_targets_m) ** 2 * building_mask).sum() / n_building)
        
        # R² score
        ss_res = ((all_targets_m - all_preds_m) ** 2 * building_mask).sum()
        ss_tot = ((all_targets_m - all_targets_m.mean()) ** 2 * building_mask).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        # Scale-invariant RMSE
        diff = torch.log(all_preds_m + 1) - torch.log(all_targets_m + 1)
        diff_masked = diff * building_mask
        si_rmse = torch.sqrt((diff_masked ** 2).sum() / n_building)
    else:
        mae = torch.tensor(0.0)
        rmse = torch.tensor(0.0)
        r2 = torch.tensor(0.0)
        si_rmse = torch.tensor(0.0)
    
    metrics = {
        'loss': total_loss / n_batches,
        'loss_building': total_building_loss / n_batches,
        'loss_background': total_background_loss / n_batches,
        'mae': mae.item(),
        'rmse': rmse.item(),
        'r2': r2.item(),
        'si_rmse': si_rmse.item()
    }
    
    return metrics


def visualize_predictions(model, dataloader, device, log_max, save_path, num_samples=4):
    """Visualize predictions with background regularization effects"""
    model.eval()
    
    # Get one batch
    for batch in dataloader:
        valid_batch = batch
        break
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    images, target_heights, masks = valid_batch
    
    with torch.no_grad():
        images = images.to(device)
        pred_heights = torch.sigmoid(model(images))
        
        for i in range(min(num_samples, images.size(0))):
            # Denormalize image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            mask = masks[i, 0].numpy()
            
            # Denormalize heights
            target_h = np.exp(target_heights[i, 0].numpy() * log_max) - 1
            pred_h = np.exp(pred_heights[i, 0].cpu().numpy() * log_max) - 1
            
            v_max = max(target_h.max(), pred_h.max(), 1)
            error_map = np.abs(target_h - pred_h) * mask
            
            # Background prediction (should be close to zero with regularization)
            background_pred = pred_h * (1 - mask)
            
            # Plot
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Building Mask')
            axes[i, 1].axis('off')
            
            im2 = axes[i, 2].imshow(target_h, cmap='viridis', vmin=0, vmax=v_max)
            axes[i, 2].set_title(f'Ground Truth\nMax: {target_h[mask > 0].max():.1f}m' if mask.sum() > 0 else 'Ground Truth')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
            
            im3 = axes[i, 3].imshow(pred_h, cmap='viridis', vmin=0, vmax=v_max)
            bg_mean = background_pred[mask == 0].mean() if (mask == 0).sum() > 0 else 0
            axes[i, 3].set_title(f'Prediction\nBG mean: {bg_mean:.2f}m')
            axes[i, 3].axis('off')
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)
            
            im4 = axes[i, 4].imshow(error_map, cmap='hot', vmin=0, vmax=10)
            mae_val = error_map[mask > 0].mean() if mask.sum() > 0 else 0
            axes[i, 4].set_title(f'Error Map\nMAE: {mae_val:.2f}m')
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
    parser = argparse.ArgumentParser(description='Model 1 Improved: With Background Regularization')
    parser.add_argument('--data_dir', default='dfc2023_height_dataset', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lambda_building', type=float, default=0.5, help='Weight for building loss')
    parser.add_argument('--lambda_background', type=float, default=0.1, help='Weight for background regularization')
    parser.add_argument('--background_loss_type', default='l1', choices=['l1', 'mse'], help='Type of background loss')
    parser.add_argument('--save_dir', default='./checkpoints_model1_improved')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', default='height-estimation-improved')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MODEL 1 IMPROVED: DeepLabV3+ + Background Regularization")
    print("=" * 80)
    print("\n📋 Configuration:")
    print(f"   Architecture: DeepLabV3+ with ResNet50 encoder")
    print(f"   Building Loss: Scale-Invariant (λ={args.lambda_building})")
    print(f"   Background Loss: {args.background_loss_type.upper()} (λ={args.lambda_background})")
    print(f"   Combined Loss: L = L_building + {args.lambda_background} * L_background")
    print()
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project=args.project_name, config=args, name="deeplabv3_background_reg")
    
    # Device
    device = get_device()
    
    # Load data
    data_dir = Path(args.data_dir)
    
    with open(data_dir / 'height_statistics.json', 'r') as f:
        height_stats = json.load(f)
    
    log_max = np.log(height_stats['max'] + 1)
    
    print(f"\n📊 Height Statistics:")
    print(f"   Min: {height_stats['min']:.2f}m")
    print(f"   Max: {height_stats['max']:.2f}m")
    print(f"   Mean: {height_stats['mean']:.2f}m")
    print(f"   Log normalization constant: {log_max:.4f}")
    
    with open(data_dir / 'train_data.json', 'r') as f:
        train_data = json.load(f)
    with open(data_dir / 'val_data.json', 'r') as f:
        val_data = json.load(f)
    
    print(f"\n📂 Dataset:")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val: {len(val_data)} samples")
    
    # Create datasets
    train_dataset = HeightDatasetLogScale(train_data, height_stats, get_training_transforms())
    val_dataset = HeightDatasetLogScale(val_data, height_stats, get_validation_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    model.to(device)
    
    print(f"\n🏗️  Model: DeepLabV3+ with ResNet50")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = ScaleInvariantLossWithBackground(
        lambda_building=args.lambda_building,
        lambda_background=args.lambda_background,
        background_loss_type=args.background_loss_type
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    best_mae = float('inf')
    patience_counter = 0
    
    print(f"\n🚀 Starting training...")
    
    for epoch in range(args.epochs):
        print(f'\n{"=" * 70}')
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print(f'{"=" * 70}')
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1, log_max)
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch + 1, log_max)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Update metrics tracker
        metrics_tracker.update(epoch + 1, train_metrics, val_metrics, current_lr)
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss_total': train_metrics['loss'],
                'train_loss_building': train_metrics['loss_building'],
                'train_loss_background': train_metrics['loss_background'],
                'val_loss_total': val_metrics['loss'],
                'val_mae': val_metrics['mae'],
                'val_rmse': val_metrics['rmse'],
                'val_r2': val_metrics['r2'],
                'val_si_rmse': val_metrics['si_rmse'],
                'lr': current_lr
            })
        
        print(f'\n📈 Training:')
        print(f'   Total Loss: {train_metrics["loss"]:.4f}')
        print(f'   Building Loss: {train_metrics["loss_building"]:.4f}')
        print(f'   Background Loss: {train_metrics["loss_background"]:.4f}')
        print(f'   MAE: {train_metrics["mae"]:.2f}m, RMSE: {train_metrics["rmse"]:.2f}m')
        
        print(f'\n📉 Validation:')
        print(f'   Total Loss: {val_metrics["loss"]:.4f}')
        print(f'   MAE: {val_metrics["mae"]:.2f}m, RMSE: {val_metrics["rmse"]:.2f}m')
        print(f'   R²: {val_metrics["r2"]:.4f}, SI-RMSE: {val_metrics["si_rmse"]:.4f}')
        
        # Save best model
        if val_metrics['mae'] < best_mae and val_metrics['mae'] > 0:
            best_mae = val_metrics['mae']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mae': best_mae,
                'val_metrics': val_metrics,
                'height_stats': height_stats,
                'log_max': log_max,
                'config': {
                    'model': 'DeepLabV3Plus',
                    'encoder': 'resnet50',
                    'loss': 'ScaleInvariantWithBackground',
                    'lambda_building': args.lambda_building,
                    'lambda_background': args.lambda_background,
                    'background_loss_type': args.background_loss_type
                }
            }, save_dir / 'best_model.pth')
            
            print(f'\n✅ New best model! MAE: {best_mae:.2f}m')
            
            visualize_predictions(model, val_loader, device, log_max, save_dir / f'pred_epoch_{epoch+1}.png')
        else:
            patience_counter += 1
        
        # Plot training curves every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics_tracker.plot_curves(save_dir / 'training_curves.png')
            metrics_tracker.save_csv(save_dir / 'metrics_history.csv')
        
        if patience_counter >= args.patience:
            print(f'\n⏹️  Early stopping after {args.patience} epochs without improvement')
            break
    
    # Final plots
    metrics_tracker.plot_curves(save_dir / 'training_curves_final.png')
    metrics_tracker.save_csv(save_dir / 'metrics_history_final.csv')
    
    print(f'\n{"=" * 70}')
    print(f'🎉 Training completed!')
    print(f'   Best MAE: {best_mae:.2f}m')
    print(f'   Model saved: {save_dir / "best_model.pth"}')
    print(f'   Metrics saved: {save_dir / "metrics_history_final.csv"}')
    print(f'   Curves saved: {save_dir / "training_curves_final.png"}')
    print(f'{"=" * 70}')
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()