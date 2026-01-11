#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
MODEL 4 V3: FPN + Multi-Task Learning - IMPROVED EDGE DETECTION
=============================================================================

YENİ İYİLEŞTİRMELER (V2 -> V3):
1. ✅ Edge extraction daha güçlü (5x5 kernel, iterations: 3+2)
2. ✅ Edge loss weight artırıldı (1.0 -> 2.0)
3. ✅ Focal Loss gamma artırıldı (2.0 -> 3.0) - daha aggressive
4. ✅ Edge head daha derin (3-layer → 4-layer)
5. ✅ Comprehensive training curves (son epoch'ta detaylı grafik)
6. ✅ Edge metrics tracking iyileştirildi

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
from matplotlib.gridspec import GridSpec


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


class HeightDatasetMultiTask(Dataset):
    def __init__(self, data_list, height_stats, transform=None):
        self.data_list = data_list
        self.transform = transform
        self.height_max = height_stats['max']
        self.height_min = height_stats['min']
        self.height_mean = height_stats['mean']
        self.height_std = height_stats['std']
        self.range = self.height_max - self.height_min + 1e-8
        
        print(f"📊 Dataset: {len(self.data_list)} samples")
        print(f"   Height range: {self.height_min:.2f}m - {self.height_max:.2f}m")
    
    def extract_edges(self, mask):
        """
        V3: Daha güçlü edge extraction
        - 5x5 kernel (3x3 -> 5x5)
        - Daha fazla dilation/erosion iterations
        """
        kernel = np.ones((5, 5), np.uint8)  # 3x3 -> 5x5
        dilated = cv2.dilate(mask, kernel, iterations=3)  # 2 -> 3
        eroded = cv2.erode(mask, kernel, iterations=2)  # 1 -> 2
        edge = dilated - eroded
        return edge
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        
        image = cv2.cvtColor(cv2.imread(sample['image']), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        height_map = np.load(sample['height'])
        
        # Min-Max normalization
        height_map = np.clip((height_map - self.height_min) / self.range, 0, 1)
        
        # Extract edges with improved method
        edge_mask = self.extract_edges((mask > 0.5).astype(np.uint8)).astype(np.float32)
        
        if self.transform:
            transformed = self.transform(image=image, masks=[mask, height_map, edge_mask])
            image = transformed['image']
            mask, height_map, edge_mask = transformed['masks']
        
        mask = torch.from_numpy(mask).unsqueeze(0) if isinstance(mask, np.ndarray) else mask.unsqueeze(0)
        height_map = torch.from_numpy(height_map).unsqueeze(0) if isinstance(height_map, np.ndarray) else height_map.unsqueeze(0)
        edge_mask = torch.from_numpy(edge_mask).unsqueeze(0) if isinstance(edge_mask, np.ndarray) else edge_mask.unsqueeze(0)
        
        return image, height_map, mask, edge_mask


def get_training_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5), 
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.HueSaturationValue(10, 15, 10), 
            A.RandomBrightnessContrast(0.2, 0.2)
        ], p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3), 
            A.GaussNoise(var_limit=(10.0, 50.0))
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_validation_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class FPNMultiTaskModel(nn.Module):
    """V3: Edge head daha derin (4-layer)"""
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet"):
        super().__init__()
        
        self.fpn = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=64,
            activation=None
        )
        
        # Height head (unchanged)
        self.height_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        
        # V3: Edge head - DAHA DERİN (4-layer)
        self.edge_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),  # Extra layer
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
    
    def forward(self, x):
        features = self.fpn(x)
        height = torch.sigmoid(self.height_head(features))
        edge = torch.sigmoid(self.edge_head(features))
        return height, edge


class FocalBCELoss(nn.Module):
    """
    V3: Focal Loss with higher gamma (3.0)
    Daha aggressive focusing on hard examples
    """
    def __init__(self, alpha=0.25, gamma=3.0):  # 2.0 -> 3.0
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class MultiTaskLossV3(nn.Module):
    """
    V3: Improved Multi-Task Loss
    
    L_total = α * L_height + β * L_edge + γ * L_background
    
    V3 Changes:
    - Edge weight artırıldı (default 2.0)
    - Focal Loss gamma artırıldı (3.0)
    """
    
    def __init__(self, height_weight=1.0, edge_weight=2.0, background_weight=0.3):
        super().__init__()
        self.height_weight = height_weight
        self.edge_weight = edge_weight
        self.background_weight = background_weight
        self.focal_bce = FocalBCELoss(alpha=0.25, gamma=3.0)  # gamma: 2.0 -> 3.0
    
    def forward(self, pred_height, pred_edge, target_height, building_mask, target_edge):
        n_building = building_mask.sum() + 1e-8
        
        # 1. Height loss ON BUILDINGS (MSE + L1)
        mse_loss = ((pred_height - target_height) ** 2) * building_mask
        l1_loss = torch.abs(pred_height - target_height) * building_mask
        height_loss = (0.5 * mse_loss.sum() + 0.5 * l1_loss.sum()) / n_building
        
        # 2. Background suppression
        background_mask = 1.0 - building_mask
        n_background = background_mask.sum() + 1e-8
        background_loss = (pred_height ** 2) * background_mask
        background_loss = background_loss.sum() / n_background
        
        # 3. Edge loss (Focal BCE with gamma=3.0)
        edge_loss = self.focal_bce(pred_edge, target_edge)
        
        # Combined loss
        total_loss = (self.height_weight * height_loss + 
                      self.edge_weight * edge_loss +
                      self.background_weight * background_loss)
        
        return total_loss, height_loss, edge_loss, background_loss


def calculate_metrics(pred, target, mask, height_min, height_range):
    pred_m = torch.clamp(pred * height_range + height_min, min=0)
    target_m = torch.clamp(target * height_range + height_min, min=0)
    
    mask_bool = mask > 0.5
    pred_b, target_b = pred_m[mask_bool], target_m[mask_bool]
    
    if len(pred_b) == 0:
        return {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0}
    
    rmse = torch.sqrt(torch.mean((pred_b - target_b) ** 2))
    mae = torch.mean(torch.abs(pred_b - target_b))
    ss_res = torch.sum((target_b - pred_b) ** 2)
    ss_tot = torch.sum((target_b - target_b.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    return {'rmse': rmse.item(), 'mae': mae.item(), 'r2': r2.item()}


def calculate_edge_metrics(pred_edge, target_edge):
    pred_binary = (pred_edge > 0.5).float()
    intersection = (pred_binary * target_edge).sum()
    union = pred_binary.sum() + target_edge.sum() - intersection
    iou = intersection / (union + 1e-8)
    
    tp = (pred_binary * target_edge).sum()
    fp = (pred_binary * (1 - target_edge)).sum()
    fn = ((1 - pred_binary) * target_edge).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'edge_iou': iou.item(), 
        'edge_f1': f1.item(), 
        'edge_precision': precision.item(), 
        'edge_recall': recall.item()
    }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, height_min, height_range):
    model.train()
    running_loss = 0.0
    running_h_loss = 0.0
    running_e_loss = 0.0
    running_bg_loss = 0.0
    metrics = {'rmse': 0, 'mae': 0, 'r2': 0, 'edge_iou': 0, 'edge_f1': 0, 
               'edge_precision': 0, 'edge_recall': 0, 'num_batches': 0}
    
    pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    for images, target_heights, masks, target_edges in pbar:
        images = images.to(device)
        target_heights = target_heights.to(device)
        masks = masks.to(device)
        target_edges = target_edges.to(device)
        
        if masks.sum() < 1.0:
            continue
        
        optimizer.zero_grad()
        pred_heights, pred_edges = model(images)
        
        if torch.isnan(pred_heights).any() or torch.isnan(pred_edges).any():
            continue
        
        total_loss, h_loss, e_loss, bg_loss = criterion(
            pred_heights, pred_edges, target_heights, masks, target_edges
        )
        
        if torch.isnan(total_loss):
            continue
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += total_loss.item()
        running_h_loss += h_loss.item()
        running_e_loss += e_loss.item()
        running_bg_loss += bg_loss.item()
        
        with torch.no_grad():
            height_metrics = calculate_metrics(pred_heights, target_heights, masks, height_min, height_range)
            edge_metrics = calculate_edge_metrics(pred_edges, target_edges)
            for k in ['rmse', 'mae', 'r2']:
                metrics[k] += height_metrics[k]
            for k in ['edge_iou', 'edge_f1', 'edge_precision', 'edge_recall']:
                metrics[k] += edge_metrics[k]
            metrics['num_batches'] += 1
        
        if metrics['num_batches'] > 0:
            n = metrics['num_batches']
            pbar.set_postfix({
                'loss': f'{running_loss/n:.4f}',
                'mae': f'{metrics["mae"]/n:.2f}m',
                'e_f1': f'{metrics["edge_f1"]/n:.3f}'
            })
    
    n = metrics['num_batches']
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, {
            'rmse': 0, 'mae': 0, 'r2': 0, 'edge_iou': 0, 'edge_f1': 0,
            'edge_precision': 0, 'edge_recall': 0
        }
    
    avg_metrics = {k: metrics[k]/n for k in [
        'rmse', 'mae', 'r2', 'edge_iou', 'edge_f1', 'edge_precision', 'edge_recall'
    ]}
    return running_loss/n, running_h_loss/n, running_e_loss/n, running_bg_loss/n, avg_metrics


def validate_epoch(model, dataloader, criterion, device, epoch, height_min, height_range):
    model.eval()
    running_loss = 0.0
    running_h_loss = 0.0
    running_e_loss = 0.0
    running_bg_loss = 0.0
    metrics = {'rmse': 0, 'mae': 0, 'r2': 0, 'edge_iou': 0, 'edge_f1': 0,
               'edge_precision': 0, 'edge_recall': 0, 'num_batches': 0}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Validation Epoch {epoch}')
        for images, target_heights, masks, target_edges in pbar:
            images = images.to(device)
            target_heights = target_heights.to(device)
            masks = masks.to(device)
            target_edges = target_edges.to(device)
            
            if masks.sum() < 1.0:
                continue
            
            pred_heights, pred_edges = model(images)
            
            if torch.isnan(pred_heights).any():
                continue
            
            total_loss, h_loss, e_loss, bg_loss = criterion(
                pred_heights, pred_edges, target_heights, masks, target_edges
            )
            
            if torch.isnan(total_loss):
                continue
            
            running_loss += total_loss.item()
            running_h_loss += h_loss.item()
            running_e_loss += e_loss.item()
            running_bg_loss += bg_loss.item()
            
            height_metrics = calculate_metrics(pred_heights, target_heights, masks, height_min, height_range)
            edge_metrics = calculate_edge_metrics(pred_edges, target_edges)
            
            for k in ['rmse', 'mae', 'r2']:
                metrics[k] += height_metrics[k]
            for k in ['edge_iou', 'edge_f1', 'edge_precision', 'edge_recall']:
                metrics[k] += edge_metrics[k]
            metrics['num_batches'] += 1
            
            if metrics['num_batches'] > 0:
                n = metrics['num_batches']
                pbar.set_postfix({
                    'loss': f'{running_loss/n:.4f}',
                    'mae': f'{metrics["mae"]/n:.2f}m',
                    'e_f1': f'{metrics["edge_f1"]/n:.3f}'
                })
    
    n = metrics['num_batches']
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, {
            'rmse': 0, 'mae': 0, 'r2': 0, 'edge_iou': 0, 'edge_f1': 0,
            'edge_precision': 0, 'edge_recall': 0
        }
    
    avg_metrics = {k: metrics[k]/n for k in [
        'rmse', 'mae', 'r2', 'edge_iou', 'edge_f1', 'edge_precision', 'edge_recall'
    ]}
    return running_loss/n, running_h_loss/n, running_e_loss/n, running_bg_loss/n, avg_metrics


def visualize_predictions(model, dataloader, device, height_min, height_range, save_path, num_samples=4):
    """Visualize model predictions"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 5, figsize=(22, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    valid_batch = next((b for b in dataloader if b[2].sum() > 0), None)
    if valid_batch is None:
        return
    
    images, target_heights, masks, target_edges = valid_batch
    
    with torch.no_grad():
        images = images.to(device)
        pred_heights, pred_edges = model(images)
        
        for i in range(min(num_samples, images.size(0))):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
            
            mask = masks[i, 0].numpy()
            target_h = np.clip(target_heights[i, 0].numpy() * height_range + height_min, 0, None)
            pred_h = np.clip(pred_heights[i, 0].cpu().numpy() * height_range + height_min, 0, None)
            pred_e = pred_edges[i, 0].cpu().numpy()
            target_e = target_edges[i, 0].numpy()
            
            # Apply mask
            target_h_masked = target_h * mask
            pred_h_masked = pred_h * mask
            
            v_max = max(target_h_masked[mask > 0].max() if mask.sum() > 0 else 1, 1)
            error_map = np.abs(target_h - pred_h) * mask
            
            # 1. Input
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input', fontsize=10)
            axes[i, 0].axis('off')
            
            # 2. GT Height
            gt_max = target_h_masked[mask > 0].max() if mask.sum() > 0 else 0
            im1 = axes[i, 1].imshow(target_h_masked, cmap='viridis', vmin=0, vmax=v_max)
            axes[i, 1].set_title(f'GT Height\nMax: {gt_max:.1f}m', fontsize=10)
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
            
            # 3. Pred Height
            pred_max = pred_h_masked[mask > 0].max() if mask.sum() > 0 else 0
            im2 = axes[i, 2].imshow(pred_h_masked, cmap='viridis', vmin=0, vmax=v_max)
            axes[i, 2].set_title(f'Pred Height\nMax: {pred_max:.1f}m', fontsize=10)
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
            
            # 4. Error Map
            mae_val = error_map[mask > 0].mean() if mask.sum() > 0 else 0
            im3 = axes[i, 3].imshow(error_map, cmap='hot', vmin=0, vmax=10)
            axes[i, 3].set_title(f'Error Map\nMAE: {mae_val:.2f}m', fontsize=10)
            axes[i, 3].axis('off')
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)
            
            # 5. Edge comparison (RGB overlay)
            edge_overlay = np.zeros((*pred_e.shape, 3))
            edge_overlay[:, :, 0] = pred_e  # Red = Predicted edges
            edge_overlay[:, :, 1] = target_e  # Green = GT edges
            # Yellow = both, Red = only pred, Green = only GT
            
            axes[i, 4].imshow(edge_overlay)
            pred_sum = pred_e.sum()
            gt_sum = target_e.sum()
            axes[i, 4].set_title(f'Edges (R:Pred, G:GT)\nPred:{pred_sum:.0f}, GT:{gt_sum:.0f}', fontsize=10)
            axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Visualization saved: {save_path}")


def plot_comprehensive_training_curves(history, save_path):
    """
    V3: Comprehensive training curves
    Displays all important metrics in one figure
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # 1. Total Loss (Train vs Val)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Total Loss (Train vs Val)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Training Loss Components
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['train_height_loss'], 'g-', label='Height', linewidth=2)
    ax2.plot(epochs, history['train_edge_loss'], 'orange', label='Edge', linewidth=2)
    ax2.plot(epochs, history['train_bg_loss'], 'purple', label='Background', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss Components', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Validation MAE
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history['val_mae'], 'r-', linewidth=2)
    best_mae = min(history['val_mae'])
    best_epoch = history['val_mae'].index(best_mae) + 1
    ax3.axhline(y=best_mae, color='g', linestyle='--', label=f'Best: {best_mae:.2f}m', linewidth=2)
    ax3.scatter([best_epoch], [best_mae], color='g', s=100, zorder=5, marker='o')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('MAE (meters)', fontsize=12)
    ax3.set_title('Validation MAE', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. Validation RMSE
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, history['val_rmse'], 'r-', linewidth=2)
    best_rmse = min(history['val_rmse'])
    best_epoch_rmse = history['val_rmse'].index(best_rmse) + 1
    ax4.axhline(y=best_rmse, color='g', linestyle='--', label=f'Best: {best_rmse:.2f}m', linewidth=2)
    ax4.scatter([best_epoch_rmse], [best_rmse], color='g', s=100, zorder=5, marker='o')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('RMSE (meters)', fontsize=12)
    ax4.set_title('Validation RMSE', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # 5. Validation R²
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, history['val_r2'], 'purple', linewidth=2)
    best_r2 = max(history['val_r2'])
    best_epoch_r2 = history['val_r2'].index(best_r2) + 1
    ax5.axhline(y=best_r2, color='g', linestyle='--', label=f'Best: {best_r2:.4f}', linewidth=2)
    ax5.scatter([best_epoch_r2], [best_r2], color='g', s=100, zorder=5, marker='o')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('R² Score', fontsize=12)
    ax5.set_title('Validation R²', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # 6. Edge Metrics (F1, IoU, Precision, Recall)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(epochs, history['val_edge_f1'], 'b-', label='F1', linewidth=2)
    ax6.plot(epochs, history['val_edge_iou'], 'r-', label='IoU', linewidth=2)
    ax6.plot(epochs, history['val_edge_precision'], 'g--', label='Precision', linewidth=1.5)
    ax6.plot(epochs, history['val_edge_recall'], 'orange', linestyle='--', label='Recall', linewidth=1.5)
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Score', fontsize=12)
    ax6.set_title('Edge Detection Metrics', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # 7. Learning Rate Schedule
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(epochs, history['learning_rate'], 'orange', linewidth=2)
    ax7.set_xlabel('Epoch', fontsize=12)
    ax7.set_ylabel('Learning Rate', fontsize=12)
    ax7.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax7.set_yscale('log')
    ax7.grid(True, alpha=0.3)
    
    # 8. Height vs Edge Loss (Validation)
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(epochs, history['val_height_loss'], 'g-', label='Height Loss', linewidth=2)
    ax8_twin = ax8.twinx()
    ax8_twin.plot(epochs, history['val_edge_loss'], 'orange', label='Edge Loss', linewidth=2)
    ax8.set_xlabel('Epoch', fontsize=12)
    ax8.set_ylabel('Height Loss', fontsize=12, color='g')
    ax8_twin.set_ylabel('Edge Loss', fontsize=12, color='orange')
    ax8.set_title('Validation Loss Components', fontsize=14, fontweight='bold')
    ax8.tick_params(axis='y', labelcolor='g')
    ax8_twin.tick_params(axis='y', labelcolor='orange')
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary Statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = f"""
    TRAINING SUMMARY
    {'='*40}
    
    Final Epoch: {len(epochs)}
    
    Best Validation Metrics:
    • MAE: {best_mae:.2f}m (epoch {best_epoch})
    • RMSE: {best_rmse:.2f}m (epoch {best_epoch_rmse})
    • R²: {best_r2:.4f} (epoch {best_epoch_r2})
    
    Edge Detection (Best):
    • F1: {max(history['val_edge_f1']):.4f}
    • IoU: {max(history['val_edge_iou']):.4f}
    • Precision: {max(history['val_edge_precision']):.4f}
    • Recall: {max(history['val_edge_recall']):.4f}
    
    Final Learning Rate: {history['learning_rate'][-1]:.2e}
    """
    
    ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes,
             fontsize=11, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Model 4 V3: Comprehensive Training Progress', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Comprehensive training curves saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Model 4 V3: FPN + Multi-Task - Improved Edge Detection')
    parser.add_argument('--data_dir', default='dfc2023_height_dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--save_dir', default='./checkpoints_model4_v3')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', default='height-estimation-model4')
    parser.add_argument('--height_weight', type=float, default=1.0)
    parser.add_argument('--edge_weight', type=float, default=2.0)  # 1.0 -> 2.0
    parser.add_argument('--background_weight', type=float, default=0.3)
    args = parser.parse_args()
    
    print("=" * 80)
    print("MODEL 4 V3: FPN + Multi-Task - IMPROVED EDGE DETECTION")
    print("=" * 80)
    print(f"\n📋 V3 Improvements:")
    print(f"   ✅ Edge extraction: 5x5 kernel, iterations 3+2 (was 3x3, 2+1)")
    print(f"   ✅ Edge weight: {args.edge_weight} (was 1.0)")
    print(f"   ✅ Focal Loss gamma: 3.0 (was 2.0)")
    print(f"   ✅ Edge head: 4-layer deeper architecture")
    print(f"   ✅ Comprehensive training curves")
    print(f"\n📋 Architecture:")
    print(f"   Model: FPN with ResNet50 encoder")
    print(f"   Loss: Height({args.height_weight}) + Edge({args.edge_weight}) + BG({args.background_weight})")
    print(f"   Edge Loss: Focal BCE (alpha=0.25, gamma=3.0)")
    print(f"   Height Loss: MSE + L1 (0.5 each)")
    
    if args.use_wandb:
        wandb.init(project=args.project_name, config=args, name="fpn_multitask_v3_improved")
    
    device = get_device()
    data_dir = Path(args.data_dir)
    
    with open(data_dir / 'height_statistics.json', 'r') as f:
        height_stats = json.load(f)
    
    height_min = height_stats['min']
    height_range = height_stats['max'] - height_stats['min'] + 1e-8
    
    print(f"\n📊 Height Stats: min={height_stats['min']:.2f}m, max={height_stats['max']:.2f}m")
    
    with open(data_dir / 'train_data.json', 'r') as f:
        train_data = json.load(f)
    with open(data_dir / 'val_data.json', 'r') as f:
        val_data = json.load(f)
    
    train_dataset = HeightDatasetMultiTask(train_data, height_stats, get_training_transforms())
    val_dataset = HeightDatasetMultiTask(val_data, height_stats, get_validation_transforms())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = FPNMultiTaskModel(encoder_name="resnet50", encoder_weights="imagenet")
    model.to(device)
    print(f"\n🏗️  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = MultiTaskLossV3(
        height_weight=args.height_weight, 
        edge_weight=args.edge_weight,
        background_weight=args.background_weight
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    best_mae, patience_counter = float('inf'), 0
    
    # Training history for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'train_height_loss': [], 'val_height_loss': [],
        'train_edge_loss': [], 'val_edge_loss': [],
        'train_bg_loss': [], 'val_bg_loss': [],
        'train_mae': [], 'val_mae': [],
        'val_rmse': [], 'val_r2': [],
        'val_edge_iou': [], 'val_edge_f1': [],
        'val_edge_precision': [], 'val_edge_recall': [],
        'learning_rate': []
    }
    
    print(f"\n🚀 Starting training...")
    
    for epoch in range(args.epochs):
        print(f'\n{"="*70}\nEpoch {epoch+1}/{args.epochs}\n{"="*70}')
        
        train_loss, train_h, train_e, train_bg, train_m = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1, height_min, height_range
        )
        val_loss, val_h, val_e, val_bg, val_m = validate_epoch(
            model, val_loader, criterion, device, epoch+1, height_min, height_range
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_height_loss'].append(train_h)
        history['val_height_loss'].append(val_h)
        history['train_edge_loss'].append(train_e)
        history['val_edge_loss'].append(val_e)
        history['train_bg_loss'].append(train_bg)
        history['val_bg_loss'].append(val_bg)
        history['train_mae'].append(train_m['mae'])
        history['val_mae'].append(val_m['mae'])
        history['val_rmse'].append(val_m['rmse'])
        history['val_r2'].append(val_m['r2'])
        history['val_edge_iou'].append(val_m['edge_iou'])
        history['val_edge_f1'].append(val_m['edge_f1'])
        history['val_edge_precision'].append(val_m.get('edge_precision', 0))
        history['val_edge_recall'].append(val_m.get('edge_recall', 0))
        history['learning_rate'].append(current_lr)
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch+1, 
                'train_loss': train_loss, 'val_loss': val_loss,
                'train_height_loss': train_h, 'val_height_loss': val_h,
                'train_edge_loss': train_e, 'val_edge_loss': val_e,
                'train_bg_loss': train_bg, 'val_bg_loss': val_bg,
                'train_mae': train_m['mae'], 'val_mae': val_m['mae'],
                'val_rmse': val_m['rmse'], 'val_r2': val_m['r2'], 
                'val_edge_iou': val_m['edge_iou'],
                'val_edge_f1': val_m['edge_f1'],
                'val_edge_precision': val_m.get('edge_precision', 0),
                'val_edge_recall': val_m.get('edge_recall', 0),
                'learning_rate': current_lr
            })
        
        print(f'\n📈 Train: Loss={train_loss:.4f} (H:{train_h:.4f}, E:{train_e:.4f}, BG:{train_bg:.4f})')
        print(f'   MAE={train_m["mae"]:.2f}m, Edge F1={train_m["edge_f1"]:.3f}')
        print(f'📉 Val: Loss={val_loss:.4f}, MAE={val_m["mae"]:.2f}m, RMSE={val_m["rmse"]:.2f}m, R²={val_m["r2"]:.4f}')
        print(f'   Edge: IoU={val_m["edge_iou"]:.3f}, F1={val_m["edge_f1"]:.3f}, '
              f'Prec={val_m.get("edge_precision", 0):.3f}, Rec={val_m.get("edge_recall", 0):.3f}')
        
        # Visualize every 5 epochs
        if (epoch + 1) % 5 == 0:
            visualize_predictions(model, val_loader, device, height_min, height_range, 
                                save_dir / f'pred_epoch_{epoch+1}.png')
        
        if val_m['mae'] < best_mae and val_m['mae'] > 0:
            best_mae, patience_counter = val_m['mae'], 0
            torch.save({
                'epoch': epoch+1, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mae': best_mae, 
                'val_metrics': val_m, 
                'height_stats': height_stats,
                'config': {
                    'model': 'FPN_MultiTask_V3', 
                    'encoder': 'resnet50',
                    'loss': 'MSE+L1+FocalBCE(gamma=3.0)+Background', 
                    'normalization': 'MinMax',
                    'edge_weight': args.edge_weight,
                    'improvements': '5x5_kernel_3+2_iterations_4layer_edge_head'
                },
                'history': history
            }, save_dir / 'best_model.pth')
            print(f'\n✅ Best model saved! MAE: {best_mae:.2f}m')
            visualize_predictions(model, val_loader, device, height_min, height_range, 
                                save_dir / f'best_pred_epoch_{epoch+1}.png')
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f'\n⏹️  Early stopping at epoch {epoch+1}')
            break
    
    # Plot comprehensive training curves
    print(f"\n📊 Generating comprehensive training curves...")
    plot_comprehensive_training_curves(history, save_dir / 'training_curves_comprehensive.png')
    
    # Save final checkpoint
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_metrics': val_m,
        'best_mae': best_mae,
        'height_stats': height_stats,
        'config': {
            'model': 'FPN_MultiTask_V3',
            'encoder': 'resnet50',
            'loss': 'MSE+L1+FocalBCE(gamma=3.0)+Background',
            'normalization': 'MinMax',
            'edge_weight': args.edge_weight,
            'improvements': '5x5_kernel_3+2_iterations_4layer_edge_head'
        },
        'history': history
    }, save_dir / 'final_model.pth')
    
    print(f'\n{"="*70}')
    print(f'🎉 Training completed!')
    print(f'   Best MAE: {best_mae:.2f}m')
    print(f'   Final Val MAE: {val_m["mae"]:.2f}m')
    print(f'   Final Val RMSE: {val_m["rmse"]:.2f}m')
    print(f'   Final Val R²: {val_m["r2"]:.4f}')
    print(f'   Final Edge F1: {val_m["edge_f1"]:.3f}')
    print(f'{"="*70}\n')
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()