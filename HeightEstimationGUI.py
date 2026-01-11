#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building Height Estimation GUI - Multi-Model Comparison
Supports all 5 models with different architectures and configurations
"""
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import json
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
from datetime import datetime


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class DualHeadEfficientNet(nn.Module):
    """
    Dual-Head Model: Height Regression + Building Segmentation
    Model 3 ile uyumlu - 4 çıktı döndürür
    
    Final output = sigmoid(Height) * sigmoid(Segmentation)
    """
    def __init__(self, encoder_name='efficientnet-b4', encoder_weights='imagenet'):
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


class FPNMultiTaskModel(nn.Module):
    """
    Multi-Task FPN for Model 4: Height + Edges
    Model 4 ile uyumlu - 2 çıktı döndürür (segmentation yok)
    """
    def __init__(self, encoder_name='resnet50', encoder_weights='imagenet'):
        super().__init__()
        
        self.fpn = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=64,
            activation=None
        )
        
        # Height head
        self.height_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        
        # Edge head
        self.edge_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
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


class OrdinalSegFormer(nn.Module):
    """SegFormer with Ordinal Regression for Model 5"""
    def __init__(self, encoder_name='mit_b2', num_bins=50):
        super().__init__()
        self.num_bins = num_bins
        
        # Segmentation head for building mask
        self.seg_model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )
        
        # Ordinal regression head
        self.ordinal_model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_bins,
            activation=None
        )
    
    def forward(self, x):
        seg = self.seg_model(x)
        ordinal = self.ordinal_model(x)
        return ordinal, seg


# =============================================================================
# MODEL LOADER
# =============================================================================

class ModelLoader:
    """Load different model architectures based on checkpoint config"""
    
    @staticmethod
    def get_device():
        """Get the best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    @staticmethod
    def load_model(checkpoint_path, device):
        """
        Load model from checkpoint and detect architecture automatically
        
        Returns:
            tuple: (model, config, height_stats, val_metrics, epoch)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        config = checkpoint.get('config', {})
        height_stats = checkpoint.get('height_stats', {})
        val_metrics = checkpoint.get('val_metrics', {})
        epoch = checkpoint.get('epoch', 0)
        
        # Detect model architecture
        model_type = config.get('model', 'DeepLabV3Plus')
        encoder = config.get('encoder', 'resnet50')
        loss_type = config.get('loss', 'ScaleInvariant')
        norm_type = config.get('normalization', 'LogScale')
        
        print(f"\n🔍 Detected Model Configuration:")
        print(f"   Architecture: {model_type}")
        print(f"   Encoder: {encoder}")
        print(f"   Loss: {loss_type}")
        print(f"   Normalization: {norm_type}")
        
        # Create model based on type
        if model_type == 'DeepLabV3Plus':
            # Model 1: DeepLabV3+
            model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None
            )
        
        elif model_type == 'Unet' and loss_type == 'L1_Gradient':
            # Model 2: HRNet U-Net
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None
            )
        
        elif model_type == 'DualHeadUNet' or model_type == 'DualHeadUnet':
            # Model 3: EfficientNet-B4 Dual-Head
            model = DualHeadEfficientNet(
                encoder_name=encoder,
                encoder_weights=None
            )
        
        elif model_type in ['MultiTaskFPN', 'FPNMultiTaskModel', 'FPN_MultiTask_V2']:
            # Model 4: FPN Multi-Task
            model = FPNMultiTaskModel(
                encoder_name=encoder,
                encoder_weights=None
            )
        
        elif model_type == 'OrdinalSegFormer':
            # Model 5: SegFormer Ordinal
            num_bins = checkpoint.get('num_bins', 50)
            model = OrdinalSegFormer(
                encoder_name='mit_b2',
                num_bins=num_bins
            )
        
        else:
            # Default: Standard U-Net
            print(f"⚠️  Unknown model type '{model_type}', using standard U-Net")
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None
            )
        
        # Load weights
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model weights loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: {e}")
            print("   Attempting to load weights with strict=False")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        model.to(device)
        model.eval()
        
        return model, config, height_stats, val_metrics, epoch
    
    @staticmethod
    def denormalize_prediction(pred, config, height_stats):
        """
        Denormalize prediction based on normalization type
        
        Args:
            pred: normalized prediction (numpy array)
            config: model configuration
            height_stats: height statistics dict
            
        Returns:
            denormalized prediction in meters
        """
        print(f"DEBUG denorm: pred type={type(pred)}, dtype={pred.dtype if hasattr(pred, 'dtype') else 'N/A'}")
        print(f"DEBUG denorm: pred shape={pred.shape if hasattr(pred, 'shape') else 'N/A'}")
        
        # Ensure pred is float32 numpy array
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred, dtype=np.float32)
        else:
            pred = pred.astype(np.float32)
        
        norm_type = config.get('normalization', 'LogScale')
        print(f"DEBUG denorm: normalization type = {norm_type}")
        
        if norm_type == 'LogScale':
            # Model 1: Log-scale normalization
            log_max = np.log(height_stats['max'] + 1)
            height = np.exp(pred * log_max) - 1
        
        elif norm_type == 'ZScore':
            # Model 2: Z-score normalization
            height = pred * height_stats['std'] + height_stats['mean']
        
        elif norm_type == 'Percentile':
            # Model 3: Percentile normalization
            p1 = max(0, height_stats['mean'] - 2 * height_stats['std'])
            p99 = min(height_stats['max'], height_stats['mean'] + 2 * height_stats['std'])
            height = pred * (p99 - p1) + p1
        
        elif norm_type == 'MinMax':
            # Model 4: Min-max normalization
            print(f"DEBUG denorm MinMax: height_stats = {height_stats}")
            height_min = float(height_stats.get('min', 0))
            height_max = float(height_stats['max'])
            print(f"DEBUG denorm MinMax: min={height_min}, max={height_max}")
            height_range = height_max - height_min
            print(f"DEBUG denorm MinMax: range={height_range}")
            height = pred * height_range + height_min
            print(f"DEBUG denorm MinMax: result dtype={height.dtype}")
        
        elif norm_type == 'Ordinal':
            # Model 5: Already denormalized in inference
            height = pred
        
        else:
            # Default: assume min-max
            height_min = float(height_stats.get('min', 0))
            height_max = float(height_stats['max'])
            height_range = height_max - height_min
            height = pred * height_range + height_min
        
        result = np.clip(height, 0, height_stats['max'] * 1.5).astype(np.float32)
        print(f"DEBUG denorm: final result dtype={result.dtype}")
        return result


# =============================================================================
# INFERENCE
# =============================================================================

class ModelInference:
    """Handle inference for different model types"""
    
    @staticmethod
    def get_transform():
        """Get validation transform"""
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    @staticmethod
    def predict(model, image, config, height_stats, device):
        """
        Run inference on image
        
        Args:
            model: loaded model
            image: input image (numpy array, RGB)
            config: model configuration
            height_stats: height statistics
            device: torch device
            
        Returns:
            dict with 'height' and optionally 'mask', 'edge'
        """
        try:
            model.eval()
            
            # Preprocess
            transform = ModelInference.get_transform()
            transformed = transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0).to(device)
            
            # Forward pass
            with torch.no_grad():
                model_type = config.get('model', 'DeepLabV3Plus')
                
                if model_type in ['DeepLabV3Plus', 'Unet']:
                    # Model 1 & 2: Single output
                    output = model(image_tensor)
                    height_pred = torch.sigmoid(output).cpu().numpy()[0, 0].astype(np.float32)
                    result = {'height': height_pred}
                
                elif model_type in ['DualHeadUNet', 'DualHeadUnet']:
                    # Model 3: Height + Segmentation (4 outputs)
                    # Returns: height_masked, height_sigmoid, seg_prob, seg_logits
                    height_masked, height_sigmoid, seg_prob, seg_logits = model(image_tensor)
                    
                    # Use the masked height (already multiplied with segmentation)
                    height_pred = height_masked.cpu().numpy()[0, 0].astype(np.float32)
                    mask_pred = seg_prob.cpu().numpy()[0, 0].astype(np.float32)
                    
                    result = {'height': height_pred, 'mask': mask_pred}
                
                elif model_type in ['MultiTaskFPN', 'FPNMultiTaskModel', 'FPN_MultiTask_V2']:
                    # Model 4: Height + Edges (2 outputs, sigmoid already applied)
                    height_out, edge_out = model(image_tensor)
                    
                    # Sigmoid already applied in forward, no need to apply again
                    # Ensure proper dtype
                    height_pred = height_out.cpu().numpy()[0, 0].astype(np.float32)
                    edge_pred = edge_out.cpu().numpy()[0, 0].astype(np.float32)
                    
                    # Model 4 doesn't have explicit segmentation, only edges
                    # We'll store edge as mask for visualization purposes
                    # But we won't apply it to mask the height (model already learned to suppress background)
                    result = {'height': height_pred, 'edge': edge_pred, 'mask': None}
                
                elif model_type == 'OrdinalSegFormer':
                    # Model 5: Ordinal regression
                    ordinal_out, seg_out = model(image_tensor)
                    ordinal_probs = torch.sigmoid(ordinal_out).cpu().numpy()[0]
                    mask_pred = torch.sigmoid(seg_out).cpu().numpy()[0, 0].astype(np.float32)
                    
                    # Convert ordinal to height
                    num_bins = ordinal_probs.shape[0]
                    bin_width = float(height_stats['max']) / num_bins
                    height_pred = (ordinal_probs.sum(axis=0) * bin_width).astype(np.float32)
                    
                    result = {'height': height_pred, 'mask': mask_pred}
                
                else:
                    # Unknown: assume single output
                    output = model(image_tensor)
                    height_pred = torch.sigmoid(output).cpu().numpy()[0, 0].astype(np.float32)
                    result = {'height': height_pred}
            
            # Denormalize height
            print(f"DEBUG: Before denorm - height shape: {result['height'].shape}, dtype: {result['height'].dtype}")
            print(f"DEBUG: height_stats keys: {height_stats.keys()}")
            print(f"DEBUG: config: {config}")
            
            result['height'] = ModelLoader.denormalize_prediction(
                result['height'], config, height_stats
            )
            
            print(f"DEBUG: After denorm - height shape: {result['height'].shape}, dtype: {result['height'].dtype}")
            
            # Apply mask if available (but not for Model 4 which has background suppression built-in)
            if 'mask' in result and result['mask'] is not None:
                mask_binary = (result['mask'] > 0.5).astype(np.float32)
                result['height'] = result['height'] * mask_binary
            
            return result
            
        except Exception as e:
            print(f"ERROR in predict: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise


# =============================================================================
# MODEL INSTANCE
# =============================================================================

class ModelInstance:
    """Container for a loaded model and its metadata"""
    def __init__(self, name, checkpoint_path, model, device, height_stats, config, val_metrics, epoch):
        self.name = name
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.device = device
        self.height_stats = height_stats
        self.config = config
        self.val_metrics = val_metrics
        self.epoch = epoch
        self.prediction = None
        self.color = None
        
    def get_description(self):
        """Get human-readable model description"""
        model_type = self.config.get('model', 'Unknown')
        encoder = self.config.get('encoder', 'Unknown')
        loss = self.config.get('loss', 'Unknown')
        norm = self.config.get('normalization', 'Unknown')
        
        mae = self.val_metrics.get('mae', 0)
        rmse = self.val_metrics.get('rmse', 0)
        
        desc = f"{model_type} ({encoder})\n"
        desc += f"Loss: {loss}, Norm: {norm}\n"
        desc += f"Val MAE: {mae:.2f}m, RMSE: {rmse:.2f}m\n"
        desc += f"Epoch: {self.epoch}"
        
        return desc


# =============================================================================
# GUI
# =============================================================================

class HeightEstimationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Building Height Estimation - Multi-Model Comparison (All 5 Models)")
        self.root.geometry("1800x1000")
        
        # Variables
        self.models = {}  # Dictionary of ModelInstance objects
        self.current_image = None
        self.image_path = None
        self.device = ModelLoader.get_device()
        
        # Model colors for visualization
        self.model_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        
        # Setup UI
        self.setup_ui()
        
        print(f"\n🖥️  GUI Initialized")
        print(f"   Device: {self.device}")
        print(f"   Ready to load models!")
        
    def setup_ui(self):
        """Setup the GUI layout"""
        # Main container with notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.main_tab = ttk.Frame(self.notebook)
        self.analytics_tab = ttk.Frame(self.notebook)
        self.comparison_tab = ttk.Frame(self.notebook)
        self.hyperparams_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.main_tab, text="  Prediction  ")
        self.notebook.add(self.analytics_tab, text="  Analytics  ")
        self.notebook.add(self.comparison_tab, text="  Comparison  ")
        self.notebook.add(self.hyperparams_tab, text="  Model Info  ")
        
        # Setup each tab
        self.setup_main_tab()
        self.setup_analytics_tab()
        self.setup_comparison_tab()
        self.setup_hyperparams_tab()
        
        # Status bar (common)
        self.status_var = tk.StringVar(value="Ready. Please load checkpoint(s) and image.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, 
                              anchor=tk.W, font=('Arial', 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
    
    def setup_main_tab(self):
        """Setup main prediction tab"""
        main_frame = ttk.Frame(self.main_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # ===== Model Selection =====
        model_frame = ttk.LabelFrame(main_frame, text="Model Management", padding="10")
        model_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(0, weight=1)
        
        # Model list with scrollbar
        list_frame = ttk.Frame(model_frame)
        list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.model_listbox = tk.Listbox(list_frame, height=4, selectmode=tk.SINGLE, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.model_listbox.yview)
        self.model_listbox.config(yscrollcommand=scrollbar.set)
        
        self.model_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Model buttons
        btn_frame = ttk.Frame(model_frame)
        btn_frame.grid(row=1, column=0, pady=5)
        
        ttk.Button(btn_frame, text="➕ Add Model", command=self.add_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🗑️ Remove Selected", command=self.remove_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📊 View Details", command=self.view_model_details).pack(side=tk.LEFT, padx=2)
        
        # ===== Image Selection =====
        file_frame = ttk.LabelFrame(main_frame, text="Image Selection", padding="10")
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="Input Image:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.image_label = ttk.Label(file_frame, text="No image selected", foreground="gray")
        self.image_label.grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Button(file_frame, text="Browse", command=self.load_image).grid(row=0, column=2, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        self.predict_button = ttk.Button(button_frame, text="🚀 Predict All Models", 
                                        command=self.predict_all, state=tk.DISABLED)
        self.predict_button.grid(row=0, column=0, padx=5)
        
        self.export_button = ttk.Button(button_frame, text="💾 Export Results", 
                                       command=self.export_results, state=tk.DISABLED)
        self.export_button.grid(row=0, column=1, padx=5)
        
        ttk.Button(button_frame, text="🗑️ Clear", command=self.clear_results).grid(row=0, column=2, padx=5)
        
        # ===== Results Visualization =====
        viz_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        viz_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create scrollable canvas for multiple model results
        canvas_frame = ttk.Frame(viz_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_canvas = tk.Canvas(canvas_frame, bg='white')
        scrollbar_y = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.results_canvas.yview)
        scrollbar_x = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.results_canvas.xview)
        
        self.results_canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.results_frame = ttk.Frame(self.results_canvas)
        self.results_canvas.create_window((0, 0), window=self.results_frame, anchor='nw')
        
        self.results_frame.bind('<Configure>', lambda e: self.results_canvas.configure(
            scrollregion=self.results_canvas.bbox('all')))
    
    def setup_analytics_tab(self):
        """Setup analytics tab"""
        analytics_frame = ttk.Frame(self.analytics_tab, padding="10")
        analytics_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(analytics_frame, text="📊 Statistical Analysis", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Create matplotlib figure for analytics
        self.analytics_fig, self.analytics_axes = plt.subplots(2, 2, figsize=(12, 8))
        self.analytics_fig.tight_layout(pad=3.0)
        
        canvas = FigureCanvasTkAgg(self.analytics_fig, analytics_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.analytics_canvas = canvas
    
    def setup_comparison_tab(self):
        """Setup comparison tab"""
        comparison_frame = ttk.Frame(self.comparison_tab, padding="10")
        comparison_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(comparison_frame, text="🔍 Model Comparison", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Create matplotlib figure for comparison
        self.comparison_fig, self.comparison_axes = plt.subplots(1, 3, figsize=(15, 5))
        self.comparison_fig.tight_layout(pad=3.0)
        
        canvas = FigureCanvasTkAgg(self.comparison_fig, comparison_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.comparison_canvas = canvas
    
    def setup_hyperparams_tab(self):
        """Setup hyperparameters/model info tab"""
        info_frame = ttk.Frame(self.hyperparams_tab, padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(info_frame, text="ℹ️ Model Information", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Text widget to display model details
        text_frame = ttk.Frame(info_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.info_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, 
                                                   font=('Courier', 10), height=30)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Refresh button
        ttk.Button(info_frame, text="🔄 Refresh Info", 
                  command=self.refresh_model_info).pack(pady=5)
    
    # =========================================================================
    # MODEL MANAGEMENT
    # =========================================================================
    
    def add_model(self):
        """Add a new model checkpoint"""
        file_path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pth *.pt"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.status_var.set(f"Loading model from {Path(file_path).name}...")
            self.root.update()
            
            # Load model
            model, config, height_stats, val_metrics, epoch = ModelLoader.load_model(
                file_path, self.device
            )
            
            # Create model name
            model_name = f"Model_{len(self.models)+1}"
            
            # Create ModelInstance
            model_instance = ModelInstance(
                name=model_name,
                checkpoint_path=file_path,
                model=model,
                device=self.device,
                height_stats=height_stats,
                config=config,
                val_metrics=val_metrics,
                epoch=epoch
            )
            
            # Assign color
            color_idx = len(self.models) % len(self.model_colors)
            model_instance.color = self.model_colors[color_idx]
            
            # Add to dictionary
            self.models[model_name] = model_instance
            
            # Update listbox
            display_text = f"{model_name}: {config.get('model', 'Unknown')} - MAE: {val_metrics.get('mae', 0):.2f}m"
            self.model_listbox.insert(tk.END, display_text)
            
            self.status_var.set(f"✅ Model loaded successfully: {model_name}")
            
            # Refresh info tab
            self.refresh_model_info()
            
            # Enable predict if image loaded
            if self.current_image is not None:
                self.predict_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.status_var.set("❌ Failed to load model")
    
    def remove_model(self):
        """Remove selected model"""
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a model to remove")
            return
        
        idx = selection[0]
        model_names = list(self.models.keys())
        model_name = model_names[idx]
        
        # Remove from dictionary
        del self.models[model_name]
        
        # Remove from listbox
        self.model_listbox.delete(idx)
        
        self.status_var.set(f"🗑️ Removed {model_name}")
        
        # Refresh info
        self.refresh_model_info()
        
        # Disable predict if no models
        if len(self.models) == 0:
            self.predict_button.config(state=tk.DISABLED)
    
    def view_model_details(self):
        """Show detailed info about selected model"""
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a model to view")
            return
        
        idx = selection[0]
        model_names = list(self.models.keys())
        model_name = model_names[idx]
        model_instance = self.models[model_name]
        
        # Create detail window
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Model Details: {model_name}")
        detail_window.geometry("600x400")
        
        text = scrolledtext.ScrolledText(detail_window, wrap=tk.WORD, font=('Courier', 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Format details
        details = f"{'='*60}\n"
        details += f"MODEL: {model_name}\n"
        details += f"{'='*60}\n\n"
        
        details += f"📁 Checkpoint Path:\n   {model_instance.checkpoint_path}\n\n"
        
        details += f"🏗️  Architecture:\n"
        for key, value in model_instance.config.items():
            details += f"   {key}: {value}\n"
        details += "\n"
        
        details += f"📊 Height Statistics:\n"
        for key, value in model_instance.height_stats.items():
            if isinstance(value, float):
                details += f"   {key}: {value:.2f}\n"
            else:
                details += f"   {key}: {value}\n"
        details += "\n"
        
        details += f"📈 Validation Metrics:\n"
        for key, value in model_instance.val_metrics.items():
            if isinstance(value, float):
                details += f"   {key}: {value:.4f}\n"
            else:
                details += f"   {key}: {value}\n"
        details += "\n"
        
        details += f"🔢 Training Epoch: {model_instance.epoch}\n"
        
        text.insert('1.0', details)
        text.config(state=tk.DISABLED)
    
    def refresh_model_info(self):
        """Refresh model information in the info tab"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        
        if len(self.models) == 0:
            self.info_text.insert('1.0', "No models loaded.\n\nPlease add models using the 'Add Model' button in the Prediction tab.")
        else:
            info = f"{'='*70}\n"
            info += f"LOADED MODELS ({len(self.models)} total)\n"
            info += f"{'='*70}\n\n"
            
            for i, (name, model_instance) in enumerate(self.models.items(), 1):
                info += f"{i}. {name}\n"
                info += f"   {'─'*66}\n"
                info += f"   Architecture: {model_instance.config.get('model', 'Unknown')}\n"
                info += f"   Encoder: {model_instance.config.get('encoder', 'Unknown')}\n"
                info += f"   Loss: {model_instance.config.get('loss', 'Unknown')}\n"
                info += f"   Normalization: {model_instance.config.get('normalization', 'Unknown')}\n"
                info += f"   \n"
                info += f"   Performance:\n"
                info += f"      MAE: {model_instance.val_metrics.get('mae', 0):.2f} m\n"
                info += f"      RMSE: {model_instance.val_metrics.get('rmse', 0):.2f} m\n"
                info += f"      R²: {model_instance.val_metrics.get('r2', 0):.4f}\n"
                info += f"   \n"
                info += f"   Height Range: {model_instance.height_stats.get('min', 0):.2f}m - "
                info += f"{model_instance.height_stats['max']:.2f}m\n"
                info += f"   Training Epoch: {model_instance.epoch}\n"
                info += f"\n"
            
            self.info_text.insert('1.0', info)
        
        self.info_text.config(state=tk.DISABLED)
    
    # =========================================================================
    # IMAGE HANDLING
    # =========================================================================
    
    def load_image(self):
        """Load input image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.tif *.tiff"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Load image
            self.current_image = cv2.imread(file_path)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.image_path = file_path
            
            # Update label
            self.image_label.config(text=Path(file_path).name, foreground="black")
            
            self.status_var.set(f"✅ Image loaded: {Path(file_path).name}")
            
            # Enable predict if models loaded
            if len(self.models) > 0:
                self.predict_button.config(state=tk.NORMAL)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            self.status_var.set("❌ Failed to load image")
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    def predict_all(self):
        """Run prediction on all loaded models"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if len(self.models) == 0:
            messagebox.showwarning("Warning", "Please load at least one model")
            return
        
        self.status_var.set("🚀 Running predictions...")
        self.root.update()
        
        try:
            # Clear previous results
            for widget in self.results_frame.winfo_children():
                widget.destroy()
            
            # Run prediction for each model
            for i, (name, model_instance) in enumerate(self.models.items()):
                self.status_var.set(f"🚀 Predicting with {name}...")
                self.root.update()
                
                # Run inference
                prediction = ModelInference.predict(
                    model_instance.model,
                    self.current_image,
                    model_instance.config,
                    model_instance.height_stats,
                    self.device
                )
                
                model_instance.prediction = prediction
                
                # Visualize result
                self.visualize_prediction(model_instance, i)
            
            # Enable export
            self.export_button.config(state=tk.NORMAL)
            
            # Update analytics and comparison
            self.update_analytics()
            self.update_comparison()
            
            self.status_var.set(f"✅ Predictions completed for {len(self.models)} models")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
            self.status_var.set("❌ Prediction failed")
            import traceback
            traceback.print_exc()
    
    def visualize_prediction(self, model_instance, index):
        """Visualize prediction result for one model"""
        # Create frame for this model
        model_frame = ttk.LabelFrame(self.results_frame, text=model_instance.name, padding="5")
        model_frame.grid(row=index, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Input image
        axes[0].imshow(self.current_image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Height prediction
        height_pred = model_instance.prediction['height']
        im = axes[1].imshow(height_pred, cmap='viridis', vmin=0, vmax=height_pred.max())
        axes[1].set_title(f'Height Map (Max: {height_pred.max():.1f}m)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # 3D visualization or mask or edge
        if 'mask' in model_instance.prediction and model_instance.prediction['mask'] is not None:
            # Display segmentation mask (Model 3 and Model 5)
            mask_pred = model_instance.prediction['mask']
            axes[2].imshow(mask_pred, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Building Mask')
            axes[2].axis('off')
        elif 'edge' in model_instance.prediction and model_instance.prediction['edge'] is not None:
            # Display edge prediction (Model 4)
            edge_pred = model_instance.prediction['edge']
            im = axes[2].imshow(edge_pred, cmap='hot', vmin=0, vmax=1)
            axes[2].set_title('Edge Detection')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046)
        else:
            # Show histogram for models without mask/edge
            axes[2].hist(height_pred[height_pred > 0].flatten(), bins=50, color=model_instance.color, alpha=0.7)
            axes[2].set_title('Height Distribution')
            axes[2].set_xlabel('Height (m)')
            axes[2].set_ylabel('Frequency')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, model_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add stats
        stats_text = f"Config: {model_instance.config.get('model', 'Unknown')} | "
        stats_text += f"Val MAE: {model_instance.val_metrics.get('mae', 0):.2f}m | "
        stats_text += f"Pred Max: {height_pred.max():.2f}m | "
        stats_text += f"Pred Mean: {height_pred[height_pred > 0].mean():.2f}m"
        
        stats_label = ttk.Label(model_frame, text=stats_text, foreground=model_instance.color, 
                               font=('Arial', 9, 'bold'))
        stats_label.pack(pady=2)
    
    def update_analytics(self):
        """Update analytics tab with statistical analysis"""
        if len(self.models) == 0 or not any(m.prediction for m in self.models.values()):
            return
        
        # Clear previous plots
        for ax in self.analytics_axes.flat:
            ax.clear()
        
        # Collect predictions
        predictions = []
        model_names = []
        for name, model in self.models.items():
            if model.prediction:
                predictions.append(model.prediction['height'])
                model_names.append(name)
        
        if len(predictions) == 0:
            return
        
        # 1. Height distributions
        ax = self.analytics_axes[0, 0]
        for i, (pred, name, model) in enumerate(zip(predictions, model_names, self.models.values())):
            heights = pred[pred > 0].flatten()
            ax.hist(heights, bins=30, alpha=0.5, label=name, color=model.color)
        ax.set_title('Height Distributions')
        ax.set_xlabel('Height (m)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Box plots
        ax = self.analytics_axes[0, 1]
        heights_data = [pred[pred > 0].flatten() for pred in predictions]
        box = ax.boxplot(heights_data, labels=model_names, patch_artist=True)
        for patch, model in zip(box['boxes'], self.models.values()):
            patch.set_facecolor(model.color)
        ax.set_title('Height Statistics (Box Plot)')
        ax.set_ylabel('Height (m)')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Model performance comparison
        ax = self.analytics_axes[1, 0]
        mae_values = [m.val_metrics.get('mae', 0) for m in self.models.values()]
        rmse_values = [m.val_metrics.get('rmse', 0) for m in self.models.values()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mae_values, width, label='MAE', 
                      color=[m.color for m in self.models.values()], alpha=0.7)
        bars2 = ax.bar(x + width/2, rmse_values, width, label='RMSE', 
                      color=[m.color for m in self.models.values()], alpha=0.4)
        
        ax.set_title('Validation Performance')
        ax.set_ylabel('Error (m)')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Prediction statistics
        ax = self.analytics_axes[1, 1]
        stats = []
        for pred in predictions:
            heights = pred[pred > 0].flatten()
            stats.append({
                'max': heights.max(),
                'mean': heights.mean(),
                'std': heights.std()
            })
        
        x = np.arange(len(model_names))
        width = 0.25
        
        ax.bar(x - width, [s['max'] for s in stats], width, label='Max', alpha=0.7)
        ax.bar(x, [s['mean'] for s in stats], width, label='Mean', alpha=0.7)
        ax.bar(x + width, [s['std'] for s in stats], width, label='Std', alpha=0.7)
        
        ax.set_title('Prediction Statistics')
        ax.set_ylabel('Height (m)')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        self.analytics_fig.tight_layout()
        self.analytics_canvas.draw()
    
    def update_comparison(self):
        """Update comparison tab with side-by-side comparison"""
        if len(self.models) == 0 or not any(m.prediction for m in self.models.values()):
            return
        
        # Clear previous plots
        for ax in self.comparison_axes.flat:
            ax.clear()
        
        # Input image
        self.comparison_axes[0].imshow(self.current_image)
        self.comparison_axes[0].set_title('Input Image')
        self.comparison_axes[0].axis('off')
        
        # Overlay all predictions
        ax = self.comparison_axes[1]
        ax.imshow(self.current_image, alpha=0.3)
        
        for name, model in self.models.items():
            if model.prediction:
                height_pred = model.prediction['height']
                # Create contour overlay
                contour = ax.contour(height_pred, levels=5, colors=[model.color], 
                                    linewidths=2, alpha=0.7)
                ax.clabel(contour, inline=True, fontsize=8)
        
        ax.set_title('All Predictions Overlay')
        ax.axis('off')
        
        # Difference map (if multiple models)
        if len(self.models) >= 2:
            ax = self.comparison_axes[2]
            preds = [m.prediction['height'] for m in self.models.values() if m.prediction]
            
            if len(preds) >= 2:
                diff = np.abs(preds[0] - preds[1])
                im = ax.imshow(diff, cmap='hot')
                ax.set_title(f'Difference: {list(self.models.keys())[0]} vs {list(self.models.keys())[1]}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        self.comparison_fig.tight_layout()
        self.comparison_canvas.draw()
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def export_results(self):
        """Export prediction results"""
        if not any(m.prediction for m in self.models.values()):
            messagebox.showwarning("Warning", "No predictions to export")
            return
        
        # Ask for directory
        save_dir = filedialog.askdirectory(title="Select Output Directory")
        if not save_dir:
            return
        
        save_dir = Path(save_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Export each model's prediction
            for name, model in self.models.items():
                if model.prediction:
                    # Save height map as numpy
                    np.save(save_dir / f"{name}_height_{timestamp}.npy", 
                           model.prediction['height'])
                    
                    # Save as image
                    height_norm = model.prediction['height'] / model.prediction['height'].max()
                    height_img = (height_norm * 255).astype(np.uint8)
                    height_img = cv2.applyColorMap(height_img, cv2.COLORMAP_VIRIDIS)
                    cv2.imwrite(str(save_dir / f"{name}_height_{timestamp}.png"), height_img)
            
            # Export summary
            summary = {
                'timestamp': timestamp,
                'input_image': str(self.image_path),
                'models': {}
            }
            
            for name, model in self.models.items():
                if model.prediction:
                    summary['models'][name] = {
                        'config': model.config,
                        'val_metrics': model.val_metrics,
                        'prediction_stats': {
                            'max': float(model.prediction['height'].max()),
                            'mean': float(model.prediction['height'][model.prediction['height'] > 0].mean()),
                            'std': float(model.prediction['height'][model.prediction['height'] > 0].std())
                        }
                    }
            
            with open(save_dir / f"summary_{timestamp}.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            messagebox.showinfo("Success", f"Results exported to:\n{save_dir}")
            self.status_var.set(f"✅ Results exported successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export:\n{str(e)}")
    
    def clear_results(self):
        """Clear all results"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        for model in self.models.values():
            model.prediction = None
        
        for ax in self.analytics_axes.flat:
            ax.clear()
        self.analytics_canvas.draw()
        
        for ax in self.comparison_axes.flat:
            ax.clear()
        self.comparison_canvas.draw()
        
        self.export_button.config(state=tk.DISABLED)
        self.status_var.set("🗑️ Results cleared")


# =============================================================================
# MAIN
# =============================================================================

def main():
    root = tk.Tk()
    app = HeightEstimationGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()