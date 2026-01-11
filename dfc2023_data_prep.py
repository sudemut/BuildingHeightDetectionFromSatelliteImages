#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFC2023 Track 2 Data Preparation for Height Estimation
Convert DFC2023 format to training-ready format
"""
import rasterio
import numpy as np
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import gc

class DFC2023HeightPreparation:
    def __init__(self, dfc_root, output_dir, img_size=(256, 256), use_sar=False):
        self.dfc_root = Path(dfc_root)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.use_sar = use_sar
        
        # Output directories
        self.images_dir = self.output_dir / 'images'
        self.masks_dir = self.output_dir / 'masks'
        self.heights_dir = self.output_dir / 'heights'
        
        for dir_path in [self.images_dir, self.masks_dir, self.heights_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0.0,
            'M2': 0.0,
            'count': 0
        }
    
    def update_statistics(self, heights):
        """Welford's online algorithm for statistics"""
        for h in heights.flatten():
            h = float(h)
            if np.isnan(h) or np.isinf(h) or h <= 0:
                continue
            self.stats['count'] += 1
            delta = h - self.stats['mean']
            self.stats['mean'] += delta / self.stats['count']
            delta2 = h - self.stats['mean']
            self.stats['M2'] += delta * delta2
            self.stats['min'] = min(self.stats['min'], h)
            self.stats['max'] = max(self.stats['max'], h)
    
    def process_sample(self, rgb_file, dsm_file, sar_file, output_id):
        """Process a single DFC2023 sample"""
        try:
            # Load RGB
            with rasterio.open(rgb_file) as src:
                rgb_data = src.read()  # (C, H, W)
                rgb_img = np.transpose(rgb_data, (1, 2, 0))  # (H, W, C)
                
                # Convert to 8-bit if needed
                if rgb_img.dtype != np.uint8:
                    rgb_img = (np.clip(rgb_img, 0, 255)).astype(np.uint8)
            
            # Load DSM (height data)
            with rasterio.open(dsm_file) as src:
                dsm_data = src.read(1).astype(np.float32)
                
                # Clean DSM: remove negative and NaN values
                dsm_data = np.nan_to_num(dsm_data, nan=0.0, posinf=0.0, neginf=0.0)
                dsm_data = np.clip(dsm_data, 0, 300)  # Max 300m
                
                # Create building mask (height > 0)
                building_mask = (dsm_data > 0.5).astype(np.float32)
            
            # Check if there are buildings
            if building_mask.sum() < 10:
                return None
            
            # Update statistics
            building_heights = dsm_data[building_mask > 0]
            if len(building_heights) > 0:
                self.update_statistics(building_heights)
            
            # Resize all data
            rgb_resized = cv2.resize(rgb_img, self.img_size, interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(building_mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            height_resized = cv2.resize(dsm_data, self.img_size, interpolation=cv2.INTER_LINEAR)
            
            # Final cleaning
            height_resized = np.nan_to_num(height_resized, nan=0.0, posinf=0.0, neginf=0.0)
            height_resized = np.clip(height_resized, 0, 300)
            
            # Verify no NaN
            if np.isnan(height_resized).any() or np.isinf(height_resized).any():
                return None
            
            # Save files
            image_path = self.images_dir / f'{output_id:06d}.png'
            mask_path = self.masks_dir / f'{output_id:06d}.png'
            height_path = self.heights_dir / f'{output_id:06d}.npy'
            
            cv2.imwrite(str(image_path), cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_path), (mask_resized * 255).astype(np.uint8))
            np.save(str(height_path), height_resized.astype(np.float32))
            
            # Cleanup
            del rgb_data, rgb_img, dsm_data, building_mask
            del rgb_resized, mask_resized, height_resized
            
            return {
                'image': str(image_path),
                'mask': str(mask_path),
                'height': str(height_path),
                'source': 'dfc2023'
            }
            
        except Exception as e:
            print(f"Error processing {rgb_file.name}: {e}")
            return None
    
    def prepare_dataset(self):
        """Process all DFC2023 training data"""
        print("🚀 DFC2023 Track 2 Data Preparation")
        print("=" * 60)
        
        # Find all RGB files in train directory
        train_rgb_dir = self.dfc_root / 'train' / 'rgb'
        rgb_files = sorted(list(train_rgb_dir.glob("*.tif")))
        
        print(f"Found {len(rgb_files)} RGB training files")
        
        processed_samples = []
        output_id = 0
        
        for rgb_file in tqdm(rgb_files, desc="Processing DFC2023 samples"):
            # Find corresponding DSM and SAR files
            file_id = rgb_file.stem
            dsm_file = self.dfc_root / 'train' / 'dsm' / f'{file_id}.tif'
            sar_file = self.dfc_root / 'train' / 'sar' / f'{file_id}.tif' if self.use_sar else None
            
            if not dsm_file.exists():
                print(f"Warning: No DSM file for {file_id}")
                continue
            
            result = self.process_sample(rgb_file, dsm_file, sar_file, output_id)
            if result is not None:
                processed_samples.append(result)
                output_id += 1
            
            if output_id % 100 == 0:
                gc.collect()
        
        print(f"\n✅ Processed {len(processed_samples)} samples")
        
        # Calculate final statistics
        if self.stats['count'] > 1:
            variance = self.stats['M2'] / (self.stats['count'] - 1)
            std = np.sqrt(variance)
        else:
            std = 1.0
        
        final_stats = {
            'min': 0.0,
            'max': float(self.stats['max']) if self.stats['max'] != float('-inf') else 200.0,
            'mean': float(self.stats['mean']) if self.stats['count'] > 0 else 10.0,
            'std': float(std) if std > 0 else 10.0,
            'samples': int(self.stats['count'])
        }
        
        print("\n📊 Height Statistics:")
        for k, v in final_stats.items():
            print(f"  {k}: {v:.2f}" if k != 'samples' else f"  {k}: {v}")
        
        # Save statistics
        with open(self.output_dir / 'height_statistics.json', 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        # Split into train/val (since we have separate test set)
        train, val = train_test_split(processed_samples, test_size=0.15, random_state=42)
        
        print(f"\n📂 Train={len(train)}, Val={len(val)}")
        
        # Save splits
        for split_name, split_data in [('train', train), ('val', val)]:
            with open(self.output_dir / f'{split_name}_data.json', 'w') as f:
                json.dump(split_data, f, indent=2)
        
        print(f"\n✅ Output: {self.output_dir}")
        return processed_samples

def main():
    preparator = DFC2023HeightPreparation(
        dfc_root="C:\\Users\\Sude\\Desktop\\402\\track2",
        output_dir="dfc2023_height_dataset",
        img_size=(256, 256),
        use_sar=False  # Set True if you want to use SAR data
    )
    
    preparator.prepare_dataset()
    print("\n🎉 DFC2023 data ready for training!")

if __name__ == "__main__":
    main()