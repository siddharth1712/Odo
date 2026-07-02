from nlf.pt.multiperson.depth_prediction import SMPLPredictor

import os
from PIL import Image
from typing import List, Tuple

from odo.image_utils import *
from torchvision import transforms
import smplx

import torch

from tqdm import tqdm
from odo.metrics.image_metrics import MetricCalculation
from torch.utils.data import Sampler

import math

import torch
import torch.nn.functional as F
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from typing import Dict, List

import json

# Global models to avoid reinitialization
_lpips_model = None
_ssim_metric = None
_psnr_metric = None

def get_metrics_models(device='cuda'):
    """Get or initialize metric models"""
    global _lpips_model, _ssim_metric, _psnr_metric
    
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex').to(device)
        _lpips_model.eval()
    
    if _ssim_metric is None:
        _ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    if _psnr_metric is None:
        _psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    return _lpips_model, _ssim_metric, _psnr_metric

def calculate_batch_metrics_vectorized(target_batch: torch.Tensor, pred_batch: torch.Tensor, device: str = 'cuda') -> Dict[str, List[float]]:
    """
    Fully vectorized version for maximum performance.
    
    Args:
        target_batch: Target images tensor [B, C, H, W] in range [0, 1]
        pred_batch: Predicted images tensor [B, C, H, W] in range [0, 1]  
        device: Device to use for computation
        
    Returns:
        Dictionary with lists of metric values
    """
    
    # Ensure tensors are on correct device
    target_batch = target_batch.to(device)
    pred_batch = pred_batch.to(device)
    
    # Ensure same shape
    if target_batch.shape != pred_batch.shape:
        pred_batch = F.interpolate(pred_batch, size=target_batch.shape[2:], mode='bilinear', align_corners=False)
    
    # Get metric models
    lpips_model, ssim_metric, psnr_metric = get_metrics_models(device)
    
    # Calculate SSIM and PSNR for entire batch (if supported)
    # Note: Some torchmetrics versions support batch processing, others don't
    try:
        # Try batch processing first
        ssim_values = ssim_metric(pred_batch, target_batch)
        psnr_values = psnr_metric(pred_batch, target_batch)
        
        # If batch processing returns single value, fall back to per-image
        if ssim_values.dim() == 0:
            ssim_values = [ssim_metric(pred_batch[i:i+1], target_batch[i:i+1]).item() 
                          for i in range(target_batch.shape[0])]
            psnr_values = [psnr_metric(pred_batch[i:i+1], target_batch[i:i+1]).item() 
                          for i in range(target_batch.shape[0])]
        else:
            ssim_values = ssim_values.cpu().tolist()
            psnr_values = psnr_values.cpu().tolist()
            
    except:
        # Fall back to per-image calculation
        ssim_values = [ssim_metric(pred_batch[i:i+1], target_batch[i:i+1]).item() 
                      for i in range(target_batch.shape[0])]
        psnr_values = [psnr_metric(pred_batch[i:i+1], target_batch[i:i+1]).item() 
                      for i in range(target_batch.shape[0])]
    
    # LPIPS calculation
    target_normalized = target_batch * 2.0 - 1.0
    pred_normalized = pred_batch * 2.0 - 1.0
    
    with torch.no_grad():
        lpips_batch = lpips_model(target_normalized, pred_normalized)
        lpips_values = lpips_batch.squeeze().cpu().tolist()
        
        # Handle single image case
        if not isinstance(lpips_values, list):
            lpips_values = [lpips_values]
    
    return {
        'SSIM': ssim_values,
        'PSNR': psnr_values,
        'LPIPS': lpips_values
    }

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, input_folder):
        self.input_folder = input_folder
        
        self.image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        
        gender = "male" if "male" in self.image_files[idx] else "female"
        image = Image.open(os.path.join(self.input_folder, self.image_files[idx]))
        
        width, height = image.size
            
        # Calculate width for each part (divide by 4)
        part_width = width // 4
        
        # Define crop boxes for each part
        crop_boxes = []
        for i in range(4):
            left = i * part_width
            right = left + part_width
            
            # For the last part, include any remaining pixels
            if i == 3:
                right = width
            
            crop_boxes.append((left, 0, right, height))
        
        # Get the 2nd part (index 1) and 4th part (index 3)
        target_pil = image.crop(crop_boxes[1])
        pred_pil = image.crop(crop_boxes[3])
        
        target = self.transform(target_pil)
        pred = self.transform(pred_pil)
        
        result={}
        result["target"] = target
        result["pred"] = pred
        result["gender"] = gender
            
        return result
    
class GenderBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.female_indices = []
        self.male_indices = []
        self.neutral_indices = []
        
        # Use same logic as dataset to ensure consistency
        for i, og_path in enumerate(dataset.image_files):
            if 'female' in og_path.lower() or 'women' in og_path.lower():
                self.female_indices.append(i)
            elif 'male' in og_path.lower() or 'men' in og_path.lower():
                self.male_indices.append(i)
            else:
                self.neutral_indices.append(i)
                
        self.batch_size = batch_size

    def __iter__(self):
        # Create batches for female images first
        for i in range(0, len(self.female_indices), self.batch_size):
            yield self.female_indices[i:i + self.batch_size]
            
        # Then create batches for male images
        for i in range(0, len(self.male_indices), self.batch_size):
            yield self.male_indices[i:i + self.batch_size]
        
        # Then create batches for neutral images
        for i in range(0, len(self.neutral_indices), self.batch_size):
            yield self.neutral_indices[i:i + self.batch_size]
    
    def __len__(self):
        # Calculate number of batches for each gender separately
        female_batches = math.ceil(len(self.female_indices) / self.batch_size) if self.female_indices else 0
        neutral_batches = math.ceil(len(self.neutral_indices) / self.batch_size) if self.neutral_indices else 0
        male_batches = math.ceil(len(self.male_indices) / self.batch_size) if self.male_indices else 0
        
        print(f"Female batches: {female_batches}")
        print(f"Neutral batches: {neutral_batches}")
        print(f"Male batches: {male_batches}")
        
        return female_batches + neutral_batches + male_batches

def get_t_pose_vertices(betas,gender):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    betas = betas.to(device)
    model = smplx.SMPL("/spiral_hdd_2/workspace/siddharth/nlf/data/body_models/smpl", use_pca=False, batch_size=betas.shape[0],gender=gender).to(device)
    output = model(betas=betas,global_orient=torch.zeros((betas.shape[0],3)).to(device),body_pose=torch.zeros((betas.shape[0],23*3)).to(device),transl=torch.zeros((betas.shape[0],3)).to(device))
    
    vertices = output.vertices
    return vertices

def main():
    input_folder_root = "/spiral_hdd_2/workspace/siddharth/IDM-VTON/FINAL_INFERENCE_SEEDREAM"
    input_folder = os.path.join(input_folder_root, "images")
    batch_size = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_estimator = SMPLPredictor(model_path="/spiral_hdd_2/workspace/siddharth/nlf/models/pt/nlf_l_crop.pt",batch_size=batch_size,resolution=(768,1024),device=device)
    metric_calculator = MetricCalculation(device)
    
    dataset = SimpleDataset(input_folder)
    sampler = GenderBatchSampler(dataset, batch_size)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    
    all_metrics = {}
    all_metrics["PVET"] = []
    all_metrics["LPIPS"] = []
    all_metrics["PSNR"] = []
    all_metrics["SSIM"] = []
    
    for batch in tqdm(dataloader, desc="Processing images"):
        
        gender = batch["gender"][0]
        
        metrics = calculate_batch_metrics_vectorized(batch["pred"], batch["target"], device=device)
        for metric, value in metrics.items():
            if value is not None:
                all_metrics[metric] += value
            else:
                print(f"{metric}: Failed to calculate")
                
        images_tensor = scale(batch["pred"])
        images_tensor_betas = smpl_estimator.get_betas(images_tensor,gender=gender)
        image_tensor_vertices_t_pose = get_t_pose_vertices(images_tensor_betas,gender)
        
        edited_tensor = scale(batch["target"])
        edited_tensor_betas = smpl_estimator.get_betas(edited_tensor,gender=gender)
        edited_tensor_vertices_t_pose = get_t_pose_vertices(edited_tensor_betas,gender)
        
        all_metrics["PVET"] += metric_calculator.calculate_pve(image_tensor_vertices_t_pose, edited_tensor_vertices_t_pose)
    
    # Calculate averages for metrics summary
    metrics_summary = {}
    print("\n===== Evaluation Metrics =====")
    for metric_name, values in all_metrics.items():
        if values:  # Check if the list is not empty
            avg_value = sum(values) / len(values)
            metrics_summary[f"{metric_name}_avg"] = float(avg_value)
            metrics_summary[f"{metric_name}_count"] = len(values)
            print(f"{metric_name}: {avg_value:.4f} (avg of {len(values)} samples)")

    # Save metrics to JSON file
    output_path = os.path.join(input_folder_root, "metrics.json")
    with open(output_path, "w") as f:
        json.dump({
            "summary": metrics_summary
        }, f, indent=4)
    
    print(f"Metrics saved to {output_path}")
            
if __name__ == "__main__":  
    main()
    
    