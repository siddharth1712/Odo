import torch
import numpy as np
import lpips
from skimage.metrics import structural_similarity
import torch.nn.functional as F
from torchvision import transforms

class MetricCalculation():
    def __init__(self,device):
        self.lpips = lpips.LPIPS(net='alex').to(device)

    def calculate_lpips(self, predicted_image, target_image):
        """
        Calculate LPIPS (Learned Perceptual Image Patch Similarity)
        Inputs should be in range [0, 1] and will be normalized to [-1, 1]
        """
        # Ensure inputs are on the same device
        device = predicted_image.device
        target_image = target_image.to(device)
        
        # Normalize from [0, 1] to [-1, 1] as expected by LPIPS
        pred_normalized = predicted_image * 2.0 - 1.0
        target_normalized = target_image * 2.0 - 1.0
        
        with torch.no_grad():
            lpips_score = self.lpips(pred_normalized, target_normalized)
        
        return lpips_score.mean().item()

    def calculate_psnr(self, predicted_image, target_image):
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR)
        Assumes inputs are in range [0, 1]
        """
        mse = F.mse_loss(predicted_image, target_image)
        if mse == 0:
            return float('inf')
        
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

    def calculate_ssim(self, predicted_image, target_image):
        """
        Calculate Structural Similarity Index (SSIM)
        Handles both RGB and grayscale images
        """
        # Convert tensors to numpy arrays in range [0, 255]
        pred_np = predicted_image.detach().cpu().permute(0, 2, 3, 1).numpy()
        target_np = target_image.detach().cpu().permute(0, 2, 3, 1).numpy()
        
        # Clip values to [0, 1] and convert to [0, 255]
        pred_np = np.clip(pred_np, 0, 1) * 255
        target_np = np.clip(target_np, 0, 1) * 255
        
        ssim_scores = []
        for p, t in zip(pred_np, target_np):
            # Handle different image types
            if p.shape[-1] == 1:  # Grayscale
                p_gray = p.squeeze(-1)
                t_gray = t.squeeze(-1)
                ssim_score = structural_similarity(p_gray, t_gray, data_range=255)
            else:  # RGB
                ssim_score = structural_similarity(p, t, channel_axis=-1, data_range=255)
            
            ssim_scores.append(ssim_score)
        
        return np.mean(ssim_scores)

    def calculate_metrics(self, predicted_images_pil, target_images, device):
        transform = transforms.ToTensor()
        
        predicted_image_tensor_list = [transform(img) for img in predicted_images_pil]
        predicted_images_tensor = torch.stack(predicted_image_tensor_list)
        
        target_images = (target_images + 1.0)/2.0
        
        predicted_images_tensor = predicted_images_tensor.to(device)
        target_images = target_images.to(device)
        
        metrics = {}
        
        try:
            metrics['LPIPS'] = self.calculate_lpips(predicted_images_tensor, target_images)
        except Exception as e:
            print(f"LPIPS calculation failed: {e}")
            metrics['LPIPS'] = None
        
        try:
            metrics['PSNR'] = self.calculate_psnr(predicted_images_tensor, target_images)
        except Exception as e:
            print(f"PSNR calculation failed: {e}")
            metrics['PSNR'] = None
        
        try:
            metrics['SSIM'] = self.calculate_ssim(predicted_images_tensor, target_images)
        except Exception as e:
            print(f"SSIM calculation failed: {e}")
            metrics['SSIM'] = None

        return metrics
    
    def calculate_mae(self, predicted_tensor, target_tensor):
        mae = torch.mean(torch.abs((predicted_tensor - target_tensor))).item()
        return mae
    
    def calculate_mse(self, predicted_tensor, target_tensor):
        mse = torch.mean((predicted_tensor - target_tensor)**2).item()
        return mse
    
    def calculate_mse_batched(self, predicted_tensor, target_tensor):
        mse = torch.mean((predicted_tensor - target_tensor)**2,dim=1).tolist()
        return mse
    
    def calculate_pve(self, predicted_tensor, target_tensor):
        """Per-vertex Euclidean error (mm) between two neutral T-pose SMPL meshes.

        Both tensors are (B, N, 3) vertex sets; returns the mean per-vertex
        distance for each mesh in the batch.
        """
        predicted_tensor = predicted_tensor * 1000
        target_tensor = target_tensor * 1000
        pve = torch.sqrt(torch.sum((predicted_tensor - target_tensor) ** 2, dim=2)).mean(dim=1).tolist()
        return pve