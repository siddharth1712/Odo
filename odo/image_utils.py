import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def get_pil_depth_image(depth_image):
    depth_image = depth_image.detach().cpu().to(torch.float32).numpy()
    depth_image = (depth_image * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_image).convert("RGB")
    return depth_image
    
def scale(image):
    image = torch.clamp(image*255.0, 0, 255.0)
    return image

def pil_to_pt(images):
    transform = transforms.ToTensor()
    images = transform(images).unsqueeze(0)
    return images
