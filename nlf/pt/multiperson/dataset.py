import torch
import os
import torchvision

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = torchvision.io.read_image(image_path)
        
        image_name  = self.image_files[idx]
        
        batch = {
            "image": image,
            "image_name": image_name
        }
        return batch
    