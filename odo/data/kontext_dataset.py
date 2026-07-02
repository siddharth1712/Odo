import os
import math
import torch
from PIL import Image
import random
import json
from torch.utils.data import Sampler

class KontextDataset(torch.utils.data.Dataset):
    def __init__(self, 
            root_dir="/spiral_hdd_2/workspace/siddharth/openpose/FINAL_DATASET_TEST",  
            width=768,
            height=1024,
        ):
        self.width = width
        self.height = height
        self.root_dir = root_dir
        
        # Load transformation pairs from JSON file
        self.transformation_pairs_path = os.path.join(self.root_dir, "transformation_pairs.json")
        if os.path.exists(self.transformation_pairs_path):
            with open(self.transformation_pairs_path, 'r') as f:
                self.transformation_pairs = json.load(f)
        else:
            self.transformation_pairs = {}
        
        self.final_image_pairs = []
        # Get all keys and shuffle them
        all_keys = list(self.transformation_pairs.keys())
        
        # Create pairs using the selected keys
        for key in all_keys:
            image_name = key + ".jpg"
            for pair in self.transformation_pairs[key]:
                og_img = pair[0]
                edited_img = pair[1]
                
                og_img_path = os.path.join(og_img, "images", image_name)
                edited_img_path = os.path.join(edited_img, "images", image_name)
                
                if os.path.exists(os.path.join(self.root_dir, og_img_path)) and os.path.exists(os.path.join(self.root_dir, edited_img_path)):
                    self.final_image_pairs.append((og_img_path, edited_img_path))
        
        # Shuffle the combined pairs
        random.seed(42)
        random.shuffle(self.final_image_pairs)
        
        with open('/spiral_hdd_2/workspace/siddharth/nlf/measurements_test.json', 'r') as file:
            data = json.load(file)
        
        self.measurements = data
            
        

    def __len__(self):
        return len(self.final_image_pairs)

    def __getitem__(self, idx):
        original_image_path, edited_image_path = self.final_image_pairs[idx]
        
        if "female" in original_image_path or "WOMEN" in original_image_path:
            gender = "female"
        elif "male" in original_image_path or "MEN" in original_image_path:
            gender = "male"
        else:
            gender = "neutral"
         
        # Get the highest directory in original_image_path
        og_img_category = original_image_path.split('/')[0]
        edited_img_category = edited_image_path.split('/')[0]
        
        og_image_basename = os.path.basename(original_image_path)
        edited_image_basename = os.path.basename(edited_image_path)
        
        og_key = og_img_category + '/' + og_image_basename
        edited_key = edited_img_category + '/' + edited_image_basename
        
        og_measurements = self.measurements[og_key]
        edited_measurements = self.measurements[edited_key]
        
        og_weight = og_measurements['weight']
        edited_weight = edited_measurements['weight']
        
        if og_img_category == "fat" and edited_img_category == "thin":
            prompt = "Make the person thinner. The person's weight should become " + str(edited_weight) + "kg. Ensure that the background of the image remains same. Make sure that therer is no change in identity, pose and clothing of the person. You only need to change the body shape of the person."
        elif og_img_category == "thin" and edited_img_category == "fat":
            prompt = "Make the person fatter. The person's weight should become " + str(edited_weight) + "kg. Ensure that the background of the image remains same. Make sure that therer is no change in identity, pose and clothing of the person. You only need to change the body shape of the person."
        elif og_img_category=="muscle" and edited_img_category=="thin":
            prompt = "Make the person thinner and less muscular. The person's weight should become " + str(edited_weight) + "kg. Ensure that the background of the image remains same. Make sure that therer is no change in identity, pose and clothing of the person. You only need to change the body shape of the person."
        elif og_img_category=="thin" and edited_img_category=="muscle":
            prompt = "Make the person muscular. The person's weight should become " + str(edited_weight) + "kg. Ensure that the background of the image remains same. Make sure that there is no change in identity, pose and clothing of the person. You only need to change the body shape of the person."
        elif og_img_category=="fat" and edited_img_category=="muscle":
            prompt = "Make the person muscular and thinner. The person's weight should become " + str(edited_weight) + "kg. Ensure that the background of the image remains same. Make sure that there is no change in identity, pose and clothing of the person. You only need to change the body shape of the person."
        elif og_img_category=="muscle" and edited_img_category=="fat":
            prompt = "Make the person fatter and less muscular. The person's weight should become " + str(edited_weight) + "kg. Ensure that the background of the image remains same. Make sure that there is no change in identity, pose and clothing of the person. You only need to change the body shape of the person."
        
        final_original_image_path = os.path.join(self.root_dir, original_image_path)
        final_edited_image_path = os.path.join(self.root_dir, edited_image_path)
        
        original_image = Image.open(final_original_image_path).convert("RGB").resize((self.width, self.height))
        edited_image = Image.open(final_edited_image_path).convert("RGB").resize((self.width, self.height))
        
        result = {}
        result["original_image_name"] = final_original_image_path
        result["edited_image_name"] = final_edited_image_path
        result["edited_img"] = edited_image
        result["og_img"] = original_image
        result["og_img_category"] = og_img_category
        result["edited_img_category"] = edited_img_category
        result["prompt"] = prompt
        result["gender"] = gender
        
        return result
    
        

def collate_fn(batch):
    original_image_name = [item["original_image_name"] for item in batch]
    edited_image_name = [item["edited_image_name"] for item in batch]
    edited_img = torch.stack([item["edited_img"] for item in batch])
    og_img = torch.stack([item["og_img"] for item in batch])
    og_img_category = [item["og_img_category"] for item in batch]
    edited_img_category = [item["edited_img_category"] for item in batch]
    prompt = [item["prompt"] for item in batch]
    gender = [item["gender"] for item in batch]
    
    return {
        "original_image_name": original_image_name, 
        "edited_image_name": edited_image_name,
        "edited_img": edited_img, 
        "og_img": og_img, 
        "og_img_category": og_img_category,
        "edited_img_category": edited_img_category,
        "prompt": prompt,
        "gender": gender
    }
    

class GenderBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.female_indices = []
        self.male_indices = []
        self.neutral_indices = []
        
        # Use same logic as dataset to ensure consistency
        for i, (og_path, ed_path) in enumerate(dataset.final_image_pairs):
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

if __name__ == "__main__":
    dataset_train = KontextDataset(
        root_dir="/spiral_hdd_2/workspace/siddharth/openpose/FINAL_DATASET_TRAIN",  
        depth_images_path = "/spiral_hdd_2/workspace/siddharth/openpose/final_depth_images_train",
        num_samples=None)
    
    dataset_train.get_per_category_counts()
    # dataset_train.get_number_distinct_faces()
    # dataset_train.create_dataset()
    # dataset_train.save_image_names()
    # dataset_train.save_all_image_pairs()
