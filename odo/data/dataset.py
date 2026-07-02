import os
import torch
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
import random
import json
import shutil

class ChangeLing18KDataset(torch.utils.data.Dataset):
    def __init__(self, 
            root_dir="/spiral_hdd_2/workspace/siddharth/openpose/FINAL_DATASET_TRAIN",  
            depth_images_path = "/spiral_hdd_2/workspace/siddharth/openpose/final_depth_images_train",
            width=768,
            height=1024,
            num_samples=None,
            use_fashion=True,
            prompt_mode="category",
        ):
        # prompt_mode: "category" -> per-transformation prompt (e.g. "Make the person fatter");
        #              "generic"  -> always "A photo of a person" (the "w/o prompts" ablation).
        assert prompt_mode in ("category", "generic"), prompt_mode
        self.width = width
        self.height = height
        self.root_dir = root_dir
        self.depth_images_path = depth_images_path
        self.num_samples = num_samples
        self.use_fashion = use_fashion
        self.prompt_mode = prompt_mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        
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
        
        if self.use_fashion == False:
            self.final_image_pairs = [pair for pair in self.final_image_pairs if "fashion" not in pair[0] and "fashion" not in pair[1]]
        
        # Shuffle the combined pairs
        random.seed(42)
        random.shuffle(self.final_image_pairs)
        
        if self.num_samples is not None:
            self.final_image_pairs = self.final_image_pairs[:self.num_samples]
        
        self.prompts = {
            'thin_muscle': "Make the person muscular",
            'thin_fat': "Make the person fatter",
            'muscle_thin': "Make the person thinner",
            'muscle_fat': "Make the person fatter",
            'fat_thin': "Make the person thinner",
            'fat_muscle': "Make the person muscular",
            'fashion_fashion': "A photo of a person"
        }

    def __len__(self):
        return len(self.final_image_pairs)

    def __getitem__(self, idx):
        original_image_path, edited_image_path = self.final_image_pairs[idx]
        
        if "female" in original_image_path or "WOMEN" in original_image_path:
            gender = "female"
        else:
            gender = "male"
         
        # Get the highest directory in original_image_path
        og_img_category = original_image_path.split('/')[0]
        edited_img_category = edited_image_path.split('/')[0]
         
        original_prompt = "A photo of a person"
        if self.prompt_mode == "category":
            edited_prompt = self.prompts[og_img_category + "_" + edited_img_category]
        else:  # "generic" — the w/o-prompts ablation
            edited_prompt = "A photo of a person"

        final_original_image_path = os.path.join(self.root_dir, original_image_path)
        final_edited_image_path = os.path.join(self.root_dir, edited_image_path)
        
        original_image = Image.open(final_original_image_path).convert("RGB").resize((self.width, self.height))
        edited_image = Image.open(final_edited_image_path).convert("RGB").resize((self.width, self.height))
        depth_image = Image.open(os.path.join(self.depth_images_path, edited_image_path)).convert("RGB").resize((self.width, self.height))
        
        clip_original_image = self.clip_image_processor(images=original_image, return_tensors="pt").pixel_values
        
        if self.transform:
            original_image = self.transform(original_image)
            edited_image = self.transform(edited_image)
            depth_image = transforms.ToTensor()(depth_image)
        
        result = {}
        result["original_image_name"] = final_original_image_path
        result["edited_image_name"] = final_edited_image_path
        result["image"] = edited_image
        result["depth_image"] = depth_image
        result["reference_image"] = clip_original_image
        result["reference_pixels"] = original_image
        result["caption"] = edited_prompt
        result["reference_prompt"] = original_prompt
        result["og_img_category"] = og_img_category
        result["edited_img_category"] = edited_img_category
        result["gender"] = gender
        
        return result
    
    def get_per_category_counts(self):
        # Count the number of each category pair
        category_counts = {}
        gender_counts = {'male': 0, 'female': 0}
        category_gender_counts = {}
        
        for og_path, ed_path in self.final_image_pairs:
            # Skip fashion category images
            if og_path.startswith('fashion/') or ed_path.startswith('fashion/'):
                continue
                
            og_category = og_path.split(os.sep)[0]
            ed_category = ed_path.split(os.sep)[0]
            pair_key = f"{og_category}_{ed_category}"
            
            # Determine gender
            if "female" in og_path or "WOMEN" in og_path:
                gender = "female"
            elif "male" in og_path or "MEN" in og_path:
                gender = "male"
            else:
                gender = "unknown"
            
            # Update category counts
            if pair_key in category_counts:
                category_counts[pair_key] += 1
            else:
                category_counts[pair_key] = 1
                category_gender_counts[pair_key] = {'male': 0, 'female': 0, 'unknown': 0}
            
            # Update gender counts
            if gender != "unknown":
                gender_counts[gender] += 1
                category_gender_counts[pair_key][gender] += 1
        
        print("Category pair counts (excluding fashion):")
        for pair_key, count in category_counts.items():
            male_count = category_gender_counts[pair_key]['male']
            female_count = category_gender_counts[pair_key]['female']
            print(f"  {pair_key}: {count} (male: {male_count}, female: {female_count})")
        
        print(f"Total gender counts - Male: {gender_counts['male']}, Female: {gender_counts['female']}")
        print(f"Total number of images= {gender_counts['male']+gender_counts['female']}")
    
    def get_number_distinct_faces(self):
        # Count distinct faces by extracting base name (without the last digit)
        distinct_faces = set()
        face_counts = {}
        processed_filenames = set()
        
        for og_path, _ in self.final_image_pairs:
            # Skip fashion category
            if og_path.startswith('fashion/'):
                continue
                
            # Skip if we've already processed this file
            if og_path in processed_filenames:
                continue
            
            processed_filenames.add(og_path)
                
            # Extract the filename without extension
            filename = os.path.basename(og_path).split('.')[0]
            # Get the base name (removing the last digit after underscore)
            parts = filename.rsplit('_', 1)
            if len(parts) > 1 and parts[1].isdigit():
                base_name = parts[0]
            else:
                base_name = filename
            
            distinct_faces.add(base_name)
            
            # Count occurrences of each face
            if base_name in face_counts:
                face_counts[base_name] += 1
            else:
                face_counts[base_name] = 1
        
        # Save face counts to JSON file
        with open(f'face_count_test.json', 'w') as f:
            json.dump(face_counts, f, indent=4)
            
        print(f"Number of distinct faces: {len(distinct_faces)}")
        print(f"Face counts saved to face_count.json")
        return len(distinct_faces)
    
    def save_image_names(self):
        # Extract distinct filenames from original image paths, excluding fashion category
        distinct_filenames = set()
        for og_path, _ in self.final_image_pairs:
            # Skip fashion category
            if og_path.startswith('fashion/'):
                continue
                
            filename = os.path.basename(og_path)
            distinct_filenames.add(filename)
        
        # Save to JSON file
        output_path = f'distinct_filenames.json'
        with open(output_path, 'w') as f:
            json.dump(list(distinct_filenames), f, indent=4)
        
        print(f"Saved {len(distinct_filenames)} distinct filenames to {output_path}")
        return distinct_filenames
    
    def save_all_image_pairs(self):
        # Save all image pairs to JSON file, excluding fashion category
        output_path = f'all_image_pairs_all.json'
        image_pairs_list = []
        for og_path, edited_path in self.final_image_pairs:
            # Skip fashion category images
            if og_path.startswith('fashion/') or edited_path.startswith('fashion/'):
                continue
                
            image_pairs_list.append({
                "original_image": og_path,
                "edited_image": edited_path
            })
        
        with open(output_path, 'w') as f:
            json.dump(image_pairs_list, f, indent=4)
        
        print(f"Saved {len(image_pairs_list)} non-fashion image pairs to {output_path}")
        
        return
    
    def create_dataset(self):
        # Create the destination directory
        dest_dir = "FINAL_DATASET_VAL"
        os.makedirs(dest_dir, exist_ok=True)
        
        # Dictionary to store transformation pairs
        transformation_pairs = {}
        
        # Process each image pair
        for og_path, edited_path in self.final_image_pairs:
            # Extract source category (thin/fat/muscle) and image name
            og_category = og_path.split('/')[0]
            og_image_name = os.path.basename(og_path)
            og_image_name_without_ext = os.path.splitext(og_image_name)[0]
            
            edited_category = edited_path.split('/')[0]
            
            # Create pair entry
            pair = [og_category, edited_category]
            
            # Add to transformation pairs dictionary
            if og_image_name_without_ext not in transformation_pairs:
                transformation_pairs[og_image_name_without_ext] = []
            
            # Add pair if not already in the list
            if pair not in transformation_pairs[og_image_name_without_ext]:
                transformation_pairs[og_image_name_without_ext].append(pair)
            
            # Copy original image to destination directory if it doesn't exist
            og_src_path = os.path.join(self.root_dir, og_path)
            og_dest_dir = os.path.join(dest_dir, f"{og_category}/images")
            og_dest_path = os.path.join(og_dest_dir, og_image_name)
            os.makedirs(og_dest_dir, exist_ok=True)
            if not os.path.exists(og_dest_path):
                shutil.copy2(og_src_path, og_dest_path)
            
            # Copy edited image to destination directory if it doesn't exist
            edited_src_path = os.path.join(self.root_dir, edited_path)
            edited_image_name = os.path.basename(edited_path)
            edited_dest_dir = os.path.join(dest_dir, f"{edited_category}/images")
            edited_dest_path = os.path.join(edited_dest_dir, edited_image_name)
            os.makedirs(edited_dest_dir, exist_ok=True)
            if not os.path.exists(edited_dest_path):
                shutil.copy2(edited_src_path, edited_dest_path)
        
        # Save transformation pairs to JSON file
        with open(os.path.join(dest_dir, "transformation_pairs.json"), 'w') as f:
            json.dump(transformation_pairs, f, indent=4)
        
        print(f"Created dataset in {dest_dir} with {len(transformation_pairs)} image sets")
        print(f"Saved transformation pairs to {os.path.join(dest_dir, 'transformation_pairs.json')}")


class BR5KDataset(ChangeLing18KDataset):
    """BR-5K training set (Table 2 "with BR-5K data" ablation).

    Assumes the BR-5K data has been preprocessed into the same on-disk layout as
    ChangeLing18K — a ``transformation_pairs.json`` at ``root_dir`` plus matching
    depth maps under ``depth_images_path`` — in which case it is a pure path swap
    (point ``--train_data_dir`` / ``--train_depth_images_path`` at the BR-5K roots).
    If the raw BR-5K layout differs, override ``__init__`` here to build
    ``self.final_image_pairs`` from the BR-5K structure.
    """
    pass


def get_dataset(name):
    """Return the dataset class for the ``--dataset`` flag."""
    return {"changeling": ChangeLing18KDataset, "br5k": BR5KDataset}[name]


def collate_fn(batch):
    original_image_name = [item["original_image_name"] for item in batch]
    edited_image_name = [item["edited_image_name"] for item in batch]
    image = torch.stack([item["image"] for item in batch])
    depth_image = torch.stack([item["depth_image"] for item in batch])
    reference_image = torch.stack([item["reference_image"] for item in batch])
    reference_pixels = torch.stack([item["reference_pixels"] for item in batch])
    caption = [item["caption"] for item in batch]
    reference_prompt = [item["reference_prompt"] for item in batch]
    og_img_category = [item["og_img_category"] for item in batch]
    edited_img_category = [item["edited_img_category"] for item in batch]
    gender = [item["gender"] for item in batch]
    
    return {
        "original_image_name": original_image_name, 
        "edited_image_name": edited_image_name,
        "image": image, 
        "depth_image": depth_image,
        "reference_image": reference_image, 
        "reference_pixels": reference_pixels, 
        "caption": caption, 
        "reference_prompt": reference_prompt,
        "og_img_category": og_img_category,
        "edited_img_category": edited_img_category,
        "gender": gender
    }
    
if __name__ == "__main__":
    dataset_train = ChangeLing18KDataset(
        root_dir="/spiral_hdd_2/workspace/siddharth/openpose/FINAL_DATASET_TEST",  
        depth_images_path = "/spiral_hdd_2/workspace/siddharth/openpose/final_depth_images_test",
        num_samples=None)
    
    dataset_train.get_per_category_counts()
    dataset_train.get_number_distinct_faces()
    # dataset_train.create_dataset()
    # dataset_train.save_image_names()
    # dataset_train.save_all_image_pairs()
