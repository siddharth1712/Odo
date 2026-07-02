import os
import json
import torch
from PIL import Image
from diffusers import FluxKontextPipeline
from torchvision import transforms

from odo.data.kontext_dataset import KontextDataset
from odo.data.dataset_inference import GenderBatchSampler
from odo.metrics.image_metrics import MetricCalculation

from odo.image_utils import *
from nlf.pt.multiperson.depth_prediction import SMPLPredictor
import smplx

def get_t_pose_vertices(betas):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    betas = betas.to(device)
    model = smplx.SMPL("/spiral_hdd_2/workspace/siddharth/nlf/data/body_models/smpl", use_pca=False, batch_size=betas.shape[0]).to(device)
    output = model(betas=betas,global_orient=torch.zeros((betas.shape[0],3)).to(device),body_pose=torch.zeros((betas.shape[0],23*3)).to(device),transl=torch.zeros((betas.shape[0],3)).to(device))
    
    vertices = output.vertices
    return vertices

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    width=768
    height=1024
    batch_size=1
    
    pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
    smpl_estimator = SMPLPredictor(model_path="/spiral_hdd_2/workspace/siddharth/nlf/models/pt/nlf_l_crop.pt",batch_size=batch_size,resolution=(width,height),device=device)
    metric_calculator = MetricCalculation(device)
    
    pipe.to(device)

    test_dataset = KontextDataset()
    sampler = GenderBatchSampler(test_dataset, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_sampler=sampler)
    
    transform_norm = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
    
    os.makedirs("kontext_inference", exist_ok=True)

    all_metrics = {}
    all_metrics["LPIPS"] = []
    all_metrics["PSNR"] = []
    all_metrics["SSIM"] = []
    all_metrics["BETA_MAE"] = []
    all_metrics["PVET"] = []
    
    for batch in test_dataloader:
        input_image = batch["og_img"]
        target_image = batch["edited_img"]
        target_image_tensor = transform_norm(target_image)
        prompt = batch["prompt"]
        gender = batch["gender"][0]

        images = pipe(image=input_image,prompt=prompt,guidance_scale=2.5).images
        num_images = len(images)

        metrics = metric_calculator.calculate_metrics(images, target_image_tensor, device)
        for metric, value in metrics.items():
            if value is not None:
                all_metrics[metric].append(value*num_images)
            else:
                print(f"{metric}: Failed to calculate")
                
        transform = transforms.ToTensor()
            
        images_tensor_list = [transform(img) for img in images]
        images_tensor = torch.stack(images_tensor_list)
        images_tensor = scale(images_tensor)
        images_tensor_betas = smpl_estimator.get_betas(images_tensor,gender=gender)
        image_tensor_vertices_t_pose = get_t_pose_vertices(images_tensor_betas)

        edited_tensor = (target_image_tensor + 1.0)/2.0
        edited_tensor = scale(edited_tensor)
        edited_tensor_betas = smpl_estimator.get_betas(edited_tensor,gender=gender)
        edited_tensor_vertices_t_pose = get_t_pose_vertices(edited_tensor_betas)
        
        all_metrics["BETA_MAE"] += metric_calculator.calculate_mse_batched(images_tensor_betas, edited_tensor_betas)
        all_metrics["PVET"] += metric_calculator.calculate_pve(image_tensor_vertices_t_pose, edited_tensor_vertices_t_pose)
        
        for i in range(len(images)):
            # Get the ground truth image from pil_image
            im_name = batch["original_image_name"][i]
            edited_im_name = batch["edited_image_name"][i]
            
            base_edited_name = os.path.basename(edited_im_name)
            edited_img_category = batch["edited_img_category"][i]
            og_img_category = batch["og_img_category"][i]
            
            og_pil = Image.open(im_name).resize((width, height))
            edited_pil = Image.open(edited_im_name).resize((width, height))
            
            # Create a side-by-side comparison image
            result_width = width * 3
            result_height = height
            result_image = Image.new('RGB', (result_width, result_height))
            
            # Paste the ground truth and generated images
            result_image.paste(og_pil, (0, 0))
            result_image.paste(edited_pil, (width, 0))
            result_image.paste(images[i], (width * 2, 0))
            
            # Save the combined image
            result_image.save(os.path.join("kontext_inference", f"{og_img_category}_to_{edited_img_category}_{base_edited_name}"))
    
    # Print all metrics in a readable format
    print("\n===== Evaluation Metrics =====")
    metrics_summary = {}
    for metric_name, values in all_metrics.items():
        if values:  # Check if the list is not empty
            avg_value = sum(values) / len(test_dataset)
            print(f"{metric_name}: {avg_value:.4f} (avg of {len(values)} samples)")
            metrics_summary[metric_name] = float(f"{avg_value:.4f}")
    
    # Save metrics to JSON file
    json_path = os.path.join("kontext_inference", "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics saved to {json_path}")


if __name__ == "__main__":
    main()
