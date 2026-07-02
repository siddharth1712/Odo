import os
import argparse
import torch
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from odo.models.reshapenet import ReshapeNetUNet
from odo.models.referencenet import ReferenceNetUNet

from odo.pipelines.odo_pipeline import OdoPipeline

from diffusers.utils.import_utils import is_xformers_available
from typing import List
from tqdm.auto import tqdm
from odo.data.dataset_inference import ChangeLing18KDataset, GenderBatchSampler

from odo.image_utils import *

from diffusers import ControlNetModel
from odo.metrics.image_metrics import MetricCalculation

from nlf.pt.multiperson.depth_prediction import SMPLPredictor

import smplx
import json
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default="/spiral_hdd_2/workspace/siddharth/IDM-VTON/final_reshape_no_ip_adapter/checkpoint-147500",required=False,help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--pretrained_controlnet_path",type=str,default="xinsir/controlnet-depth-sdxl-1.0",required=False,help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--root_dir",type=str,default="/spiral_hdd_2/workspace/siddharth/openpose/FINAL_DATASET_TEST",required=False,help="Path to the root directory.",)
    parser.add_argument("--depth_images_path",type=str,default="/spiral_hdd_2/workspace/siddharth/openpose/final_depth_images_test",required=False,help="Path to the depth images.",)
    parser.add_argument("--width",type=int,default=768)
    parser.add_argument("--height",type=int,default=1024)
    parser.add_argument("--output_dir",type=str,default="final_inference_no_ip_adapter",help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--mixed_precision",type=str,default="bf16",choices=["no", "fp16", "bf16"],help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="" 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"" flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),)
    parser.add_argument("--guidance_scale",type=float,default=2.0,)
    parser.add_argument("--seed", type=int, default=42,)    
    parser.add_argument("--num_inference_steps",type=int,default=50,)    
    parser.add_argument("--smpl_estimator_path",type=str,default='/spiral_hdd_2/workspace/siddharth/nlf/models/pt/nlf_l_crop.pt',help="Path to the SMPL estimator.",)
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def get_t_pose_vertices(betas,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    betas = betas.to(device)
    model = smplx.SMPL("/spiral_hdd_2/workspace/siddharth/nlf/data/body_models/smpl", use_pca=False, batch_size=betas.shape[0]).to(device)
    output = model(betas=betas,global_orient=torch.zeros((betas.shape[0],3)).to(device),body_pose=torch.zeros((betas.shape[0],23*3)).to(device),transl=torch.zeros((betas.shape[0],3)).to(device))
    
    vertices = output.vertices
    return vertices
    
def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16
    
    unet_encoder = ReferenceNetUNet.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet_encoder")
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",rescale_betas_zero_snr=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    
    unet_encoder.config.addition_embed_type = None
    unet_encoder.config["addition_embed_type"] = None
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path,subfolder="image_encoder")
    
    #customize unet start
    unet = ReshapeNetUNet.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",low_cpu_mem_usage=False, device_map=None)
    
    # pretrained_ip_adapter_path = os.path.join(args.pretrained_model_name_or_path,"ip_adapter_weights.bin")

    # state_dict = torch.load(pretrained_ip_adapter_path, map_location="cpu")

    # # Group parameters by module index
    # ip_adapter_state = state_dict["ip_adapter"]
    # ip_processor_keys = [key for key in unet.attn_processors.keys() if hasattr(unet.attn_processors[key], 'to_k_ip')]

    # # Group parameters by module index
    # module_params = defaultdict(dict)
    # for param_key, param_value in ip_adapter_state.items():
    #     module_idx, param_name = param_key.split('.', 1)
    #     module_idx = int(module_idx)
    #     module_params[module_idx][param_name] = param_value

    # # Sort by module index to maintain order
    # sorted_modules = sorted(module_params.items())

    # # Load weights into each processor sequentially
    # for i, (original_module_idx, params) in enumerate(sorted_modules):
    #     if i < len(ip_processor_keys):  # Safety check
    #         unet_key = ip_processor_keys[i]  # Use sequential index, not original module_idx
    #         unet.attn_processors[unet_key].load_state_dict(params, strict=False)
    
    controlnet = ControlNetModel.from_pretrained(args.pretrained_controlnet_path)
    
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    vae.to(device, dtype=weight_dtype) 
    text_encoder.to(device, dtype=weight_dtype)
    text_encoder_2.to(device, dtype=weight_dtype)
    # image_encoder.to(device, dtype=weight_dtype)
    unet_encoder.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    # image_encoder.eval()
    unet_encoder.eval()
    controlnet.eval()
    unet.eval()
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    test_dataset = ChangeLing18KDataset(
        root_dir=args.root_dir,
        depth_images_path = args.depth_images_path,
    )
    
    sampler = GenderBatchSampler(test_dataset, batch_size=args.batch_size)
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_sampler=sampler)
    
    metric_calculator = MetricCalculation(device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    inference_images_dir = os.path.join(args.output_dir,"images")
    os.makedirs(inference_images_dir, exist_ok=True)
    
    smpl_estimator = SMPLPredictor(model_path=args.smpl_estimator_path,batch_size=args.batch_size,resolution=(args.width,args.height),device=device)
    
    all_metrics = {}
    all_metrics["LPIPS"] = []
    all_metrics["PSNR"] = []
    all_metrics["SSIM"] = []
    # all_metrics["DEPTH_MSE"] = []
    all_metrics["BETA_MAE"] = []
    all_metrics["PVET"] = []
        
    with torch.no_grad():
        pipe = OdoPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            vae= vae,
            scheduler=noise_scheduler,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            # image_encoder=image_encoder,
            unet_encoder = unet_encoder,
            controlnet=controlnet,
            torch_dtype=weight_dtype,
            add_watermarker=False,
            safety_checker=None,
        ).to(device)
        
        
        for batch_num, sample in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Processing batches"):
            img_emb_list = []
            num_images = sample['image'].shape[0]
            gender = sample["gender"][0]
            
            base_img_names = [os.path.basename(name) for name in sample['edited_image_name']]
            edited_img_category = sample["edited_img_category"]
            og_img_category = sample["og_img_category"]
            
            # should_continue = True
            # for i in range(num_images):
            #     if not os.path.exists(os.path.join(inference_images_dir, f"{og_img_category[i]}_to_{edited_img_category[i]}_{base_img_names[i]}")):
            #         should_continue = False
            #         break
            
            # if should_continue:
            #     print("Skipping batch", batch_num)
            #     continue
            
            # for i in range(sample['reference_image'].shape[0]):
            #     img_emb_list.append(sample['reference_image'][i])

            prompt = sample["caption"]

            num_prompts = sample['reference_image'].shape[0]                                        
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
            target_depth_map = sample["depth_image"]
            target_depth_map_tensor = target_depth_map.to(device,dtype=weight_dtype)

            if not isinstance(prompt, List):
                prompt = [prompt] * num_prompts
            if not isinstance(negative_prompt, List):
                negative_prompt = [negative_prompt] * num_prompts

            # image_embeds = torch.cat(img_emb_list,dim=0).to(device,dtype=weight_dtype)
            
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            
            prompt_embeds = prompt_embeds.to(device,dtype=weight_dtype)
            negative_prompt_embeds = negative_prompt_embeds.to(device,dtype=weight_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device,dtype=weight_dtype)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device,dtype=weight_dtype)
            
            prompt = sample["reference_prompt"]
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

            if not isinstance(prompt, List):
                prompt = [prompt] * num_prompts
            if not isinstance(negative_prompt, List):
                negative_prompt = [negative_prompt] * num_prompts

            (
                prompt_embeds_c,
                _,
                _,
                _,
            ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt,
            )
            
            prompt_embeds_c = prompt_embeds_c.to(device,dtype=weight_dtype)

            generator = torch.Generator(pipe.device).manual_seed(args.seed) if args.seed is not None else None
            
            images = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                reference_prompt_embeds=prompt_embeds_c,
                reference_image = sample["reference_pixels"].to(device,dtype=weight_dtype),
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                # ip_adapter_image = image_embeds,
                depth_image = target_depth_map_tensor,
            )[0]
            
            metrics = metric_calculator.calculate_metrics(images, sample["image"], device)
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
            image_tensor_vertices_t_pose = get_t_pose_vertices(images_tensor_betas,args)
            
            edited_tensor = (sample["image"] + 1.0)/2.0
            edited_tensor = scale(edited_tensor)
            edited_tensor_betas = smpl_estimator.get_betas(edited_tensor,gender=gender)
            edited_tensor_vertices_t_pose = get_t_pose_vertices(edited_tensor_betas,args)
            
            all_metrics["BETA_MAE"] += metric_calculator.calculate_mse_batched(images_tensor_betas, edited_tensor_betas)
            all_metrics["PVET"] += metric_calculator.calculate_pve(image_tensor_vertices_t_pose, edited_tensor_vertices_t_pose)
            
            for i in range(len(images)):
                # Get the ground truth image from pil_image
                im_name = sample["original_image_name"][i]
                edited_im_name = sample["edited_image_name"][i]
                
                base_edited_name = os.path.basename(edited_im_name)
                edited_img_category = sample["edited_img_category"][i]
                og_img_category = sample["og_img_category"][i]
                
                og_pil = Image.open(im_name).resize((args.width, args.height))
                edited_pil = Image.open(edited_im_name).resize((args.width, args.height))
                
                # edited_tensor = pil_to_pt(edited_pil).to(device,dtype=weight_dtype)
                # edited_tensor = scale(edited_tensor)
                # edited_tensor_betas = smpl_estimator.get_betas(edited_tensor,gender=gender)
                # edited_tensor_vertices_t_pose = get_t_pose_vertices(edited_tensor_betas,args)
                
                # images_tensor = pil_to_pt(images[i]).to(device,dtype=weight_dtype)
                # images_tensor = scale(images_tensor)
                # images_tensor_betas = smpl_estimator.get_betas(images_tensor,gender=gender)
                # image_tensor_vertices_t_pose = get_t_pose_vertices(images_tensor_betas,args)
                
                # all_metrics["BETA_MAE"] += metric_calculator.calculate_mse_batched(images_tensor_betas, edited_tensor_betas)
                # all_metrics["PVET"] += metric_calculator.calculate_pve(image_tensor_vertices_t_pose, edited_tensor_vertices_t_pose)
                
                target_depth_pil = target_depth_map[i].detach().cpu().to(torch.float32).numpy()
                target_depth_pil = (target_depth_pil * 255).astype(np.uint8)
                target_depth_pil = np.transpose(target_depth_pil, (1, 2, 0))
                target_depth_pil = Image.fromarray(target_depth_pil)
                
                # Create a side-by-side comparison image
                result_width = args.width * 4
                result_height = args.height
                result_image = Image.new('RGB', (result_width, result_height))
                
                # Paste the ground truth and generated images
                result_image.paste(og_pil, (0, 0))
                result_image.paste(edited_pil, (args.width, 0))
                result_image.paste(target_depth_pil, (args.width * 2, 0))
                result_image.paste(images[i], (args.width * 3, 0))
                
                # Save the combined image
                result_image.save(os.path.join(inference_images_dir, f"{og_img_category}_to_{edited_img_category}_{base_edited_name}"))

    # Print all metrics in a readable format
    print("\n===== Evaluation Metrics =====")
    metrics_summary = {}
    for metric_name, values in all_metrics.items():
        if values:  # Check if the list is not empty
            avg_value = sum(values) / len(test_dataset)
            print(f"{metric_name}: {avg_value:.4f} (avg of {len(values)} samples)")
            metrics_summary[metric_name] = float(f"{avg_value:.4f}")
    
    # Save metrics to JSON file
    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics saved to {json_path}")
    
if __name__ == "__main__":
    main()