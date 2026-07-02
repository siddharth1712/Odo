import os
import argparse
import itertools
import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
import gc
import shutil

from odo.models.reshapenet import ReshapeNetUNet
from odo.models.referencenet import ReferenceNetUNet

from odo.pipelines.odo_pipeline import OdoPipeline

from odo.models.ip_adapter.ip_adapter import Resampler
from typing import List
import math
from tqdm.auto import tqdm
from diffusers.training_utils import compute_snr
from odo.data.dataset import get_dataset, collate_fn

from nlf.pt.multiperson.depth_prediction import SMPLPredictor
from odo.image_utils import *

from diffusers import ControlNetModel
from odo.metrics.image_metrics import MetricCalculation
from odo.metrics.pve import get_t_pose_vertices

from diffusers.utils import is_wandb_available

from collections import defaultdict

if is_wandb_available():
    import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default="stabilityai/stable-diffusion-xl-base-1.0",required=False,help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--pretrained_referencenet_path",type=str,default="stabilityai/stable-diffusion-xl-base-1.0",required=False,help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--checkpointing_epoch",type=int,default=1,help=("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"" training using `--resume_from_checkpoint`."),)
    parser.add_argument("--pretrained_ip_adapter_path",type=str,default="ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin",help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",)
    parser.add_argument("--image_encoder_path",type=str,default="ckpt/image_encoder",required=False,help="Path to CLIP image encoder",)
    parser.add_argument("--gradient_checkpointing",action="store_true",help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)
    parser.add_argument("--width",type=int,default=768,)
    parser.add_argument("--height",type=int,default=1024,)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--logging_steps",type=int,default=5000,help=("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"" training using `--resume_from_checkpoint`."),)
    parser.add_argument("--output_dir",type=str,default="final_reshape",help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--snr_gamma",type=float,default=None,help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. ""More details here: https://arxiv.org/abs/2303.09556.",)
    parser.add_argument("--num_tokens",type=int,default=16,help=("IP adapter token nums"),)
    parser.add_argument("--learning_rate",type=float,default=1e-5,help="Learning rate to use.",)
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps",type=int,default=None,help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    # parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--mixed_precision",type=str,default=None,choices=["no", "fp16", "bf16"],help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="" 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"" flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),)
    parser.add_argument("--guidance_scale",type=float,default=2.0,)
    parser.add_argument("--seed", type=int, default=42,)    
    parser.add_argument("--num_inference_steps",type=int,default=50,)    
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-04, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--train_data_dir", type=str, default="/spiral_hdd_2/workspace/siddharth/openpose/FINAL_DATASET_TRAIN", help="Train data directory")
    parser.add_argument("--val_data_dir", type=str, default="/spiral_hdd_2/workspace/siddharth/openpose/FINAL_DATASET_VAL", help="Validation data directory")
    parser.add_argument("--train_depth_images_path", type=str, default="/spiral_hdd_2/workspace/siddharth/openpose/final_depth_images_train", help="Depth images directory")
    parser.add_argument("--val_depth_images_path", type=str, default="/spiral_hdd_2/workspace/siddharth/openpose/final_depth_images_val", help="Depth images directory")
    parser.add_argument("--smpl_estimator_path",type=str,default='/spiral_hdd_2/workspace/siddharth/nlf/models/pt/nlf_l_crop.pt',help="Path to the SMPL estimator.",)
    parser.add_argument("--start_step",type=int,default=0,help="Start step for training")
    parser.add_argument("--resume_from_checkpoint",type=str,default=None,help="Path to checkpoint to resume training from")
    parser.add_argument("--report_to",type=str,default=None,help="Logging support")
    parser.add_argument("--debug_smpl_metric",action="store_true",help="Debug SMPL metric")
    parser.add_argument("--calc_smpl_metric",action="store_true",help="Calculate SMPL metric")
    parser.add_argument("--num_train_samples",type=int,default=None,help="Number of training samples")
    parser.add_argument("--num_val_samples",type=int,default=None,help="Number of validation samples")
    # --- Ablation configuration (paper Table 2) ---
    parser.add_argument("--use_referencenet", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the ReferenceNet feature injection. --no-use_referencenet gives the 'w/o ReferenceNet' ablation (IP-Adapter only).")
    parser.add_argument("--prompt_mode", type=str, default="category", choices=["category", "generic"],
                        help="'category' uses the per-transformation prompt; 'generic' ('A photo of a person') is the 'w/o prompts' ablation.")
    parser.add_argument("--dataset", type=str, default="changeling", choices=["changeling", "br5k"],
                        help="Training dataset: ChangeLing18K (default) or BR-5K (Table 2 'with BR-5K data' ablation).")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def save_checkpoint(pipeline, save_path):
    # Save the pipeline normally
    pipeline.save_pretrained(save_path)
    
    # Save IP adapter weights in the same format as original checkpoint
    ip_processor_keys = [key for key in pipeline.unet.attn_processors.keys() 
                        if hasattr(pipeline.unet.attn_processors[key], 'to_k_ip')]
    
    ip_adapter_weights = {}
    for i, unet_key in enumerate(ip_processor_keys):
        processor = pipeline.unet.attn_processors[unet_key]
        # Save with module index format like "0.to_k_ip.weight", "0.to_v_ip.weight"
        ip_adapter_weights[f"{i}.to_k_ip.weight"] = processor.to_k_ip.weight.data
        ip_adapter_weights[f"{i}.to_v_ip.weight"] = processor.to_v_ip.weight.data
    
    if ip_adapter_weights:
        # Save in the same format as original checkpoint
        checkpoint_data = {
            "ip_adapter": ip_adapter_weights
        }
        print("SAVING IP ADAPTER WEIGHTS")
        torch.save(checkpoint_data, os.path.join(save_path, "ip_adapter_weights.bin"))
        
def main():
    args = parse_args()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config,
        log_with=args.report_to,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    if args.resume_from_checkpoint is not None:
        args.pretrained_model_name_or_path = args.resume_from_checkpoint
        args.pretrained_referencenet_path = args.resume_from_checkpoint
        unet_encoder = ReferenceNetUNet.from_pretrained(args.pretrained_referencenet_path, subfolder="unet_encoder")
        
        # Extract checkpoint number from the path basename
        checkpoint_dir = os.path.basename(args.resume_from_checkpoint)
        args.start_step = int(checkpoint_dir.split("-")[1])
    else:
        if args.pretrained_referencenet_path=="yisol/IDM-VTON":
            unet_encoder = ReferenceNetUNet.from_pretrained(args.pretrained_referencenet_path, subfolder="unet_encoder")
        else:
            unet_encoder = ReferenceNetUNet.from_pretrained(args.pretrained_referencenet_path, subfolder="unet")
    
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",rescale_betas_zero_snr=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",torch_dtype=torch.bfloat16)
    
    unet_encoder.config.addition_embed_type = None
    unet_encoder.config["addition_embed_type"] = None
    if args.resume_from_checkpoint is None:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    else:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.resume_from_checkpoint, subfolder="image_encoder")

    #customize unet start
    unet = ReshapeNetUNet.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",low_cpu_mem_usage=False, device_map=None)
    
    if args.resume_from_checkpoint is not None:
        args.pretrained_ip_adapter_path = os.path.join(args.resume_from_checkpoint,"ip_adapter_weights.bin")
    
    unet.config.encoder_hid_dim = image_encoder.config.hidden_size
    unet.config.encoder_hid_dim_type = "ip_image_proj"
    unet.config["encoder_hid_dim"] = image_encoder.config.hidden_size
    unet.config["encoder_hid_dim_type"] = "ip_image_proj"
    
    state_dict = torch.load(args.pretrained_ip_adapter_path, map_location="cpu")
    
    if args.resume_from_checkpoint is None:
        #ip-adapter
        image_proj_model = Resampler(
            dim=image_encoder.config.hidden_size,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=args.num_tokens,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(accelerator.device, dtype=torch.float32)

        image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        image_proj_model.requires_grad_(True)

        unet.encoder_hid_proj = image_proj_model
    else:
        image_proj_model = unet.encoder_hid_proj
        image_proj_model.requires_grad_(True)
    
    controlnet = ControlNetModel.from_pretrained("xinsir/controlnet-depth-sdxl-1.0",torch_dtype=torch.bfloat16)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=weight_dtype) 
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    unet_encoder.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    unet.requires_grad_(True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        unet_encoder.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()
    
    unet.train()
    
    ip_adapter_state = state_dict["ip_adapter"]
    ip_processor_keys = [key for key in unet.attn_processors.keys() if hasattr(unet.attn_processors[key], 'to_k_ip')]

    # Group parameters by module index
    module_params = defaultdict(dict)
    for param_key, param_value in ip_adapter_state.items():
        module_idx, param_name = param_key.split('.', 1)
        module_idx = int(module_idx)
        module_params[module_idx][param_name] = param_value

    # Sort by module index to maintain order
    sorted_modules = sorted(module_params.items())

    # Load weights into each processor sequentially
    for i, (original_module_idx, params) in enumerate(sorted_modules):
        if i < len(ip_processor_keys):  # Safety check
            unet_key = ip_processor_keys[i]  # Use sequential index, not original module_idx
            unet.attn_processors[unet_key].load_state_dict(params, strict=False)

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_opt = itertools.chain(unet.parameters())
    
    if accelerator.is_main_process:
        # Log parameter counts for different modules
        
        print("Number of parameters:")
        total_params = sum(p.numel() for p in unet.parameters())
        trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        print(f"  UNet: {total_params:,} total, {trainable_params:,} trainable")

        total_params = sum(p.numel() for p in unet_encoder.parameters())
        trainable_params = sum(p.numel() for p in unet_encoder.parameters() if p.requires_grad)
        print(f"  UNet Encoder: {total_params:,} total, {trainable_params:,} trainable")

        total_params = sum(p.numel() for p in controlnet.parameters())
        trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
        print(f"  ControlNet: {total_params:,} total, {trainable_params:,} trainable")

        total_params = sum(p.numel() for p in image_proj_model.parameters())
        trainable_params = sum(p.numel() for p in image_proj_model.parameters() if p.requires_grad)
        print(f"  Image Projection Model: {total_params:,} total, {trainable_params:,} trainable")
    

    optimizer = optimizer_class(
        params_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    dataset_cls = get_dataset(args.dataset)
    train_dataset = dataset_cls(
        root_dir=args.train_data_dir,
        depth_images_path=args.train_depth_images_path,
        width=args.width,
        height=args.height,
        num_samples=args.num_train_samples,
        prompt_mode=args.prompt_mode,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        pin_memory=True,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=16,
        collate_fn=collate_fn,
    )
    test_dataset = dataset_cls(
        root_dir=args.val_data_dir,
        depth_images_path=args.val_depth_images_path,
        width=args.width,
        height=args.height,
        num_samples=args.num_val_samples,
        prompt_mode=args.prompt_mode,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=4,
        collate_fn=collate_fn,
    )
    
    metric_calculator = MetricCalculation(accelerator.device)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    unet,image_proj_model,unet_encoder,image_encoder,optimizer,train_dataloader,test_dataloader = accelerator.prepare(unet, image_proj_model,unet_encoder,image_encoder,optimizer,train_dataloader,test_dataloader)
    initial_global_step = args.start_step

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = initial_global_step
    first_epoch = global_step // num_update_steps_per_epoch
    train_loss = 0.0
    
    if accelerator.is_main_process:
        tracker_name = args.output_dir
        accelerator.init_trackers(tracker_name, config=vars(args))
        
    # Train!
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    if accelerator.is_main_process:
        print("***** Running training *****")
        print(f"  Total Num examples = {len(train_dataset)}")
        print(f"  Num batches each epoch = {len(train_dataloader)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_train_steps}")
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet), accelerator.accumulate(image_proj_model):
                if global_step % args.logging_steps == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        with torch.no_grad():
                            print("\n")
                            print("="*50)
                            print(f"Running Validation at step: {global_step}")
                            print("="*50)
                            save_dir = os.path.join(args.output_dir, "validation_images",str(global_step))
                            os.makedirs(save_dir, exist_ok=True)
                            unwrapped_unet= accelerator.unwrap_model(unet)
                            newpipe = OdoPipeline.from_pretrained(
                                args.pretrained_model_name_or_path,
                                unet=unwrapped_unet,
                                vae= vae,
                                scheduler=noise_scheduler,
                                tokenizer=tokenizer,
                                tokenizer_2=tokenizer_2,
                                text_encoder=text_encoder,
                                text_encoder_2=text_encoder_2,
                                image_encoder=image_encoder,
                                unet_encoder = unet_encoder,
                                controlnet=controlnet,
                                torch_dtype=weight_dtype,
                                add_watermarker=False,
                                safety_checker=None,
                            ).to(accelerator.device)
                            
                            if args.calc_smpl_metric:
                                smpl_estimator = SMPLPredictor(
                                    model_path=args.smpl_estimator_path,
                                    batch_size=args.test_batch_size,
                                    resolution=(args.width,args.height),
                                    device=accelerator.device
                                )
                            
                            all_metrics = {}
                            all_metrics["LPIPS"] = []
                            all_metrics["PSNR"] = []
                            all_metrics["SSIM"] = []
                            if args.calc_smpl_metric:
                                all_metrics["PVE_T_SC"] = []
                            total_images_processed = 0
                            for batch_num, sample in enumerate(tqdm(test_dataloader, desc="Evaluating batches")):
                                img_emb_list = []
                                num_images = sample['image'].shape[0]
                                total_images_processed += num_images
                                for i in range(sample['reference_image'].shape[0]):
                                    img_emb_list.append(sample['reference_image'][i])

                                prompt = sample["caption"]

                                num_prompts = sample['reference_image'].shape[0]                                        
                                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                                
                                target_depth_map = sample["depth_image"]
                                target_depth_map_tensor = target_depth_map.to(accelerator.device,dtype=weight_dtype)

                                if not isinstance(prompt, List):
                                    prompt = [prompt] * num_prompts
                                if not isinstance(negative_prompt, List):
                                    negative_prompt = [negative_prompt] * num_prompts

                                image_embeds = torch.cat(img_emb_list,dim=0).to(accelerator.device,dtype=weight_dtype)
                                
                                (
                                    prompt_embeds,
                                    negative_prompt_embeds,
                                    pooled_prompt_embeds,
                                    negative_pooled_prompt_embeds,
                                ) = newpipe.encode_prompt(
                                    prompt,
                                    num_images_per_prompt=1,
                                    do_classifier_free_guidance=True,
                                    negative_prompt=negative_prompt,
                                )
                                
                                prompt_embeds = prompt_embeds.to(accelerator.device,dtype=weight_dtype)
                                negative_prompt_embeds = negative_prompt_embeds.to(accelerator.device,dtype=weight_dtype)
                                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device,dtype=weight_dtype)
                                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(accelerator.device,dtype=weight_dtype)
                                
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
                                ) = newpipe.encode_prompt(
                                    prompt,
                                    num_images_per_prompt=1,
                                    do_classifier_free_guidance=False,
                                    negative_prompt=negative_prompt,
                                )
                                
                                prompt_embeds_c = prompt_embeds_c.to(accelerator.device,dtype=weight_dtype)

                                generator = torch.Generator(newpipe.device).manual_seed(args.seed) if args.seed is not None else None
                                images = newpipe(
                                    prompt_embeds=prompt_embeds,
                                    negative_prompt_embeds=negative_prompt_embeds,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                    num_inference_steps=args.num_inference_steps,
                                    generator=generator,
                                    reference_prompt_embeds=prompt_embeds_c,
                                    reference_image = sample["reference_pixels"].to(accelerator.device,dtype=weight_dtype) if args.use_referencenet else None,
                                    height=args.height,
                                    width=args.width,
                                    guidance_scale=args.guidance_scale,
                                    ip_adapter_image = image_embeds,
                                    depth_image = target_depth_map_tensor,
                                )[0]
                                
                                metrics = metric_calculator.calculate_metrics(images, sample["image"], accelerator.device)
                                for metric, value in metrics.items():
                                    if value is not None:
                                        all_metrics[metric].append(value*num_images)
                                    else:
                                        print(f"{metric}: Failed to calculate")

                                
                                for i in range(len(images)):
                                    # Get the ground truth image from pil_image
                                    im_name = sample["original_image_name"][i]
                                    edited_im_name = sample["edited_image_name"][i]
                                    
                                    gender = sample["gender"][i]
                                    
                                    base_edited_name = os.path.basename(edited_im_name)
                                    edited_img_category = sample["edited_img_category"][i]
                                    og_img_category = sample["og_img_category"][i]
                                    
                                    og_pil = Image.open(im_name).resize((args.width, args.height))
                                    edited_pil = Image.open(edited_im_name).resize((args.width, args.height))
                                    
                                    if args.calc_smpl_metric:
                                        # PVE-T-SC: fit SMPL to the ground-truth and generated images,
                                        # pose both to a neutral T-pose, and measure per-vertex error (mm).
                                        edited_tensor = scale(pil_to_pt(edited_pil).to(accelerator.device,dtype=weight_dtype))
                                        images_tensor = scale(pil_to_pt(images[i]).to(accelerator.device,dtype=weight_dtype))

                                        edited_betas = smpl_estimator.get_betas(edited_tensor, gender=gender)
                                        images_betas = smpl_estimator.get_betas(images_tensor, gender=gender)
                                        edited_vertices_t_pose = get_t_pose_vertices(edited_betas, gender)
                                        images_vertices_t_pose = get_t_pose_vertices(images_betas, gender)

                                        all_metrics["PVE_T_SC"].extend(
                                            metric_calculator.calculate_pve(images_vertices_t_pose, edited_vertices_t_pose)
                                        )
                                    
                                    # target_depth_pil = get_pil_depth_image(target_depth_map[i])
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
                                    result_image.save(os.path.join(save_dir, f"{og_img_category}_to_{edited_img_category}_{base_edited_name}"))
                                    
                                    if args.debug_smpl_metric and args.calc_smpl_metric:
                                        debug_dir = os.path.join(args.output_dir,"debug_images",str(global_step))
                                        os.makedirs(debug_dir, exist_ok=True)
                                        debug_depth_image = Image.new('RGB', (args.width * 2, args.height))
                                        images_depth_t_pose = smpl_estimator.get_depth_map(images_tensor, pose=torch.zeros((1,72)), gender=gender)
                                        edited_depth_t_pose = smpl_estimator.get_depth_map(edited_tensor, pose=torch.zeros((1,72)), gender=gender)
                                        images_tensor_depth_map_t_pose_pil = get_pil_depth_image(images_depth_t_pose[0])
                                        edited_tensor_depth_map_t_pose_pil = get_pil_depth_image(edited_depth_t_pose[0])
                                        
                                        debug_depth_image.paste(edited_tensor_depth_map_t_pose_pil, (0, 0))
                                        debug_depth_image.paste(images_tensor_depth_map_t_pose_pil, (args.width, 0))
                                        
                                        debug_depth_image.save(os.path.join(debug_dir, f"{og_img_category}_to_{edited_img_category}_depth_{base_edited_name}"))
                            
                            # Prepare metrics for logging
                            log_dict = {}
                            for metric_name, values in all_metrics.items():
                                if values:
                                    avg_value = sum(values) / total_images_processed
                                    log_dict[f"val_{metric_name.lower()}"] = avg_value
                            
                            # Log to tracker
                            accelerator.log(log_dict, step=global_step)
                        
                        del unwrapped_unet
                        del newpipe
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                pixel_values = batch["image"].to(dtype=vae.dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)

                bsz = model_input.shape[0]
                timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(model_input, noise, timesteps)
            
                text_input_ids = tokenizer(
                    batch['caption'],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                text_input_ids_2 = tokenizer_2(
                    batch['caption'],
                    max_length=tokenizer_2.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids

                encoder_output = text_encoder(text_input_ids.to(accelerator.device), output_hidden_states=True)
                text_embeds = encoder_output.hidden_states[-2]
                encoder_output_2 = text_encoder_2(text_input_ids_2.to(accelerator.device), output_hidden_states=True)
                pooled_text_embeds = encoder_output_2[0]
                text_embeds_2 = encoder_output_2.hidden_states[-2]
                encoder_hidden_states = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat


                def compute_time_ids(original_size, crops_coords_top_left = (0,0)):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.height, args.height) 
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device)
                    return add_time_ids
                
                add_time_ids = torch.cat(
                    [compute_time_ids((args.height, args.height)) for i in range(bsz)]
                )
                        
                img_emb_list = []
                for i in range(bsz):
                    img_emb_list.append(batch['reference_image'][i])
                
                image_embeds = torch.cat(img_emb_list,dim=0)
                image_embeds = image_encoder(image_embeds, output_hidden_states=True).hidden_states[-2]
                ip_tokens =image_proj_model(image_embeds)

                # add cond
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                unet_added_cond_kwargs["image_embeds"] = ip_tokens

                reference_latents = batch["reference_pixels"].to(accelerator.device,dtype=vae.dtype)
                reference_latents = vae.encode(reference_latents).latent_dist.sample()
                reference_latents = reference_latents * vae.config.scaling_factor

                text_input_ids = tokenizer(
                    batch['reference_prompt'],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                text_input_ids_2 = tokenizer_2(
                    batch['reference_prompt'],
                    max_length=tokenizer_2.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids

            
                encoder_output = text_encoder(text_input_ids.to(accelerator.device), output_hidden_states=True)
                reference_prompt_embeds = encoder_output.hidden_states[-2]
                encoder_output_2 = text_encoder_2(text_input_ids_2.to(accelerator.device), output_hidden_states=True)
                text_embeds_2_reference = encoder_output_2.hidden_states[-2]
                reference_prompt_embeds = torch.concat([reference_prompt_embeds, text_embeds_2_reference], dim=-1) # concat

                if args.use_referencenet:
                    down,reference_features = unet_encoder(reference_latents,timesteps, reference_prompt_embeds,return_dict=False)
                    reference_features = list(reference_features)
                else:
                    reference_features = None

                controlnet_image = batch["depth_image"].to(accelerator.device,dtype=weight_dtype)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=unet_added_cond_kwargs,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )
                
                noise_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states,
                    added_cond_kwargs=unet_added_cond_kwargs,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    reference_features=reference_features
                    ).sample

                target = noise
                
                if args.snr_gamma is None:
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                
                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_opt, 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "learning_rate": args.learning_rate}, step=global_step)
                train_loss = 0.0
            logs = {"step_loss": loss.detach().item(), "learning_rate": args.learning_rate}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        
        if epoch % args.checkpointing_epoch == 0:
            if accelerator.is_main_process:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                unwrapped_unet = accelerator.unwrap_model(
                    unet, keep_fp32_wrapper=True
                )
                pipeline = OdoPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrapped_unet,
                    vae= vae,
                    scheduler=noise_scheduler,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    image_encoder=image_encoder,
                    unet_encoder=unet_encoder,
                    controlnet=controlnet,
                    torch_dtype=torch.bfloat16,
                    add_watermarker=False,
                    safety_checker=None,
                )
                
                if os.path.exists(args.output_dir):
                    checkpoint_folders = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, f))]
                    for folder in checkpoint_folders:
                        folder_path = os.path.join(args.output_dir, folder)
                        shutil.rmtree(folder_path)
                        
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                save_checkpoint(pipeline, save_path)
                args.pretrained_model_name_or_path = save_path
                del pipeline

                
if __name__ == "__main__":
    main()    
