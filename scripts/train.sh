#!/bin/bash
# Train Odo (ReshapeNet + frozen ReferenceNet + Depth ControlNet + IP-Adapter).
accelerate launch scripts/train.py \
    --gradient_checkpointing \
    --use_8bit_adam \
    --mixed_precision="bf16" \
    --output_dir=final_controlnet_results \
    --train_batch_size=4
