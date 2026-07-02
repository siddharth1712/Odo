import os
# Force override DATA_ROOT
os.environ['DATA_ROOT'] = '/spiral_hdd_2/workspace/siddharth/nlf/data'
import sys
# This will force modules to use our DATA_ROOT that import posepile.paths later
sys.modules['posepile.paths'] = type('', (), {'DATA_ROOT': '/spiral_hdd_2/workspace/siddharth/nlf/data', 'CACHE_DIR': '/spiral_hdd_2/workspace/siddharth/nlf/model_dir/cache'})

import torch
import nlf.pt.backbones.efficientnet as effnet_pytorch
import nlf.pt.models.field as pt_field
import nlf.pt.models.nlf_model as pt_nlf_model
from nlf.paths import DATA_ROOT
from torch import nn
import simplepyutils as spu
import argparse
from nlf.pt.util import get_config

from nlf.pt.multiperson.multiperson_model import MultipersonNLF
from nlf.pt.multiperson import person_detector

spu.FLAGS = argparse.Namespace()
spu.FLAGS.config_name = "nlf_l"
spu.FLAGS.centered_stride = False  # Adding this as it's used in efficientnet.py

import smplx
from nlf.pt.multiperson.rendering import *
from nlf.pt import ptu, ptu3d

class SMPLPredictor(nn.Module):
    def __init__(self, model_path, batch_size, resolution, device, model_type="smpl"):
        super().__init__()
        self.device = device
        
        skeleton_infos = spu.load_pickle(f"{DATA_ROOT}/skeleton_conversion/skeleton_types_huge8.pkl")
        cfg = get_config("nlf_l")
        backbone_raw = getattr(effnet_pytorch, f'efficientnet_v2_l')().to(self.device)
        preproc_layer = effnet_pytorch.PreprocLayer().to(self.device)
        backbone = torch.nn.Sequential(preproc_layer, backbone_raw.features).to(self.device)
        weight_field = pt_field.build_field().to(self.device)
        model_pytorch = pt_nlf_model.NLFModel(backbone, weight_field).to(self.device)
        model_pytorch.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        
        model_pytorch.eval()

        model_pytorch.backbone.half()
        model_pytorch.heatmap_head.layer.half()
        
        detector = person_detector.PersonDetector(f'{DATA_ROOT}/yolov8x.torchscript')
        self.model_type = model_type
        
        self.multimodel = {
            "neutral": MultipersonNLF(model_pytorch, detector, skeleton_infos, gender="neutral").to(self.device),
            "male": MultipersonNLF(model_pytorch, detector, skeleton_infos, gender="male").to(self.device),
            "female": MultipersonNLF(model_pytorch, detector, skeleton_infos, gender="female").to(self.device)
        } 
        
        for key in self.multimodel.keys():
            for param in self.multimodel[key].parameters():
                param.requires_grad = False
            
            self.multimodel[key].eval()
        
        if self.model_type == "smpl":
            self.model = {
                "neutral": smplx.SMPL("/spiral_hdd_2/workspace/siddharth/nlf/data/body_models/smpl", use_pca=False, batch_size=batch_size).to(self.device),
                "male": smplx.SMPL("/spiral_hdd_2/workspace/siddharth/nlf/data/body_models/smpl", gender="male", use_pca=False, batch_size=batch_size).to(self.device),
                "female": smplx.SMPL("/spiral_hdd_2/workspace/siddharth/nlf/data/body_models/smpl", gender="female", use_pca=False, batch_size=batch_size).to(self.device)
            }
        else:
            self.model = {
                "neutral": smplx.SMPLX("/spiral_hdd_2/workspace/siddharth/nlf/data/body_models/smplx", use_pca=False, batch_size=batch_size).to(self.device),
                "male": smplx.SMPLX("/spiral_hdd_2/workspace/siddharth/nlf/data/body_models/smplx", gender="male", use_pca=False, batch_size=batch_size).to(self.device),
                "female": smplx.SMPLX("/spiral_hdd_2/workspace/siddharth/nlf/data/body_models/smplx", gender="female", use_pca=False, batch_size=batch_size).to(self.device)
            }
        
        self.img_w, self.img_h = resolution
        
        intrinsic_matrix = ptu3d.intrinsic_matrix_from_field_of_view(55.0, [self.img_h, self.img_w], device=self.device)[0]
        self.focal_length = intrinsic_matrix[0,0]
        self.camera_center = [self.img_w // 2, self.img_h // 2]
        
    def forward(self,image_tensor,betas=None):
        # image_tensor is a tensor of shape (B,H,W,C)
        # where B is batch size, H is height, W is width, and C is channels
        if self.model_type == "smpl":
            pred = self.multimodel["neutral"].detect_smpl_batched(model_name='smpl',images=image_tensor.to(self.device))
        else:
            pred = self.multimodel["neutral"].detect_smpl_batched(model_name='smplx',images=image_tensor.to(self.device))
        
        return pred

    def get_depth_map(self,image_tensor,gender="neutral",betas=None,pose=None):
        # image_tensor is a tensor of shape (B,H,W,C)
        # where B is batch size, H is height, W is width, and C is channels
        image_tensor = image_tensor.to(self.device,dtype=torch.float32)
        
        if self.model_type == "smpl":
            pred = self.multimodel[gender].detect_smpl_batched(model_name='smpl',images=image_tensor)
        else:
            pred = self.multimodel[gender].detect_smpl_batched(model_name='smplx',images=image_tensor)
        
        if pose is None:
            pose = torch.concat(pred['pose'], dim=0)
        else:
            pose = pose.to(self.device)

        if betas is None:
            betas = torch.concat(pred['betas'], dim=0).to(self.device)
        else:
            betas = betas.to(self.device)
        
        transl = torch.concat(pred['trans'], dim=0).to(self.device)
        mean_poses = torch.concat(pred['mean_poses'], dim=0).to(self.device)
            
        if self.model_type == "smpl":
            model_output = self.model[gender](betas=betas,global_orient=pose[:,:3],body_pose=pose[:,3:],transl=transl)
        else:
            global_orient = pose[:,:3]
            body_pose = pose[:,3:22*3]
            leye_pose = pose[:,23*3:24*3]
            reye_pose = pose[:,24*3:25*3]
            left_hand_pose = pose[:,25*3:40*3]
            right_hand_pose = pose[:,40*3:55*3]
            
            model_output = self.model[gender](
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose
            )
       
        vertices = model_output.vertices
        vertices = vertices + (mean_poses / 1000.0)
        vertices = vertices * torch.tensor([1.0, -1.0, -1.0], dtype=vertices.dtype, device=vertices.device)
        
        depth_map = render_depth_map(
            vertices,  # Shape: [B, N, 3]
            focal_length=self.focal_length,
            camera_center=self.camera_center,
            image_size=(self.img_h, self.img_w),
            faces=self.model[gender].faces.astype(np.int64),
            invert_depth=True,
            device=self.device
        )
        
        return depth_map
    
    def get_betas(self,image_tensor,gender="neutral"):
        image_tensor = image_tensor.to(self.device,dtype=torch.float32)
        
        if self.model_type == "smpl":
            pred = self.multimodel[gender].detect_smpl_batched(model_name='smpl',images=image_tensor)
        else:
            pred = self.multimodel[gender].detect_smpl_batched(model_name='smplx',images=image_tensor)
        
        betas = torch.concat(pred['betas'], dim=0).to(self.device)
        return betas
    
    def get_pose(self,image_tensor,gender="neutral"):
        image_tensor = image_tensor.to(self.device,dtype=torch.float32)
        
        if self.model_type == "smpl":
            pred = self.multimodel[gender].detect_smpl_batched(model_name='smpl',images=image_tensor)
        else:
            pred = self.multimodel[gender].detect_smpl_batched(model_name='smplx',images=image_tensor)
        
        pose = torch.concat(pred['pose'], dim=0).to(self.device)
        return pose
    
    def get_smpl_params(self,image_tensor,gender="neutral"):
        image_tensor = image_tensor.to(self.device,dtype=torch.float32)
        
        if self.model_type == "smpl":
            pred = self.multimodel[gender].detect_smpl_batched(model_name='smpl',images=image_tensor)
        else:
            pred = self.multimodel[gender].detect_smpl_batched(model_name='smplx',images=image_tensor)
        
        betas = torch.concat(pred['betas'], dim=0).to(self.device)
        pose = torch.concat(pred['pose'], dim=0).to(self.device)
        transl = torch.concat(pred['trans'], dim=0).to(self.device)
        mean_poses = torch.concat(pred['mean_poses'], dim=0).to(self.device)
        return betas,pose,transl,mean_poses
        