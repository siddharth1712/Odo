import smplx
from typing import NewType
import torch

Tensor = NewType('Tensor', torch.Tensor)

def get_weight(gender,betas,device):
    smpl= smplx.create(model_path="/spiral_hdd_2/workspace/siddharth/nlf/data/body_models",gender=gender,num_betas=10,model_type='smplx').to(device)
    
    body = smpl(betas=betas)
    shaped_vertices = body['v_shaped']
    shaped_triangles = shaped_vertices[:,smpl.faces_tensor]
    
    return compute_mass(shaped_triangles)

def compute_mass(tris: Tensor) -> Tensor:
    ''' Computes the mass from volume and average body density
    '''
    
    DENSITY = 985
    
    x = tris[:, :, :, 0]
    y = tris[:, :, :, 1]
    z = tris[:, :, :, 2]
    volume = (
        -x[:, :, 2] * y[:, :, 1] * z[:, :, 0] +
        x[:, :, 1] * y[:, :, 2] * z[:, :, 0] +
        x[:, :, 2] * y[:, :, 0] * z[:, :, 1] -
        x[:, :, 0] * y[:, :, 2] * z[:, :, 1] -
        x[:, :, 1] * y[:, :, 0] * z[:, :, 2] +
        x[:, :, 0] * y[:, :, 1] * z[:, :, 2]
    ).sum(dim=1).abs() / 6.0
    return volume * DENSITY