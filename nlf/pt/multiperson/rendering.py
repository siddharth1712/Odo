import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PerspectiveCameras,
    TexturesVertex,
    SoftPhongShader,
    PointLights
)
from pytorch3d.renderer.mesh.shader import ShaderBase
import numpy as np
import cv2
from PIL import Image

class DepthShader(ShaderBase):
    """Simple shader that returns depth values"""
    def forward(self, fragments, meshes, **kwargs):
        depth = fragments.zbuf[..., 0]
        mask = fragments.pix_to_face[..., 0] >= 0
        depth = torch.where(mask, depth, torch.zeros_like(depth))
        return depth

def render_depth_map(
    verts, 
    focal_length, 
    camera_center, 
    image_size, 
    faces, 
    invert_depth=False, 
    device='cuda'
):
    """
    Differentiable depth rendering that matches pyrender setup exactly using PyTorch3D.
    """
    # Move to device
    verts = verts.to(device)
    faces_tensor = torch.tensor(faces, device=device)

    H, W = image_size
    batch_size = verts.shape[0]

    # Flip Z to match PyTorch3D's -Z convention
    verts_transformed = verts.clone()
    verts_transformed[:, :, 2] = -verts_transformed[:, :, 2]

    # Create mesh
    faces_batch = faces_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    meshes = Meshes(verts=verts_transformed, faces=faces_batch)

    # Camera intrinsics (use pixel-based, not NDC)
    fx = fy = float(focal_length)
    cx, cy = camera_center

    cameras = PerspectiveCameras(
        device=device,
        R=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
        T=torch.zeros(batch_size, 3, device=device),
        focal_length=torch.tensor([[fx, fy]], device=device).expand(batch_size, -1),
        principal_point=torch.tensor([[cx, cy]], device=device).expand(batch_size, -1),
        image_size=torch.tensor([[H, W]], device=device).expand(batch_size, -1),
        in_ndc=False  # Important!
    )

    # Renderer
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = DepthShader(device=device)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    # Render depth map
    depth_map = renderer(meshes, cameras=cameras)

    # Flip horizontally to match Pyrender view if needed
    depth_map = torch.flip(depth_map, dims=[2])

    # Normalize depth
    valid_mask = depth_map > 0
    depth_normalized = torch.zeros_like(depth_map)

    if valid_mask.any():
        depth_valid = depth_map[valid_mask]
        depth_min = depth_valid.min()
        depth_max = depth_valid.max()
        depth_normalized[valid_mask] = (depth_map[valid_mask] - depth_min) / (depth_max - depth_min)

        if invert_depth:
            depth_normalized = 1.0 - depth_normalized

        depth_normalized = depth_normalized * valid_mask

        # Convert tensor to numpy and preserve batch dimension
        depth_numpy = depth_normalized.cpu().numpy().astype(np.float32)
        batch_size, H, W = depth_numpy.shape
        
        # Apply CLAHE to each image in the batch
        clahe_outputs = []
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        
        for i in range(batch_size):
            # Get single image from batch
            single_depth = depth_numpy[i]  # Shape: (H, W)
            
            # Convert to uint8, ensuring valid range
            single_depth = np.clip(single_depth, 0, 1)
            depth_uint8 = (single_depth * 255).astype(np.uint8)
            
            # Apply CLAHE
            clahe_result = clahe.apply(depth_uint8)
            
            # Convert back to float32 in [0,1] range
            clahe_float = clahe_result.astype(np.float32) / 255.0
            clahe_outputs.append(clahe_float)
        
        # Stack back to batch format
        clahe_output_numpy = np.stack(clahe_outputs, axis=0)  # Shape: (batch_size, H, W)
        
        # Convert back to tensor
        clahe_output = torch.from_numpy(clahe_output_numpy)
        if depth_map.is_cuda:
            clahe_output = clahe_output.cuda()
    else:
        clahe_output = depth_normalized
    
    return clahe_output

def render_depth_colormap_pytorch3d(
    verts, 
    focal_length, 
    camera_center, 
    image_size, 
    faces, 
    invert_depth=False, 
    device='cuda'
):
    """
    Render a depth map as a colored RGB image using PyTorch3D only.
    Depth is mapped to RGB using a heatmap-like color scheme.
    """
    verts = verts.to(device)
    faces_tensor = torch.tensor(faces, dtype=torch.long, device=device)

    H, W = image_size
    batch_size = verts.shape[0]

    # Flip Z to match PyTorch3D convention
    verts_transformed = verts.clone()
    verts_transformed[:, :, 2] = -verts_transformed[:, :, 2]

    # Normalize Z values for colormap [0, 1]
    z = verts_transformed[:, :, 2]
    z_min = z.min(dim=1, keepdim=True)[0]
    z_max = z.max(dim=1, keepdim=True)[0]
    z_norm = (z - z_min) / (z_max - z_min + 1e-6)
    if invert_depth:
        z_norm = 1.0 - z_norm

    # Map normalized Z to RGB using a jet-like colormap (custom torch implementation)
    def jet_colormap(z_val):
        r = torch.clamp(1.5 - torch.abs(4 * (z_val - 0.75)), 0, 1)
        g = torch.clamp(1.5 - torch.abs(4 * (z_val - 0.5)), 0, 1)
        b = torch.clamp(1.5 - torch.abs(4 * (z_val - 0.25)), 0, 1)
        return torch.stack([r, g, b], dim=-1)

    colors = jet_colormap(z_norm)

    # Create Meshes with vertex colors
    faces_batch = faces_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    meshes = Meshes(verts=verts_transformed, faces=faces_batch, textures=TexturesVertex(verts_features=colors))

    # Define camera
    fx = fy = float(focal_length)
    cx, cy = camera_center

    cameras = PerspectiveCameras(
        device=device,
        R=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
        T=torch.zeros(batch_size, 3, device=device),
        focal_length=torch.tensor([[fx, fy]], device=device).expand(batch_size, -1),
        principal_point=torch.tensor([[cx, cy]], device=device).expand(batch_size, -1),
        image_size=torch.tensor([[H, W]], device=device).expand(batch_size, -1),
        in_ndc=False
    )

    # Rasterizer and renderer
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )

    # Render color-mapped depth
    image = renderer(meshes)  # (B, H, W, 3)
    
    # Use rasterizer to get the silhouette mask
    fragments = renderer.rasterizer(meshes)
    mask = fragments.zbuf[..., 0] > 0  # (B, H, W), valid if z > 0

    # Set background to black
    image = image * mask.unsqueeze(-1)  # Zero out invalid pixels
    
    image = torch.flip(image, dims=[2])  # Match pyrender horizontal flip

    return image.permute(0, 3, 1, 2)  # Return (B, 3, H, W)

def overlay_depth_on_image(depth_map, cv2_img, alpha=0.8):
    """
    Overlay grayscale depth map on image for alignment check.
    """
    # Convert depth to numpy (detach first to avoid gradient issues)
    depth_np = depth_map.detach().cpu().numpy()

    # Create mask for valid depth values
    valid_mask = depth_np > 0

    # Convert to uint8 (0-255)
    depth_uint8 = (depth_np * 255).astype(np.uint8)

    # Create grayscale RGB (3-channel) version of depth map
    depth_gray_rgb = np.stack([depth_uint8] * 3, axis=-1)

    # Set invalid regions to 0
    depth_gray_rgb[~valid_mask] = 0

    # Create overlay - only blend where we have valid depth
    overlay = cv2_img.copy()
    overlay[valid_mask] = cv2.addWeighted(
        cv2_img[valid_mask],
        1 - alpha,
        depth_gray_rgb[valid_mask],
        alpha,
        0
    )

    return overlay

# Simple save function
def save_depth_gray(depth_map, path):
    depth = depth_map.detach().cpu().numpy()
    depth_img = (depth * 255).astype(np.uint8)
    Image.fromarray(depth_img).save(path)
    
def save_rgb_image(rgb_tensor, path):
    """
    Save an RGB image tensor of shape [1,3,H,W] to the specified path.
    
    Args:
        rgb_tensor (torch.Tensor): RGB tensor with shape [1,3,H,W]
        path (str): Path where the image will be saved
    """
    rgb = rgb_tensor.detach().cpu()  # shape: [3, H, W]
    
    # Permute to [H, W, 3]
    rgb = rgb.permute(1, 2, 0).numpy()
    
    # Ensure values are in [0, 255] range
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    
    # Convert to RGB if needed (e.g., in case alpha sneaks in)
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    
    # Save as RGB (PNG supports RGBA but JPEG doesn’t)
    img = Image.fromarray(rgb)
    if path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
        img = img.convert('RGB')  # drop alpha if any
    img.save(path)