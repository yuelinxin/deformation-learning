import numpy as np
import noise
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms


def generate_deformation_field(image_shape, scale=0.02, magnitude=20, seed=None):
    height, width = image_shape
    dx_field = np.zeros(image_shape)
    dy_field = np.zeros(image_shape)

    if seed is not None:
        np.random.seed(seed)
        base_x, base_y = np.random.randint(0, 10000, 2)
    else:
        base_x, base_y = 0, 0

    for y in range(height):
        for x in range(width):
            dx_field[y, x] = noise.pnoise2((x + base_x) * scale, (y + base_y) * scale, 
                                           repeatx=width, repeaty=height) * magnitude
            dy_field[y, x] = noise.pnoise2((x + base_x + 100) * scale, (y + base_y + 100) * scale, 
                                           repeatx=width, repeaty=height) * magnitude

    return dx_field, dy_field


def apply_deformation_to_batch(images, dx, dy):
    batch, channels, height, width = images.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    deformed_images = torch.zeros_like(images)
    for i in range(batch):
        x_deformed = np.clip(x + dx[i], 0, width - 1).astype(int)
        y_deformed = np.clip(y + dy[i], 0, height - 1).astype(int)
        for c in range(channels):
            deformed_images[i, c] = images[i, c, y_deformed, x_deformed]
    return deformed_images.to(images.device)


def generate_deformed_images(images, random=True):
    B, _, H, W = images.shape
    dx, dy = [], []
    for _ in range(B):
        if random:
            seed1, seed2 = np.random.randint(0, 10000, 2)
        else:
            seed1, seed2 = None, None
        dx1, dy1 = generate_deformation_field((H, W), scale=0.01, magnitude=50, seed=seed1)
        dx2, dy2 = generate_deformation_field((H, W), scale=0.03, magnitude=10, seed=seed2)
        dx.append(dx1 + dx2)
        dy.append(dy1 + dy2)
    dx = np.stack(dx)
    dy = np.stack(dy)
    
    deformed_images = apply_deformation_to_batch(images, dx, dy)
    
    return deformed_images


def apply_individual_deformation(images, deformations, random=True):
    """
    Apply a unique deformation to each image in the batch.
    :param images: Tensor of shape [B, C, H, W]
    :param deformations: List of tuples of (dx, dy) fields
    :return: Deformed images
    """
    B, C, H, W = images.size()
    deformed_images = torch.zeros_like(images)
    
    for i in range(B):
        if random:
            # Randomly select a deformation for each image
            deformation_idx = torch.randint(0, len(deformations), (1,)).item()
        else:
            # Cycle through selected deformations
            deformation_idx = i % len(deformations)

        dx, dy = deformations[deformation_idx]
        dx = torch.Tensor(dx).to(images.device)
        dy = torch.Tensor(dy).to(images.device)
        
        # Convert dx, dy to flow fields
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        grid_x = grid_x.to(images.device)
        grid_y = grid_y.to(images.device)
        flow_field = torch.stack((grid_x + 2*dx/W, grid_y + 2*dy/H), dim=2)  # Scale dx, dy to [-1, 1]
        
        # Apply deformation using grid_sample
        deformed_image = F.grid_sample(images[i].unsqueeze(0), flow_field.unsqueeze(0), mode='bilinear', padding_mode='border', align_corners=True)
        deformed_images[i] = deformed_image.squeeze(0)
    
    return deformed_images


def apply_deformation(image, deformations):
    """
    Apply a random deformation to an image.
    :param image: Tensor of shape [C, H, W]
    :param deformations: List of tuples of (dx, dy) fields
    :return: Deformed image
    """
    C, H, W = image.size()
    
    deformation_idx = torch.randint(0, len(deformations), (1,)).item()
    dx, dy = deformations[deformation_idx]
    dx = torch.Tensor(dx).to(image.device)
    dy = torch.Tensor(dy).to(image.device)

    # Convert dx, dy to flow fields
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    grid_x = grid_x.to(image.device)
    grid_y = grid_y.to(image.device)

    # Normalize dx and dy to be in the range [-1, 1]
    dx_normalized = 2. * dx / (W - 1)
    dy_normalized = 2. * dy / (H - 1)

    # Create the deformed grid by adding the normalized dx and dy to the original grid positions
    deformed_grid_x = grid_x + dx_normalized
    deformed_grid_y = grid_y + dy_normalized

    # Stack to get the flow field in the expected format [N, H, W, 2]
    flow_field = torch.stack((deformed_grid_x, deformed_grid_y), dim=-1)

    # Apply the deformation using grid_sample
    # Add an extra batch dimension to image and flow field to match grid_sample's input shape requirements
    deformed_image = F.grid_sample(image.unsqueeze(0), flow_field.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False)

    # Remove the added batch dimension
    return deformed_image.squeeze(0)
