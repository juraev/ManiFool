import torch
import torch.nn.functional as F

def proj_warp_gpu(H_gpu, img_gpu):
    """
    Applies the projective transformation using the transformation matrix H_gpu.

    :Tensor H_gpu: 3x3 transformation matrix
    :Tensor img_gpu: the image to be transformed (C x H x W)

    :returns: transformed image (C x H x W)
    """
    if not isinstance(img_gpu, torch.Tensor):
        raise TypeError(f"Expected img_gpu to be a torch.Tensor but got {type(img_gpu)}")
    if not isinstance(H_gpu, torch.Tensor):
        raise TypeError(f"Expected H_gpu to be a torch.Tensor but got {type(H_gpu)}")

    if img_gpu.dim() != 3:
        raise ValueError("img_gpu should have shape (C, H, W)")
    nChannels, ny, nx = img_gpu.shape

    device = img_gpu.device

    # Create grid of coordinates
    x = torch.arange(nx, device=device)
    y = torch.arange(ny, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # grid_x, grid_y shape (ny, nx)
    coords = torch.stack([grid_x, grid_y], dim=2).float()  # shape (ny, nx, 2)
    coords_flat = coords.view(-1, 2)  # shape (ny * nx, 2)
    ones = torch.ones(coords_flat.size(0), 1, device=device)
    coords_hom = torch.cat([coords_flat, ones], dim=1).t()  # shape (3, ny * nx)

    # Apply homography
    source_coords_hom = H_gpu @ coords_hom  # shape (3, ny * nx)
    source_coords = source_coords_hom[:2, :] / source_coords_hom[2:, :]  # shape (2, ny * nx)
    source_coords = source_coords.t().view(ny, nx, 2)  # shape (ny, nx, 2)

    # Normalize coordinates to [-1, 1]
    source_coords_normalized = torch.zeros_like(source_coords)
    source_coords_normalized[..., 0] = 2.0 * source_coords[..., 0] / (nx - 1) - 1.0
    source_coords_normalized[..., 1] = 2.0 * source_coords[..., 1] / (ny - 1) - 1.0

    # Adjust for grid_sample coordinate system (x, y)
    grid = source_coords_normalized.unsqueeze(0)  # shape (1, ny, nx, 2)
    img_gpu = img_gpu.unsqueeze(0)  # shape (1, C, ny, nx)

    # Perform grid sampling
    warped_img = F.grid_sample(img_gpu, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    warped_img = warped_img.squeeze(0)  # shape (C, ny, nx)

    return warped_img