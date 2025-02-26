import torch
import torch.nn.functional as F

from nnunetv2.utilities.helpers import empty_cache
from torch.backends import cudnn


@torch.inference_mode
def same_padding_pool3d(x, kernel_size, use_min_pool: bool = False):
    """
    Applies 3D max pooling with manual asymmetric padding such that
    the output shape is the same as the input shape.

    Args:
        x (Tensor): Input tensor of shape (N, C, D, H, W)
        kernel_size (int or tuple): Kernel size for the pooling.
            If int, the same kernel size is used for all three dimensions.

    Returns:
        Tensor: Output tensor with the same (D, H, W) dimensions as the input.
    """
    benchmark = cudnn.benchmark
    cudnn.benchmark = False
    # Convert kernel_size to tuple if it's an int.
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)

    kD, kH, kW = kernel_size

    # Compute asymmetric padding for each dimension:
    pad_d_front = (kD - 1) // 2
    pad_d_back = (kD - 1) - pad_d_front

    pad_h_top = (kH - 1) // 2
    pad_h_bottom = (kH - 1) - pad_h_top

    pad_w_left = (kW - 1) // 2
    pad_w_right = (kW - 1) - pad_w_left

    # For 3D (input shape: [N, C, D, H, W]), F.pad expects the padding in the order:
    # (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    x_padded = F.pad(x, (pad_w_left, pad_w_right,
                         pad_h_top, pad_h_bottom,
                         pad_d_front, pad_d_back), mode='replicate')
    del x

    # Apply max pooling with no additional padding.
    if not use_min_pool:
        ret = F.max_pool3d(x_padded, kernel_size=kernel_size, stride=1, padding=0)
        empty_cache(x_padded.device)
        cudnn.benchmark = benchmark
        return ret
    else:
        x_padded *= -1
        x_padded = - F.max_pool3d(x_padded, kernel_size=kernel_size, stride=1, padding=0)
        empty_cache(x_padded.device)
        cudnn.benchmark = benchmark
        return x_padded
