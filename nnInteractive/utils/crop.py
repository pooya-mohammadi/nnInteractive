from typing import Sequence

import torch


def crop_and_pad_into_buffer(target_tensor: torch.Tensor,
                             bbox: Sequence[Sequence[int]],
                             source_tensor: torch.Tensor) -> None:
    """
    Copies a sub-region from source_tensor into target_tensor based on a bounding box.

    Args:
        target_tensor (torch.Tensor): A preallocated tensor that will be updated.
        bbox (sequence of [int, int]): A bounding box for each dimension of the source tensor
            that is covered by the bbox. The bbox is defined as [start, end) (half-open interval)
            and may extend outside the source tensor. If source_tensor has more dimensions than
            len(bbox), the leading dimensions will be fully included.
        source_tensor (torch.Tensor): The tensor to copy data from.

    Behavior:
        For each dimension that the bbox covers (i.e. the last len(bbox) dims of source_tensor):
            - Compute the overlapping region between the bbox and the source tensor.
            - Determine the corresponding indices in the target tensor where the data will be copied.
        For any extra leading dimensions (i.e. source_tensor.ndim > len(bbox)):
            - Use slice(None) to include the entire dimension.
        If source_tensor and target_tensor are on different devices, only the overlapping subregion
        is transferred to the device of target_tensor.
    """
    total_dims = source_tensor.ndim
    bbox_dims = len(bbox)
    # Compute the number of leading dims that are not covered by bbox.
    leading_dims = total_dims - bbox_dims

    source_slices = []
    target_slices = []

    # For the leading dimensions, include the entire dimension.
    for _ in range(leading_dims):
        source_slices.append(slice(None))
        target_slices.append(slice(None))

    # Process the dimensions covered by the bbox.
    for d in range(bbox_dims):
        box_start, box_end = bbox[d]
        d_source = leading_dims + d
        source_size = source_tensor.shape[d_source]

        # Compute the overlapping region in source coordinates.
        copy_start_source = max(box_start, 0)
        copy_end_source = min(box_end, source_size)
        copy_size = copy_end_source - copy_start_source

        # Compute the corresponding indices in the target tensor.
        copy_start_target = max(0, -box_start)
        copy_end_target = copy_start_target + copy_size

        source_slices.append(slice(copy_start_source, copy_end_source))
        target_slices.append(slice(copy_start_target, copy_end_target))

    # Extract the overlapping region from the source.
    sub_source = source_tensor[tuple(source_slices)]
    # Transfer only this subregion to the target tensor's device.
    sub_source = sub_source.to(target_tensor.device)
    # Write the data into the preallocated target_tensor.
    target_tensor[tuple(target_slices)] = sub_source