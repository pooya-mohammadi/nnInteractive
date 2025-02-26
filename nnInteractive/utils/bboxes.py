from time import time
from typing import List, Union
import numpy as np
import torch
import math
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd


def bounding_box_center_distance(bbox1: List[List[float]], bbox2: List[List[float]]) -> float:
    """
    Computes the Euclidean distance between the centers of two bounding boxes.

    Parameters:
        bbox1 (List[List[float]]): First bounding box, specified as [[x1, x2], [y1, y2], ...].
        bbox2 (List[List[float]]): Second bounding box, specified as [[x1, x2], [y1, y2], ...].

    Returns:
        float: Euclidean distance between the centers of the two bounding boxes.
    """
    if len(bbox1) != len(bbox2):
        raise ValueError("Bounding boxes must have the same dimensionality.")

    # Compute the center of each bounding box
    center1 = [(dim[0] + dim[1]) / 2 for dim in bbox1]
    center2 = [(dim[0] + dim[1]) / 2 for dim in bbox2]

    # Compute the Euclidean distance between the centers
    distance = math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(center1, center2)))
    return distance


def minimum_distance_to_others(bbox, list_of_other_bboxes):
    d = 1e9
    for other in list_of_other_bboxes:
        d = min(d, bounding_box_center_distance(bbox, other))
    return d


def cover_structure_with_bboxes(priority_values, bboxes, neighbor_distance, verbose: bool = False):
    if not isinstance(priority_values, np.ndarray):
        priority_values = np.array(priority_values)
    assert len(priority_values) == len(bboxes)
    assert min(priority_values) > 0

    neighbor_idx = []
    not_visited_idx = list(range(len(priority_values)))
    return_order_idx = []

    while len(return_order_idx) < len(bboxes):
        if len(neighbor_idx) == 0:
            # pick the highest prio box from not_visited
            add = not_visited_idx[np.argmax(priority_values[not_visited_idx])]
            return_order_idx.append(add)
            not_visited_idx.remove(add)
        else:
            add = neighbor_idx[np.argmax(priority_values[neighbor_idx])]
            return_order_idx.append(add)
            neighbor_idx.remove(add)

        neighbors_here = []
        for nv in not_visited_idx:
            if bounding_box_center_distance(bboxes[nv], bboxes[add]) < neighbor_distance:
                neighbors_here.append(nv)
        for nh in neighbors_here:
            neighbor_idx.append(nh)
            not_visited_idx.remove(nh)
        # _add_neighbors(add)
    ret_bb = [bboxes[i] for i in return_order_idx]
    if verbose:
        print(ret_bb)
        print(priority_values[return_order_idx])
    return ret_bb


def get_bboxes_and_prios_from_image(image: torch.Tensor, roi: List[List[int]], bbox_size_inner, bbox_size_outer):
    """
    roi is EXCLUSIVE in upper coordinate! half open interval [x, y)
    """
    roi_size = [j - i for i, j in roi]
    assert image.ndim == 3

    # zip(image_size, target_step_sizes_in_voxels, tile_size)]
    num_steps = [int(np.ceil(max(0, i - j) / j)) + 2 for i, j, in zip(roi_size, bbox_size_inner)]

    steps = []
    for dim in range(len(roi_size)):
        if roi_size[dim] < bbox_size_inner[dim]:
            # no point in setting more than one step if the bbox_size_inner can be covered with one. Minimum
            # step amount of 2 doesn't apply here
            steps.append([round((roi[dim][1] - roi[dim][0]) / 2) + roi[dim][0]])
        else:
            # the highest step value for this dimension is
            max_step_value = roi_size[dim] - 1
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) + roi[dim][0] for i in range(num_steps[dim])]

            steps.append(steps_here)

    neg_offset_outer = [i // 2 for i in bbox_size_outer]
    pos_offset_outer = [i // 2 + i % 2 for i in bbox_size_outer]
    neg_offset_inner = [i // 2 for i in bbox_size_inner]
    pos_offset_inner = [i // 2 + i % 2 for i in bbox_size_inner]

    bboxes = []
    prios = []
    for sx in steps[0]:
        for sy in steps[1]:
            for sz in steps[2]:
                center = [sx, sy, sz]
                bbox_inner = [[center[i] - neg_offset_inner[i], center[i] + pos_offset_inner[i]] for i in range(image.ndim)]
                # float or we overfloat, yo
                prio = torch.sum(crop_and_pad_nd(image, bbox_inner), dtype=torch.float).item()
                if prio > 0:
                    bboxes.append(
                        [[center[i] - neg_offset_outer[i], center[i] + pos_offset_outer[i]] for i in range(image.ndim)]
                    )
                    prios.append(prio)
    return bboxes, prios



def filter_bboxes(list_new_bboxes, list_other, min_distance_to_selected):
    # filters list_new_bboxes so that only bounding boxes that are further than min_distance_to_selected from any
    # of the bboxes in list_other (+ already selected new bounding boxes) are included.
    combined_other_new = list(list_other) # make a copy
    filtered_bboxes = []
    for new_bbox in list_new_bboxes:
        dist = minimum_distance_to_others(new_bbox, combined_other_new) if len(combined_other_new) > 0 else np.inf
        if dist > min_distance_to_selected:
            filtered_bboxes.append(new_bbox)
            combined_other_new.append(new_bbox)
    return filtered_bboxes


def prediction_propagation_add_bounding_boxes(previous_prediction_in_bbox: torch.Tensor,
                                              current_prediction_in_bbox: torch.Tensor,
                                              current_bbox: List[List[int]],
                                              new_bbox_distance: List[int],
                                              abs_pxl_change_threshold: int,
                                              rel_pxl_change_threshold: float,
                                              min_pxl_change_threshold: int):
    shifted_bboxes = []

    for dim in range(len(current_bbox)):
        # lower bound
        idx = 0
        slice_prev = previous_prediction_in_bbox.index_select(dim, torch.tensor(idx, device=previous_prediction_in_bbox.device))
        slice_curr = current_prediction_in_bbox.index_select(dim, torch.tensor(idx, device=current_prediction_in_bbox.device))
        pixels_prev = torch.sum(slice_prev)
        pixels_current = torch.sum(slice_curr)
        pixels_diff = torch.sum(slice_prev != slice_curr)
        rel_change = max(pixels_prev, pixels_current) / max(min(pixels_prev, pixels_current), 1e-5) - 1

        if pixels_diff > min_pxl_change_threshold and (pixels_diff > abs_pxl_change_threshold or rel_change > rel_pxl_change_threshold):
            new_bbox = [list(coord) for coord in current_bbox]  # Create a copy of the original bbox
            new_bbox[dim][0] -= new_bbox_distance[dim]
            new_bbox[dim][1] -= new_bbox_distance[dim]  # Maintain the original size
            shifted_bboxes.append(new_bbox)
            # print(dim)
            # print('pixels_diff', {pixels_diff})
            # print('rel_change', {rel_change})

        # upper bound
        idx = current_prediction_in_bbox.shape[dim] - 1
        slice_prev = previous_prediction_in_bbox.index_select(dim, torch.tensor(idx, device=previous_prediction_in_bbox.device))
        slice_curr = current_prediction_in_bbox.index_select(dim, torch.tensor(idx, device=current_prediction_in_bbox.device))
        pixels_prev = torch.sum(slice_prev)
        pixels_current = torch.sum(slice_curr)
        pixels_diff = torch.sum(slice_prev != slice_curr)
        rel_change = max(pixels_prev, pixels_current) / min(pixels_prev, pixels_current) - 1

        if pixels_diff > min_pxl_change_threshold and (pixels_diff > abs_pxl_change_threshold or rel_change > rel_pxl_change_threshold):
            new_bbox = [list(coord) for coord in current_bbox]  # Create a copy of the original bbox
            new_bbox[dim][0] += new_bbox_distance[dim]
            new_bbox[dim][1] += new_bbox_distance[dim]  # Maintain the original size
            shifted_bboxes.append(new_bbox)
            # print(dim)
            # print('pixels_diff', {pixels_diff})
            # print('rel_change', {rel_change})
    return shifted_bboxes



if __name__ == '__main__':
    times = []
    for _ in range(50):
        a = torch.round(torch.rand((192, 192, 192)) * 0.51)
        st = time()
        torch.any(a)
        end = time()
        times.append(end - st)
