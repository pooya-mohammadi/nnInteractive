from concurrent.futures import ThreadPoolExecutor
from time import time
from typing import List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, insert_crop_into_image, \
    crop_and_pad_nd
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from torch.nn.functional import interpolate

from nnInteractive.inference.nnInteractiveInferenceSessionV2 import nnInteractiveInferenceSessionV2
from nnInteractive.utils.bboxes import cover_structure_with_bboxes, get_bboxes_and_prios_from_image
from nnInteractive.utils.crop import crop_and_pad_into_buffer
from nnInteractive.utils.erosion_dilation import same_padding_pool3d


class nnInteractiveInferenceSessionV3(nnInteractiveInferenceSessionV2):
    def __init__(self,
                 device: torch.device = torch.device('cuda'),
                 use_torch_compile: bool = False,
                 verbose: bool = False,
                 torch_n_threads: int = 16,
                 interaction_decay: float = 0.9,
                 use_background_preprocessing: bool = True,
                 do_prediction_propagation: bool = True,
                 use_pinned_memory: bool = True,
                 verbose_run_times: bool = False
                 ):
        super().__init__(device, use_torch_compile, verbose, torch_n_threads, interaction_decay,
                         use_background_preprocessing, do_prediction_propagation, use_pinned_memory)
        self.new_interaction_zoom_out_factors: List[float] = []
        del self.new_interaction_bboxes
        self.new_interaction_centers = []
        self.has_positive_bbox = False
        self.verbose_run_times = verbose_run_times

        # Create a thread pool executor for background tasks.
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.preprocess_future = None
        self.interactions_future = None

    def set_image(self, image: np.ndarray, image_properties: dict):
        """
        Image must be 4D to satisfy nnU-Net needs: [c, x, y, z]
        Offload the processing to a background thread.
        """
        self._reset_session()
        assert image.ndim == 4, f'expected a 4d image as input, got {image.ndim}d. Shape {image.shape}'
        if self.verbose:
            print(f'Initialize with raw image shape {image.shape}')

        # Offload all image preprocessing to a background thread.
        self.preprocess_future = self.executor.submit(self._background_set_image, image, image_properties)
        self.input_image_shape = image.shape

    def _background_set_image(self, image: np.ndarray, image_properties: dict):
        # Convert and clone the image tensor.
        image_torch = torch.clone(torch.from_numpy(image))

        # Crop to nonzero region.
        if self.verbose:
            print('Cropping input image to nonzero region')
        nonzero_idx = torch.where(image_torch != 0)
        # Create bounding box: for each dimension, get the min and max (plus one) of the nonzero indices.
        bbox = [[i.min().item(), i.max().item() + 1] for i in nonzero_idx]
        del nonzero_idx
        slicer = bounding_box_to_slice(bbox)  # Assuming this returns a tuple of slices.
        image_torch = image_torch[slicer].float()
        if self.verbose:
            print(f'Cropped image shape: {image_torch.shape}')

        # As soon as we have the target shape, start initializing the interaction tensor in its own thread.
        self.interactions_future = self.executor.submit(self._initialize_interactions, image_torch)

        # Normalize the cropped image.
        if self.verbose:
            print('Normalizing cropped image')
        image_torch -= image_torch.mean()
        image_torch /= image_torch.std()

        self.preprocessed_image = image_torch
        if self.use_pinned_memory and self.device.type == 'cuda':
            if self.verbose:
                print('Pin memory: image')
            # Note: pin_memory() in PyTorch typically returns a new tensor.
            self.preprocessed_image = self.preprocessed_image.pin_memory()

        self.preprocessed_props = {'bbox_used_for_cropping': bbox[1:]}

        # we need to wait for this here I believe
        self.interactions_future.result()
        del self.interactions_future
        self.interactions_future = None

    def _initialize_interactions(self, image_torch: torch.Tensor):
        if self.verbose:
            print(f'Initialize interactions. Pinned: {self.use_pinned_memory}')
        # Create the interaction tensor based on the target shape.
        self.interactions = torch.zeros(
            (7, *image_torch.shape[1:]),
            device='cpu',
            dtype=torch.float16,
            pin_memory=(self.device.type == 'cuda' and self.use_pinned_memory)
        )

    def _reset_session(self):
        super()._reset_session()
        self.interactions_future = None
        self.preprocess_future = None
        self.input_image_shape = None

    def _finish_preprocessing_and_initialize_interactions(self):
        """
        Block until both the image preprocessing and the interactions tensor initialization
        are finished.
        """
        if self.preprocess_future is not None:
            # Wait for image preprocessing to complete.
            self.preprocess_future.result()
            del self.preprocess_future
            self.preprocess_future = None

    def reset_interactions(self):
        super().reset_interactions()
        self.has_positive_bbox = False

    @torch.inference_mode
    # @benchmark_decorator
    def _predict(self):
        assert len(self.new_interaction_centers) == len(self.new_interaction_zoom_out_factors)
        if len(self.new_interaction_centers) > 1:
            print('It seems like more than one interaction was added since the last prediction. This is not '
                  'recommended and may cause unexpected behavior or inefficient predictions')

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            for prediction_center, initial_zoom_out_factor in zip(self.new_interaction_centers, self.new_interaction_zoom_out_factors):
                # make a prediction at initial zoom out factor. If more zoom out is required, do this until the
                # entire object fits the FOV. Then go back to original resolution and refine.

                # we need this later.
                start = time()
                previous_prediction = torch.clone(self.interactions[0])
                if self.verbose_run_times: print('Time clone prev prediction', time() - start)

                if not self.do_prediction_propagation:
                    initial_zoom_out_factor = 1

                initial_zoom_out_factor = min(initial_zoom_out_factor, 4)
                zoom_out_factor = initial_zoom_out_factor
                max_zoom_out_factor = initial_zoom_out_factor

                while zoom_out_factor is not None and zoom_out_factor <= 4:
                    print('Performing prediction at zoom out factor', zoom_out_factor)
                    max_zoom_out_factor = max(max_zoom_out_factor, zoom_out_factor)
                    # initial prediction at initial_zoom_out_factor
                    scaled_patch_size = [round(i * zoom_out_factor) for i in self.configuration_manager.patch_size]
                    scaled_bbox = [[c - p // 2, c + p // 2 + p % 2] for c, p in zip(prediction_center, scaled_patch_size)]

                    start = time()
                    crop_img = crop_and_pad_nd(self.preprocessed_image, scaled_bbox, pad_mode=self.pad_mode_data)
                    crop_interactions = crop_and_pad_nd(self.interactions, scaled_bbox)
                    if self.verbose_run_times: print(f'Time crop of size {scaled_patch_size}', time() - start)

                    # resize input_for_predict (which may be larger than patch size) to patch size
                    # this implementation may not seem straightforward but it does save VRAM which is crucial here
                    if not all([i == j for i, j in zip(self.configuration_manager.patch_size, crop_interactions.shape[-3:])]):
                        start = time()
                        crop_interactions_resampled_gpu = torch.empty((7, *self.configuration_manager.patch_size), dtype=torch.float16, device=self.device)
                        # previous seg, bbox+, bbox-
                        for i in range(0, 3):
                            # this is area for a reason but I aint telling ya why
                            crop_interactions_resampled_gpu[i] = interpolate(crop_interactions[i][None, None].to(self.device), self.configuration_manager.patch_size, mode='area')[0][0]
                        empty_cache(self.device)
                        if self.verbose_run_times: print(f'Time resample interactions 0-3', time() - start)

                        max_pool_ks = round(zoom_out_factor * 2 - 1)
                        # point+, point-, scribble+, scribble-
                        time_mp = []
                        time_interpolate = []
                        for i in range(3, 7):
                            tmp = crop_interactions[i].to(self.device)
                            if max_pool_ks > 1:
                                # dilate to preserve interactions after downsampling
                                start = time()
                                tmp = same_padding_pool3d(tmp[None, None], max_pool_ks)[0, 0]
                                time_mp.append(time() - start)
                            start = time()
                            # this is area for a reason but I aint telling ya why
                            crop_interactions_resampled_gpu[i] = interpolate(tmp[None, None], self.configuration_manager.patch_size, mode='area')[0][0]
                            time_interpolate.append(time() - start)
                            del tmp
                        if self.verbose_run_times: print(f'Time max poolings (avg)', np.mean(time_mp))
                        if self.verbose_run_times: print(f'Time resampling interactions 4-7', np.mean(time_interpolate))
                        start = time()
                        crop_img = interpolate(crop_img[None].to(self.device), self.configuration_manager.patch_size, mode='trilinear')[0]
                        if self.verbose_run_times: print(f'Time resample image', time() - start)
                        crop_interactions = crop_interactions_resampled_gpu
                        del crop_interactions_resampled_gpu
                        empty_cache(self.device)
                    else:
                        start = time()
                        crop_img = crop_img.to(self.device, non_blocking=self.device.type == 'cuda')
                        crop_interactions = crop_interactions.to(self.device, non_blocking=self.device.type == 'cuda')
                        if self.verbose_run_times: print('Time no resampling just copy to GPU', time() - start)

                    start = time()
                    input_for_predict = torch.cat((crop_img, crop_interactions))
                    if self.verbose_run_times: print('Time cat', time() - start)
                    del crop_img, crop_interactions

                    start = time()
                    pred = self.network(input_for_predict[None])[0].argmax(0).detach()
                    if self.verbose_run_times: print("Time predict", time() - start)

                    del input_for_predict

                    # detect changes at borders
                    previous_zoom_prediction = crop_and_pad_nd(self.interactions[0], scaled_bbox).to(self.device)
                    if not all([i == j for i, j in zip(pred.shape, previous_zoom_prediction.shape)]):
                        start = time()
                        previous_zoom_prediction = interpolate(previous_zoom_prediction[None, None].to(float), pred.shape, mode='nearest')[0, 0]
                        if self.verbose_run_times: print('Time resampling prev prediction', time() - start)

                    abs_pxl_change_threshold = 1500
                    rel_pxl_change_threshold = 0.2
                    min_pxl_change_threshold = 100
                    continue_zoom = False
                    if zoom_out_factor < 4 and self.do_prediction_propagation:
                        start = time()
                        for dim in range(len(scaled_bbox)):
                            for idx in [0, pred.shape[dim] - 1]:
                                slice_prev = previous_zoom_prediction.index_select(dim, torch.tensor(idx, device=self.device))
                                slice_curr = pred.index_select(dim, torch.tensor(idx, device=self.device))
                                pixels_prev = torch.sum(slice_prev)
                                pixels_current = torch.sum(slice_curr)
                                pixels_diff = torch.sum(slice_prev != slice_curr)
                                rel_change = max(pixels_prev, pixels_current) / max(min(pixels_prev, pixels_current),
                                                                                    1e-5) - 1
                                if pixels_diff > abs_pxl_change_threshold:
                                    continue_zoom = True
                                    if self.verbose:
                                        print(f'continue zooming because change at borders of {pixels_diff} > {abs_pxl_change_threshold}')
                                if pixels_diff > min_pxl_change_threshold and rel_change > rel_pxl_change_threshold:
                                    continue_zoom = True
                                    if self.verbose:
                                        print(f'continue zooming because relative change of {rel_change} > {rel_pxl_change_threshold} and n_pixels {pixels_diff} > {min_pxl_change_threshold}')
                                del slice_prev, slice_curr, pixels_prev, pixels_current, pixels_diff

                        del previous_zoom_prediction
                        if self.verbose_run_times: print('Time detect change', time() - start)

                    # resize prediction to correct size and place in target buffer + interactions
                    start = time()
                    if not all([i == j for i, j in zip(pred.shape, scaled_patch_size)]):
                        pred = (interpolate(pred[None, None].to(float), scaled_patch_size, mode='trilinear')[0, 0] >= 0.5).to(torch.uint8)
                    pred = pred.cpu()
                    if self.verbose_run_times: print('Time resize prediction and copy to CPU', time() - start)

                    # if we do not continue zooming we need a difference map for sampling patches
                    if not continue_zoom and zoom_out_factor > 1:
                        # wow this circus saves ~30ms relative to naive implementation
                        start = time()
                        seen_bbox = [[max(0, i[0]), min(i[1], s)] for i, s in zip(scaled_bbox, previous_prediction.shape)]
                        bbox_tmp = [[i[0] - s[0], i[1] - s[0]] for i, s in zip(seen_bbox, scaled_bbox)]
                        bbox_tmp = [[max(0, i[0]), min(i[1], s)] for i, s in zip(bbox_tmp, scaled_patch_size)]
                        slicer = bounding_box_to_slice(seen_bbox)
                        slicer2 = bounding_box_to_slice(bbox_tmp)
                        diff_map = pred[slicer2] != previous_prediction[slicer]
                        # dont allocate new memory, just reuse previous_prediction
                        previous_prediction.zero_()
                        diff_map = insert_crop_into_image(previous_prediction, diff_map, seen_bbox)
                        if self.verbose_run_times: print('Time create diff_map', time() - start)

                        # erode the difference map to keep computational load in check
                        start = time()
                        eroded = same_padding_pool3d(diff_map[slicer][None, None].to(self.device).half(), kernel_size=5, use_min_pool=True)[0, 0]
                        has_diff = torch.any(eroded)
                        diff_map[slicer] = eroded.cpu()
                        if self.verbose_run_times: print('Time erode diff map', time() - start)

                        del previous_prediction, eroded
                    else:
                        has_diff = False

                    if zoom_out_factor == 1 or (not continue_zoom and has_diff): # rare case where no changes are needed because of useless interaction. Need to check for not continue_zoom because otherwise diff_map wint exist
                        start = time()
                        insert_crop_into_image(self.interactions[0], pred, scaled_bbox)

                        # place into target buffer
                        bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                                zip(scaled_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
                        insert_crop_into_image(self.target_buffer,
                                               pred if isinstance(self.target_buffer, torch.Tensor) else pred.numpy(), bbox)
                        if self.verbose_run_times: print('Time insert prediction', time() - start)
                    del pred

                    empty_cache(self.device)

                    if continue_zoom:
                        zoom_out_factor *= 1.5
                        zoom_out_factor = min(4, zoom_out_factor)
                    else:
                        zoom_out_factor = None

                if max_zoom_out_factor > 1 and has_diff:
                    # only use the region that was previously looked at. Use last scaled_bbox
                    if self.has_positive_bbox:
                        start = time()
                        # mask positive bbox channel with dilated current segmentation to avoid bbox nonsense.
                        # Basically convert bbox to lasso
                        pos_bbox_idx = -6
                        # dilated_pred = same_padding_maxpool3d(self.interactions[0][None, None].to(self.device), 3)[0, 0]
                        self.interactions[pos_bbox_idx][(~(self.interactions[0] > 0.5)).cpu()] = 0
                        self.has_positive_bbox = False
                        if self.verbose_run_times: print('Time for bbox to lasso (maybe)', time() - start)

                    start = time()
                    nonzero_indices = torch.nonzero(self.interactions[0], as_tuple=False)
                    mn = torch.min(nonzero_indices, dim=0)[0]
                    mx = torch.max(nonzero_indices, dim=0)[0]
                    roi = [[i.item(), x.item() + 1] for i, x in zip(mn, mx)]
                    if self.verbose_run_times: print('Time nonzero indices', time() - start)
                    start = time()
                    bboxes, prios = get_bboxes_and_prios_from_image(diff_map,
                                                                    roi,
                                                                    [round(i // (1 / 0.7)) for i in # i // (1 / 0.7)
                                                                     self.configuration_manager.patch_size], # the smaller this is the more precise we are at the cost of more patches
                                                                    self.configuration_manager.patch_size)
                    if self.verbose_run_times: print('Time get_bboxes_and_prios_from_image', time() - start)

                    # # place pseudoprompts because we can and its fast af
                    # doesnt work too well unfortunately. I don't know why
                    # random_points = len(bboxes) * 2
                    # for i in range(random_points):
                    #     random_coord = nonzero_indices[np.random.choice(len(nonzero_indices))]
                    #     self.point_interaction.place_point(random_coord, self.interactions[-4])

                    del diff_map, nonzero_indices
                    start = time()
                    bboxes_ordered = cover_structure_with_bboxes(prios, bboxes,
                                                                 np.mean(self.configuration_manager.patch_size) / 2,
                                                                 verbose=False) if len(prios) > 0 else []
                    if self.verbose_run_times: print('Time cover_structure_with_bboxes', time() - start)

                    print(f'Using {len(bboxes_ordered)} bounding boxes for refinement')

                    preallocated_input = torch.zeros((8, *self.configuration_manager.patch_size), device=self.device, dtype=torch.float)
                    for nref, refinement_bbox in enumerate(bboxes_ordered):
                        start = time()
                        assert self.pad_mode_data == 'constant'
                        crop_and_pad_into_buffer(preallocated_input[0], refinement_bbox, self.preprocessed_image[0])
                        crop_and_pad_into_buffer(preallocated_input[1:], refinement_bbox, self.interactions)

                        pred = self.network(preallocated_input[None])[0].argmax(0).detach().cpu()

                        insert_crop_into_image(self.interactions[0], pred, refinement_bbox)
                        # place into target buffer
                        bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in zip(refinement_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
                        insert_crop_into_image(self.target_buffer, pred if isinstance(self.target_buffer, torch.Tensor) else pred.numpy(), bbox)
                        del pred
                        preallocated_input.zero_()
                        if self.verbose_run_times: print(f'Time refinement loop {nref}, bbox {refinement_bbox}', time() - start)
                    del preallocated_input
                    empty_cache(self.device)

        self.new_interaction_centers = []
        self.new_interaction_zoom_out_factors = []
        empty_cache(self.device)

    def _add_patch_for_point_interaction(self, coordinates):
        self.new_interaction_zoom_out_factors.append(1)
        self.new_interaction_centers.append(coordinates)
        print(f'Added new point interaction: center {self.new_interaction_zoom_out_factors[-1]}, scale {self.new_interaction_centers}')

    def _add_patch_for_bbox_interaction(self, bbox):
        bbox_center = [round((i[0] + i[1]) / 2) for i in bbox]
        bbox_size = [i[1]-i[0] for i in bbox]
        # we want to see some context, so the crop we see for the initial prediction should be patch_size / 3 larger
        requested_size = [i + j // 3 for i, j in zip(bbox_size, self.configuration_manager.patch_size)]
        self.new_interaction_zoom_out_factors.append(max(1, max([i / j for i, j in zip(requested_size, self.configuration_manager.patch_size)])))
        self.new_interaction_centers.append(bbox_center)
        print(f'Added new bbox interaction: center {self.new_interaction_zoom_out_factors[-1]}, scale {self.new_interaction_centers}')

    def _add_patch_for_scribble_interaction(self, scribble_image):
        return self._generic_add_patch_from_image(scribble_image)

    def _add_patch_for_lasso_interaction(self, lasso_image):
        return self._generic_add_patch_from_image(lasso_image)

    def _add_patch_for_initial_seg_interaction(self, initial_seg):
        return self._generic_add_patch_from_image(initial_seg)

    def add_bbox_interaction(self, bbox_coords, include_interaction: bool, run_prediction: bool = True) -> np.ndarray:
        if include_interaction:
            self.has_positive_bbox = True
        return super().add_bbox_interaction(bbox_coords, include_interaction, run_prediction)

    # @benchmark_decorator
    def _generic_add_patch_from_image(self, image: torch.Tensor):
        if not torch.any(image):
            print('Received empty image prompt. Cannot add patches for prediction')
            return
        nonzero_indices = torch.nonzero(image, as_tuple=False)
        mn = torch.min(nonzero_indices, dim=0)[0]
        mx = torch.max(nonzero_indices, dim=0)[0]
        roi = [[i.item(), x.item() + 1] for i, x in zip(mn, mx)]
        roi_center = [round((i[0] + i[1]) / 2) for i in roi]
        roi_size = [i[1]- i[0] for i in roi]
        requested_size = [i + j // 3 for i, j in zip(roi_size, self.configuration_manager.patch_size)]
        self.new_interaction_zoom_out_factors.append(max(1, max([i / j for i, j in zip(requested_size, self.configuration_manager.patch_size)])))
        self.new_interaction_centers.append(roi_center)
        print(f'Added new image interaction: scale {self.new_interaction_zoom_out_factors[-1]}, center {self.new_interaction_centers}')

