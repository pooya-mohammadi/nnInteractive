import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, \
    crop_and_pad_nd
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from torch.nn.functional import interpolate

from nnInteractive.inference.nnInteractiveInferenceSessionV3 import nnInteractiveInferenceSessionV3
from nnInteractive.utils.bboxes import generate_bounding_boxes
from nnInteractive.utils.crop import crop_and_pad_into_buffer, crop_to_valid, pad_cropped, paste_tensor
from nnInteractive.utils.erosion_dilation import iterative_3x3_same_padding_pool3d
from nnInteractive.utils.rounding import round_to_nearest_odd
from time import time


class nnInteractiveInferenceSessionV4(nnInteractiveInferenceSessionV3):
    @torch.inference_mode
    def _predict(self):
        """
        This function is a smoking mess to read. This is deliberate. Initially it was super pretty and easy to
        understand. Then the run time optimization began.
        If it feels like we are excessively transferring tensors between CPU and GPU, this is deliberate as well.
        Our goal is to keep this tool usable even for people with smaller GPUs (8-10GB VRAM). In an ideal world
        everyone would have 24GB+ of VRAM and all tensors would like on GPU all the time.
        The amount of hours spent optimizing this function is substantial. Almost every line was turned and twisted
        multiple times. If something appears odd, it is probably so for a reason. Don't change things all willy nilly
        without first understanding what is going on. And don't make changes without verifying that the run time or
        VRAM consumption is not adversely affected.

        Returns:

        """
        assert self.pad_mode_data == 'constant', 'pad modes other than constant are not implemented here'
        assert len(self.new_interaction_centers) == len(self.new_interaction_zoom_out_factors)
        if len(self.new_interaction_centers) > 1:
            print('It seems like more than one interaction was added since the last prediction. This is not '
                  'recommended and may cause unexpected behavior or inefficient predictions')

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            for prediction_center, initial_zoom_out_factor in zip(self.new_interaction_centers, self.new_interaction_zoom_out_factors):
                # make a prediction at initial zoom out factor. If more zoom out is required, do this until the
                # entire object fits the FOV. Then go back to original resolution and refine.

                # we need this later.
                previous_prediction = torch.clone(self.interactions[0])

                if not self.do_prediction_propagation:
                    initial_zoom_out_factor = 1

                initial_zoom_out_factor = min(initial_zoom_out_factor, 4)
                zoom_out_factor = initial_zoom_out_factor
                max_zoom_out_factor = initial_zoom_out_factor

                start = time()
                while zoom_out_factor is not None and zoom_out_factor <= 4:
                    print('Performing prediction at zoom out factor', zoom_out_factor)
                    max_zoom_out_factor = max(max_zoom_out_factor, zoom_out_factor)
                    # initial prediction at initial_zoom_out_factor
                    scaled_patch_size = [round(i * zoom_out_factor) for i in self.configuration_manager.patch_size]
                    scaled_bbox = [[c - p // 2, c + p // 2 + p % 2] for c, p in zip(prediction_center, scaled_patch_size)]

                    crop_img, pad = crop_to_valid(self.preprocessed_image, scaled_bbox)
                    crop_img = crop_img.to(self.device, non_blocking=self.device.type == 'cuda')
                    crop_interactions, pad_interaction = crop_to_valid(self.interactions, scaled_bbox)

                    # resize input_for_predict (which may be larger than patch size) to patch size
                    # this implementation may not seem straightforward but it does save VRAM which is crucial here
                    if not all([i == j for i, j in zip(self.configuration_manager.patch_size, scaled_patch_size)]):
                        crop_interactions_resampled_gpu = torch.empty((7, *self.configuration_manager.patch_size), dtype=torch.float16, device=self.device)
                        # previous seg, bbox+, bbox-
                        for i in range(0, 3):
                            # this is area for a reason but I aint telling ya why
                            if any([x for y in pad_interaction for x in y]):
                                tmp = pad_cropped(crop_interactions[i].to(self.device, non_blocking=self.device.type == 'cuda'), pad_interaction)
                            else:
                                tmp = crop_interactions[i].to(self.device)
                            crop_interactions_resampled_gpu[i] = interpolate(tmp[None, None], self.configuration_manager.patch_size, mode='area')[0][0]
                        empty_cache(self.device)

                        max_pool_ks = round_to_nearest_odd(zoom_out_factor * 2 - 1)
                        # point+, point-, scribble+, scribble-
                        for i in range(3, 7):
                            if any([x for y in pad_interaction for x in y]):
                                tmp = pad_cropped(crop_interactions[i].to(self.device, non_blocking=self.device.type == 'cuda'), pad_interaction)
                            else:
                                tmp = crop_interactions[i].to(self.device, non_blocking=self.device.type == 'cuda')
                            if max_pool_ks > 1:
                                # dilate to preserve interactions after downsampling
                                tmp = iterative_3x3_same_padding_pool3d(tmp[None, None], max_pool_ks)[0, 0]
                            # this is 'area' for a reason but I aint telling ya why
                            crop_interactions_resampled_gpu[i] = interpolate(tmp[None, None], self.configuration_manager.patch_size, mode='area')[0][0]
                        del tmp

                        crop_img = interpolate(pad_cropped(crop_img, pad)[None] if any([x for y in pad_interaction for x in y]) else crop_img[None], self.configuration_manager.patch_size, mode='trilinear')[0]
                        crop_interactions = crop_interactions_resampled_gpu

                        del crop_interactions_resampled_gpu
                        empty_cache(self.device)
                    else:
                        # crop_img is already on device
                        crop_img = pad_cropped(crop_img, pad) if any([x for y in pad_interaction for x in y]) else crop_img
                        crop_interactions = pad_cropped(crop_interactions.to(self.device, non_blocking=self.device.type == 'cuda'), pad_interaction) if any([x for y in pad_interaction for x in y]) else crop_interactions.to(self.device, non_blocking=self.device.type == 'cuda')

                    input_for_predict = torch.cat((crop_img, crop_interactions))
                    del crop_img, crop_interactions

                    pred = self.network(input_for_predict[None])[0].argmax(0).detach()

                    del input_for_predict

                    # detect changes at borders
                    previous_zoom_prediction = crop_and_pad_nd(self.interactions[0], scaled_bbox).to(self.device, non_blocking=self.device.type == 'cuda')
                    if not all([i == j for i, j in zip(pred.shape, previous_zoom_prediction.shape)]):
                        previous_zoom_prediction = interpolate(previous_zoom_prediction[None, None].to(float), pred.shape, mode='nearest')[0, 0]

                    abs_pxl_change_threshold = 1500
                    rel_pxl_change_threshold = 0.2
                    min_pxl_change_threshold = 100
                    continue_zoom = False
                    if zoom_out_factor < 4 and self.do_prediction_propagation:
                        for dim in range(len(scaled_bbox)):
                            if continue_zoom:
                                break
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
                                    break
                                if pixels_diff > min_pxl_change_threshold and rel_change > rel_pxl_change_threshold:
                                    continue_zoom = True
                                    if self.verbose:
                                        print(f'continue zooming because relative change of {rel_change} > {rel_pxl_change_threshold} and n_pixels {pixels_diff} > {min_pxl_change_threshold}')
                                    break
                                del slice_prev, slice_curr, pixels_prev, pixels_current, pixels_diff

                        del previous_zoom_prediction

                    # resize prediction to correct size and place in target buffer + interactions
                    if not all([i == j for i, j in zip(pred.shape, scaled_patch_size)]):
                        pred = (interpolate(pred[None, None].to(float), scaled_patch_size, mode='trilinear')[0, 0] >= 0.5).to(torch.uint8)

                    # if we do not continue zooming we need a difference map for sampling patches
                    if not continue_zoom and zoom_out_factor > 1:
                        # wow this circus saves ~30ms relative to naive implementation
                        previous_prediction = previous_prediction.to(self.device, non_blocking=self.device.type == 'cuda')
                        seen_bbox = [[max(0, i[0]), min(i[1], s)] for i, s in zip(scaled_bbox, previous_prediction.shape)]
                        bbox_tmp = [[i[0] - s[0], i[1] - s[0]] for i, s in zip(seen_bbox, scaled_bbox)]
                        bbox_tmp = [[max(0, i[0]), min(i[1], s)] for i, s in zip(bbox_tmp, scaled_patch_size)]
                        slicer = bounding_box_to_slice(seen_bbox)
                        slicer2 = bounding_box_to_slice(bbox_tmp)
                        diff_map = pred[slicer2] != previous_prediction[slicer]
                        # dont allocate new memory, just reuse previous_prediction. We don't need it anymore
                        previous_prediction.zero_()
                        diff_map = paste_tensor(previous_prediction, diff_map, seen_bbox)

                        # open the difference map to keep computational load in check (fewer refinement boxes)
                        # open distance map
                        diff_map[slicer] = iterative_3x3_same_padding_pool3d(diff_map[slicer][None, None], kernel_size=5, use_min_pool=True)[0, 0]
                        diff_map[slicer] = iterative_3x3_same_padding_pool3d(diff_map[slicer][None, None], kernel_size=5, use_min_pool=False)[0, 0]

                        has_diff = torch.any(diff_map[slicer])

                        del previous_prediction
                    else:
                        has_diff = False

                    if zoom_out_factor == 1 or (not continue_zoom and has_diff): # rare case where no changes are needed because of useless interaction. Need to check for not continue_zoom because otherwise diff_map wint exist
                        pred = pred.cpu()

                        if zoom_out_factor == 1:
                            paste_tensor(self.interactions[0], pred.half(), scaled_bbox)
                        else:
                            seen_bbox = [[max(0, i[0]), min(i[1], s)] for i, s in
                                         zip(scaled_bbox, diff_map.shape)]
                            bbox_tmp = [[i[0] - s[0], i[1] - s[0]] for i, s in zip(seen_bbox, scaled_bbox)]
                            bbox_tmp = [[max(0, i[0]), min(i[1], s)] for i, s in zip(bbox_tmp, scaled_patch_size)]
                            slicer = bounding_box_to_slice(seen_bbox)
                            slicer2 = bounding_box_to_slice(bbox_tmp)
                            mask = (diff_map[slicer] > 0).cpu()
                            self.interactions[0][slicer][mask] = pred[slicer2][mask].half()

                        # place into target buffer
                        bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                                zip(scaled_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
                        paste_tensor(self.target_buffer, pred, bbox)
                    del pred

                    empty_cache(self.device)

                    if continue_zoom:
                        zoom_out_factor *= 1.5
                        zoom_out_factor = min(4, zoom_out_factor)
                    else:
                        zoom_out_factor = None
                end = time()
                print(f'Auto zoom stage took {round(end - start, ndigits=2)}s. Max zoom out factor was {max_zoom_out_factor}')

                if max_zoom_out_factor > 1 and has_diff:
                    start_refinement = time()
                    # only use the region that was previously looked at. Use last scaled_bbox
                    if self.has_positive_bbox:
                        # mask positive bbox channel with dilated current segmentation to avoid bbox nonsense.
                        # Basically convert bbox to lasso
                        pos_bbox_idx = -6
                        self.interactions[pos_bbox_idx][(~(self.interactions[0] > 0.5)).cpu()] = 0
                        self.has_positive_bbox = False

                    bboxes_ordered = generate_bounding_boxes(diff_map, self.configuration_manager.patch_size, stride='auto', margin=(10, 10, 10), max_depth=3)

                    del diff_map

                    if self.verbose:
                        print(f'Using {len(bboxes_ordered)} bounding boxes for refinement')

                    preallocated_input = torch.zeros((8, *self.configuration_manager.patch_size), device=self.device, dtype=torch.float)
                    for nref, refinement_bbox in enumerate(bboxes_ordered):
                        assert self.pad_mode_data == 'constant'
                        crop_and_pad_into_buffer(preallocated_input[0], refinement_bbox, self.preprocessed_image[0])
                        crop_and_pad_into_buffer(preallocated_input[1:], refinement_bbox, self.interactions)

                        pred = self.network(preallocated_input[None])[0].argmax(0).detach().cpu()

                        paste_tensor(self.interactions[0], pred, refinement_bbox)
                        # place into target buffer
                        bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in zip(refinement_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
                        paste_tensor(self.target_buffer, pred, bbox)
                        del pred
                        preallocated_input.zero_()
                    del preallocated_input
                    empty_cache(self.device)
                    end_refinement = time()
                    print(f'Took {round(end_refinement - start_refinement, 2)} s for refining the segmentation with {len(bboxes_ordered)} bounding boxes')
                else:
                    print('No refinement necessary')
        print(f'Done. Total time {round(time() - start, 2)}s')

        self.new_interaction_centers = []
        self.new_interaction_zoom_out_factors = []
        empty_cache(self.device)
