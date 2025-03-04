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
    # @benchmark_decorator
    def _predict(self):
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
                # start = time()
                previous_prediction = torch.clone(self.interactions[0])
                # if self.verbose_run_times: print('Time clone prev prediction', time() - start)

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

                    # start = time()
                    crop_img, pad = crop_to_valid(self.preprocessed_image, scaled_bbox)
                    crop_img = crop_img.to(self.device, non_blocking=self.device.type == 'cuda')
                    crop_interactions, pad_interaction = crop_to_valid(self.interactions, scaled_bbox)

                    # crop_img = crop_and_pad_nd(self.preprocessed_image, scaled_bbox, pad_mode=self.pad_mode_data)
                    # crop_interactions = crop_and_pad_nd(self.interactions, scaled_bbox)
                    # if self.verbose_run_times: print(f'Time crop of size {scaled_patch_size}', time() - start)

                    # resize input_for_predict (which may be larger than patch size) to patch size
                    # this implementation may not seem straightforward but it does save VRAM which is crucial here
                    if not all([i == j for i, j in zip(self.configuration_manager.patch_size, scaled_patch_size)]):
                        crop_interactions_resampled_gpu = torch.empty((7, *self.configuration_manager.patch_size), dtype=torch.float16, device=self.device)
                        # previous seg, bbox+, bbox-
                        # start = time()
                        for i in range(0, 3):
                            # this is area for a reason but I aint telling ya why
                            if any([x for y in pad_interaction for x in y]):
                                tmp = pad_cropped(crop_interactions[i].to(self.device, non_blocking=self.device.type == 'cuda'), pad_interaction)
                            else:
                                tmp = crop_interactions[i].to(self.device)
                            crop_interactions_resampled_gpu[i] = interpolate(tmp[None, None], self.configuration_manager.patch_size, mode='area')[0][0]
                        empty_cache(self.device)
                        # if self.verbose_run_times: print(f'Time resample interactions 0-3', time() - start)

                        max_pool_ks = round_to_nearest_odd(zoom_out_factor * 2 - 1)
                        # point+, point-, scribble+, scribble-
                        # time_mp = []
                        # time_interpolate = []
                        # time_pad = []
                        for i in range(3, 7):
                            # start = time()
                            if any([x for y in pad_interaction for x in y]):
                                tmp = pad_cropped(crop_interactions[i].to(self.device, non_blocking=self.device.type == 'cuda'), pad_interaction)
                            else:
                                tmp = crop_interactions[i].to(self.device, non_blocking=self.device.type == 'cuda')
                            # time_pad.append(time() - start)
                            if max_pool_ks > 1:
                                # dilate to preserve interactions after downsampling
                                # start = time()
                                tmp = iterative_3x3_same_padding_pool3d(tmp[None, None], max_pool_ks)[0, 0]
                                # time_mp.append(time() - start)
                            # start = time()
                            # this is 'area' for a reason but I aint telling ya why
                            crop_interactions_resampled_gpu[i] = interpolate(tmp[None, None], self.configuration_manager.patch_size, mode='area')[0][0]
                            # time_interpolate.append(time() - start)
                        del tmp
                        # if self.verbose_run_times: print(f'Time padding (sum)', np.sum(time_pad))
                        # if self.verbose_run_times: print(f'Time max poolings (sum)', np.sum(time_mp))
                        # if self.verbose_run_times: print(f'Time resampling interactions 4-7 (sum)', np.sum(time_interpolate))
                        # start = time()
                        crop_img = interpolate(pad_cropped(crop_img, pad)[None] if any([x for y in pad_interaction for x in y]) else crop_img[None], self.configuration_manager.patch_size, mode='trilinear')[0]
                        # if self.verbose_run_times: print(f'Time resample image', time() - start)
                        crop_interactions = crop_interactions_resampled_gpu
                        del crop_interactions_resampled_gpu
                        empty_cache(self.device)
                    else:
                        # start = time()
                        # crop_img is already on device
                        crop_img = pad_cropped(crop_img, pad) if any([x for y in pad_interaction for x in y]) else crop_img
                        # crop_img = crop_img.to(self.device, non_blocking=self.device.type == 'cuda')
                        crop_interactions = pad_cropped(crop_interactions.to(self.device, non_blocking=self.device.type == 'cuda'), pad_interaction) if any([x for y in pad_interaction for x in y]) else crop_interactions.to(self.device, non_blocking=self.device.type == 'cuda')
                        # if self.verbose_run_times: print('Time no resampling just copy to GPU', time() - start)

                    # start = time()
                    input_for_predict = torch.cat((crop_img, crop_interactions))
                    # if self.verbose_run_times: print('Time cat', time() - start)
                    del crop_img, crop_interactions

                    # start = time()
                    pred = self.network(input_for_predict[None])[0].argmax(0).detach()
                    # if self.verbose_run_times: print("Time predict", time() - start)

                    del input_for_predict

                    # detect changes at borders
                    previous_zoom_prediction = crop_and_pad_nd(self.interactions[0], scaled_bbox).to(self.device, non_blocking=self.device.type == 'cuda')
                    if not all([i == j for i, j in zip(pred.shape, previous_zoom_prediction.shape)]):
                        # start = time()
                        previous_zoom_prediction = interpolate(previous_zoom_prediction[None, None].to(float), pred.shape, mode='nearest')[0, 0]
                        # if self.verbose_run_times: print('Time resampling prev prediction', time() - start)

                    abs_pxl_change_threshold = 1500
                    rel_pxl_change_threshold = 0.2
                    min_pxl_change_threshold = 100
                    continue_zoom = False
                    if zoom_out_factor < 4 and self.do_prediction_propagation:
                        # start = time()
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
                        # if self.verbose_run_times: print('Time detect change', time() - start)

                    # resize prediction to correct size and place in target buffer + interactions
                    # start = time()
                    if not all([i == j for i, j in zip(pred.shape, scaled_patch_size)]):
                        pred = (interpolate(pred[None, None].to(float), scaled_patch_size, mode='trilinear')[0, 0] >= 0.5).to(torch.uint8)
                    # if self.verbose_run_times: print('Time resize prediction and copy to CPU', time() - start)

                    # if we do not continue zooming we need a difference map for sampling patches
                    if not continue_zoom and zoom_out_factor > 1:
                        # wow this circus saves ~30ms relative to naive implementation
                        previous_prediction = previous_prediction.to(self.device, non_blocking=self.device.type == 'cuda')
                        # start = time()
                        seen_bbox = [[max(0, i[0]), min(i[1], s)] for i, s in zip(scaled_bbox, previous_prediction.shape)]
                        bbox_tmp = [[i[0] - s[0], i[1] - s[0]] for i, s in zip(seen_bbox, scaled_bbox)]
                        bbox_tmp = [[max(0, i[0]), min(i[1], s)] for i, s in zip(bbox_tmp, scaled_patch_size)]
                        slicer = bounding_box_to_slice(seen_bbox)
                        slicer2 = bounding_box_to_slice(bbox_tmp)
                        diff_map = pred[slicer2] != previous_prediction[slicer]
                        # dont allocate new memory, just reuse previous_prediction. We don't need it anymore
                        previous_prediction.zero_()
                        diff_map = paste_tensor(previous_prediction, diff_map, seen_bbox)
                        # if self.verbose_run_times: print('Time create diff_map', time() - start)

                        # open the difference map to keep computational load in check (fewer refinement boxes)
                        # start = time()
                        # open distance map
                        diff_map[slicer] = iterative_3x3_same_padding_pool3d(diff_map[slicer][None, None], kernel_size=5, use_min_pool=True)[0, 0]
                        diff_map[slicer] = iterative_3x3_same_padding_pool3d(diff_map[slicer][None, None], kernel_size=5, use_min_pool=False)[0, 0]

                        has_diff = torch.any(diff_map[slicer])

                        # if self.verbose_run_times: print('Time open diff map', time() - start)

                        del previous_prediction
                    else:
                        has_diff = False

                    if zoom_out_factor == 1 or (not continue_zoom and has_diff): # rare case where no changes are needed because of useless interaction. Need to check for not continue_zoom because otherwise diff_map wint exist
                        pred = pred.cpu()

                        # start = time()
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
                        # if self.verbose_run_times: print(f'Time insert prediction of shape {pred.shape} into interactions[0] (zoom factor {zoom_out_factor}, has diff {has_diff})', time() - start)

                        # place into target buffer
                        # start = time()
                        bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                                zip(scaled_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
                        paste_tensor(self.target_buffer, pred, bbox)
                        #
                        # insert_crop_into_image(self.target_buffer,
                        #                        pred if isinstance(self.target_buffer, torch.Tensor) else pred.numpy(), bbox)
                        # if self.verbose_run_times: print(f'Time insert prediction of shape {pred.shape} into target buffer', time() - start)
                    del pred

                    empty_cache(self.device)

                    if continue_zoom:
                        zoom_out_factor *= 1.5
                        zoom_out_factor = min(4, zoom_out_factor)
                    else:
                        zoom_out_factor = None
                end = time()
                print(f'Initial zoom stage took {round(end - start, ndigits=2)}s. Max zoom out factor was {max_zoom_out_factor}')

                if max_zoom_out_factor > 1 and has_diff:
                    start_refinement = time()
                    # only use the region that was previously looked at. Use last scaled_bbox
                    if self.has_positive_bbox:
                        # start = time()
                        # mask positive bbox channel with dilated current segmentation to avoid bbox nonsense.
                        # Basically convert bbox to lasso
                        pos_bbox_idx = -6
                        # dilated_pred = same_padding_maxpool3d(self.interactions[0][None, None].to(self.device), 3)[0, 0]
                        self.interactions[pos_bbox_idx][(~(self.interactions[0] > 0.5)).cpu()] = 0
                        self.has_positive_bbox = False
                        # if self.verbose_run_times: print('Time for bbox to lasso (maybe)', time() - start)

                    # start = time()
                    bboxes_ordered = generate_bounding_boxes(diff_map, self.configuration_manager.patch_size, stride='auto', margin=(10, 10, 10), max_depth=3)
                    # if self.verbose_run_times: print('Time cover_structure_with_bboxes', time() - start)

                    del diff_map

                    if self.verbose:
                        print(f'Using {len(bboxes_ordered)} bounding boxes for refinement')

                    preallocated_input = torch.zeros((8, *self.configuration_manager.patch_size), device=self.device, dtype=torch.float)
                    for nref, refinement_bbox in enumerate(bboxes_ordered):
                        # start = time()
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
                        # if self.verbose_run_times: print(f'Time refinement loop {nref}, bbox {refinement_bbox}', time() - start)
                    del preallocated_input
                    empty_cache(self.device)
                    end_refinement = time()
                    print(f'Took {round(end_refinement - start_refinement, 2)} s for refining the segmentation with {len(bboxes_ordered)} bounding boxes')
        print(f'Done. Total time {round(time() - start, 2)}s')

        self.new_interaction_centers = []
        self.new_interaction_zoom_out_factors = []
        empty_cache(self.device)
