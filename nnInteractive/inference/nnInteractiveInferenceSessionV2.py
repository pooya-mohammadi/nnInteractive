from typing_extensions import Unpack

import threading
from concurrent.futures import ProcessPoolExecutor, Future
from os import cpu_count
from typing import Union, List, Tuple, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, insert_crop_into_image, \
    crop_and_pad_nd
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from torch import nn
from torch._dynamo import OptimizedModule
from time import time, sleep

import nnunetv2
from nnunetv2.imageio.nibabel_reader_writer import NibabelIO
from nnInteractive.utils.bboxes import cover_structure_with_bboxes, \
    get_bboxes_and_prios_from_image, prediction_propagation_add_bounding_boxes, filter_bboxes
from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.training.interaction_simulation.interactions.points import PointInteraction
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class nnInteractiveInferenceSessionV2():
    def __init__(self,
                 device: torch.device = torch.device('cuda'),
                 use_torch_compile: bool = False,
                 verbose: bool = False,
                 torch_n_threads: int = 8,
                 interaction_decay: float = 0.9,
                 use_background_preprocessing: bool = True,
                 do_prediction_propagation: bool = True,
                 use_pinned_memory: bool = True
                 ):
        """
        Only intended to work with nnInteractiveTrainerV2 and its derivatives
        """
        # set as part of initialization
        assert use_torch_compile is False, ('This implementation places the preprocessed image and the interactions '
                                            'into pinned memory for speed reasons. This is incompatible with '
                                            'torch.compile because of inconsistent strides in the memory layout. '
                                            'Note to self: .contiguous() on GPU could be a solution. Unclear whether '
                                            'that will yield a benefit though.')
        self.network = None
        self.label_manager = None
        self.dataset_json = None
        self.trainer_name = None
        self.configuration_manager = None
        self.plans_manager = None
        self.use_pinned_memory = use_pinned_memory
        self.device = device
        self.use_torch_compile = use_torch_compile
        self.interaction_decay = interaction_decay
        self.use_background_preprocessing = use_background_preprocessing

        # image specific
        self.all_interaction_bboxes = []
        self.new_interaction_bboxes = []
        self.interactions: torch.Tensor = None
        self.preprocessed_image: torch.Tensor = None
        self.preprocessed_props = None
        self.target_buffer: Union[np.ndarray, torch.Tensor] = None

        # this will be set when loading the model (initialize_from_trained_model_folder)
        self.pad_mode_data = self.preferred_scribble_thickness = self.point_interaction = None

        self.verbose = verbose

        self.preprocessing_executor = ProcessPoolExecutor(1) if use_background_preprocessing else None

        self.do_prediction_propagation: bool = do_prediction_propagation
        self.prediction_propagation_max_n_patches: int = 30

        # event to signal that interactions have been initialized
        self._interactions_ready = threading.Event()
        self._interaction_initializer_thread = None

        torch.set_num_threads(min(torch_n_threads, cpu_count()))

        self.input_image_shape = None


    def set_image(self, image: np.ndarray, image_properties: dict):
        """
        Image must be 4D to satisfy nnU-Net needs: [c, x, y, z]
        """
        self._reset_session()
        self.input_image_shape = image.shape
        if self.verbose:
            print(f'Initialize with raw image shape {image.shape}')
        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)

        if self.use_background_preprocessing:
            future = self.preprocessing_executor.submit(
                preprocessor.run_case_npy, image, None, image_properties,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json
            )
            self.preprocessed_image = future
        else:
            data, _, properties = preprocessor.run_case_npy(
                image, None, image_properties,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json
            )
            data = torch.from_numpy(data)
            self.preprocessed_image = data
            properties['shape_of_preprocessed_image'] = data.shape[1:]
            self.preprocessed_props = properties
            if self.verbose:
                print(f"Preprocessed image, shape: {data.shape}")

        # start a dedicated thread to initialize interactions as soon as the preprocessed image is ready
        self._interactions_ready.clear()
        self._interaction_initializer_thread = threading.Thread(
            target=self._interaction_initializer, daemon=True
        )
        self._interaction_initializer_thread.start()

    def _interaction_initializer(self):
        """
        Continuously check whether the preprocessed image is ready.
        Once it is available, do any necessary post-processing (e.g. pinning) and initialize self.interactions.
        Finally, signal that interactions are ready.
        """
        # Wait until self.preprocessed_image is no longer a Future (or is done)
        while isinstance(self.preprocessed_image, Future):
            if self.preprocessed_image.done():
                break
            sleep(0.1)
        # Retrieve and convert the preprocessed image if it came as a Future.
        if isinstance(self.preprocessed_image, Future):
            data, _, properties = self.preprocessed_image.result()
            data = torch.from_numpy(data)
            self.preprocessed_image = data
            properties['shape_of_preprocessed_image'] = data.shape[1:]
            self.preprocessed_props = properties
            if self.verbose:
                print(f"Retrieved preprocessed image (from thread), shape: {data.shape}")

        # Ensure pinned memory if needed
        if (self.device.type == 'cuda' and self.use_pinned_memory and
                not self.preprocessed_image.is_pinned()):
            self.preprocessed_image = self.preprocessed_image.pin_memory()

        # Allocate interactions if not already allocated.
        if self.interactions is None:
            self.interactions = torch.zeros(
                (7, *self.preprocessed_image.shape[1:]),
                device='cpu',
                dtype=torch.float16,
                pin_memory=(self.device.type == 'cuda' and self.use_pinned_memory)
            )
            if self.verbose:
                print(f'Initialized interactions (from thread) with shape {self.interactions.shape}')

        # Signal that interactions (and preprocessed image) are ready.
        self._interactions_ready.set()

    def _finish_preprocessing_and_initialize_interactions(self):
        """
        Waits until the preprocessed image has been retrieved and interactions initialized,
        then performs final consistency checks.
        """
        assert self.preprocessed_image is not None, 'Set an image first'
        # Instead of blocking indefinitely, loop with a timeout so that ctrl+c (KeyboardInterrupt) is responsive.
        while not self._interactions_ready.wait(timeout=0.1):
            pass

        # Check that the preprocessed image shape is as expected.
        for d in range(len(self.preprocessed_props['bbox_used_for_cropping'])):
            expected = self.preprocessed_props['bbox_used_for_cropping'][d][1] - \
                       self.preprocessed_props['bbox_used_for_cropping'][d][0]
            assert self.preprocessed_image.shape[d + 1] == expected, \
                "Image has been resampled. This is not supported here!"

    def set_target_buffer(self, target_buffer: Union[np.ndarray, torch.Tensor]):
        """
        Must be 3d numpy array or torch.Tensor
        """
        self.target_buffer = target_buffer


    def set_do_prediction_propagation(self, do_propagation: bool, max_num_patches: Optional[int] = None):
        self.do_prediction_propagation = do_propagation
        if max_num_patches:
            self.prediction_propagation_max_n_patches = max_num_patches

    def _reset_session(self):
        del self.preprocessed_image
        del self.target_buffer
        del self.interactions
        del self.preprocessed_props
        del self.all_interaction_bboxes
        self.all_interaction_bboxes = []
        self.new_interaction_bboxes = []
        self.preprocessed_image = None
        self.target_buffer = None
        self.interactions = None
        self.preprocessed_props = None
        empty_cache(self.device)
        self._interactions_ready.clear()
        self.input_image_shape = None

    def reset_interactions(self):
        """
        Use this to reset all interactions and start from scratch for the current image. This includes the initial
        segmentation!
        """
        if self.interactions is not None:
            self.interactions.fill_(0)

        self.all_interaction_bboxes = []

        if self.target_buffer is not None:
            if isinstance(self.target_buffer, np.ndarray):
                self.target_buffer.fill(0)
            elif isinstance(self.target_buffer, torch.Tensor):
                self.target_buffer.zero_()
        empty_cache(self.device)

    # @benchmark_decorator
    def add_bbox_interaction(self, bbox_coords, include_interaction: bool, run_prediction: bool = True) -> np.ndarray:
        self._finish_preprocessing_and_initialize_interactions()

        lbs_transformed = [round(i) for i in transform_coordinates_noresampling([i[0] for i in bbox_coords],
                                                             self.preprocessed_props['bbox_used_for_cropping'])]
        ubs_transformed = [round(i) for i in transform_coordinates_noresampling([i[1] for i in bbox_coords],
                                                             self.preprocessed_props['bbox_used_for_cropping'])]
        transformed_bbox_coordinates = [[i, j] for i, j in zip(lbs_transformed, ubs_transformed)]

        if self.verbose:
            print(f'Added bounding box coordinates.\n'
                  f'Raw: {bbox_coords}\n'
                  f'Transformed: {transformed_bbox_coordinates}\n'
                  f"Crop Bbox: {self.preprocessed_props['bbox_used_for_cropping']}")

        # Prevent collapsed bounding boxes and clip to image shape
        image_shape = self.preprocessed_image.shape  # Assuming shape is (C, H, W, D) or similar

        for dim in range(len(transformed_bbox_coordinates)):
            transformed_start, transformed_end = transformed_bbox_coordinates[dim]

            # Clip to image boundaries
            transformed_start = max(0, transformed_start)
            transformed_end = min(image_shape[dim + 1], transformed_end)  # +1 to skip channel dim

            # Ensure the bounding box does not collapse to a single point
            if transformed_end <= transformed_start:
                if transformed_start == 0:
                    transformed_end = min(1, image_shape[dim + 1])
                else:
                    transformed_start = max(transformed_start - 1, 0)

            transformed_bbox_coordinates[dim] = [transformed_start, transformed_end]

        if self.verbose:
            print(f'Bbox coordinates after clip to image boundaries and preventing dim collapse:\n'
                  f'Bbox: {transformed_bbox_coordinates}\n'
                  f'Internal image shape: {self.preprocessed_image.shape}')

        self._add_patch_for_bbox_interaction(transformed_bbox_coordinates)

        # decay old interactions
        self.interactions[-6:-4] *= self.interaction_decay

        # place bbox
        slicer = tuple([slice(*i) for i in transformed_bbox_coordinates])
        channel = -6 if include_interaction else -5
        self.interactions[channel, Unpack[slicer]] = 1

        # forward pass
        if run_prediction:
            self._predict()

    # @benchmark_decorator
    def add_point_interaction(self, coordinates: Tuple[int, ...], include_interaction: bool, run_prediction: bool = True):
        self._finish_preprocessing_and_initialize_interactions()

        transformed_coordinates = [round(i) for i in transform_coordinates_noresampling(coordinates,
                                                             self.preprocessed_props['bbox_used_for_cropping'])]

        self._add_patch_for_point_interaction(transformed_coordinates)

        # decay old interactions
        self.interactions[-4:-2] *= self.interaction_decay

        interaction_channel = -4 if include_interaction else -3
        self.interactions[interaction_channel] = self.point_interaction.place_point(
            transformed_coordinates, self.interactions[interaction_channel])
        if run_prediction:
            self._predict()

    # @benchmark_decorator
    def add_scribble_interaction(self, scribble_image: np.ndarray,  include_interaction: bool, run_prediction: bool = True):
        assert all([i == j for i, j in zip(self.input_image_shape[1:], scribble_image.shape)]), f'Given scribble image must match input image shape. Input image was: {self.input_image_shape[1:]}, given: {scribble_image.shape}'
        self._finish_preprocessing_and_initialize_interactions()

        scribble_image = torch.from_numpy(scribble_image)

        # crop (as in preprocessing)
        scribble_image = crop_and_pad_nd(scribble_image, self.preprocessed_props['bbox_used_for_cropping'])

        self._add_patch_for_scribble_interaction(scribble_image)

        # decay old interactions
        self.interactions[-2:] *= self.interaction_decay

        interaction_channel = -2 if include_interaction else -1
        torch.maximum(self.interactions[interaction_channel], scribble_image.to(self.interactions.device),
                      out=self.interactions[interaction_channel])
        del scribble_image
        empty_cache(self.device)
        if run_prediction:
            self._predict()

    # @benchmark_decorator
    def add_lasso_interaction(self, lasso_image: np.ndarray,  include_interaction: bool, run_prediction: bool = True):
        assert all([i == j for i, j in zip(self.input_image_shape[1:], lasso_image.shape)]), f'Given lasso image must match input image shape. Input image was: {self.input_image_shape[1:]}, given: {lasso_image.shape}'
        self._finish_preprocessing_and_initialize_interactions()

        lasso_image = torch.from_numpy(lasso_image)

        # crop (as in preprocessing)
        lasso_image = crop_and_pad_nd(lasso_image, self.preprocessed_props['bbox_used_for_cropping'])

        self._add_patch_for_lasso_interaction(lasso_image)

        # decay old interactions
        self.interactions[-6:-4] *= self.interaction_decay

        # lasso is written into bbox channel
        interaction_channel = -6 if include_interaction else -5
        torch.maximum(self.interactions[interaction_channel], lasso_image.to(self.interactions.device),
                      out=self.interactions[interaction_channel])
        del lasso_image
        empty_cache(self.device)
        if run_prediction:
            self._predict()

    # @benchmark_decorator
    def add_initial_seg_interaction(self, initial_seg: np.ndarray, run_prediction: bool = False):
        """
        WARNING THIS WILL RESET INTERACTIONS!
        """
        assert all([i == j for i, j in zip(self.input_image_shape[1:], initial_seg.shape)]), f'Given initial seg must match input image shape. Input image was: {self.input_image_shape[1:]}, given: {initial_seg.shape}'

        self._finish_preprocessing_and_initialize_interactions()

        self.reset_interactions()

        if isinstance(self.target_buffer, np.ndarray):
            self.target_buffer[:] = initial_seg

        initial_seg = torch.from_numpy(initial_seg)

        if isinstance(self.target_buffer, torch.Tensor):
            self.target_buffer[:] = initial_seg

        # crop (as in preprocessing)
        initial_seg = crop_and_pad_nd(initial_seg, self.preprocessed_props['bbox_used_for_cropping'])

        self._add_patch_for_initial_seg_interaction(initial_seg)

        # initial seg is written into initial seg buffer
        interaction_channel = -7
        self.interactions[interaction_channel] = initial_seg
        del initial_seg
        empty_cache(self.device)
        if run_prediction:
            self._predict()

    @torch.inference_mode
    # @benchmark_decorator
    def _predict(self):
        max_idx = len(self.new_interaction_bboxes) if not self.do_prediction_propagation else self.prediction_propagation_max_n_patches#(self.prediction_propagation_max_n_patches + len(self.new_interaction_bboxes))

        allow_revisiting = False

        # we need to predict all new bounding boxes. We start with the last added interaction
        bboxes_for_prediction = self.new_interaction_bboxes[::-1]

        idx = 0
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            while idx < min(max_idx, len(bboxes_for_prediction)):
                bbox = bboxes_for_prediction[idx]
                # this may break pinned memory but this is how it is...
                crop_img = crop_and_pad_nd(self.preprocessed_image, bbox, pad_mode=self.pad_mode_data)
                crop_interactions = crop_and_pad_nd(self.interactions, bbox)

                # copy stuff to GPU and stack
                crop_img = crop_img.to(self.device, non_blocking=self.device.type == 'cuda')
                crop_interactions = crop_interactions.to(self.device, non_blocking=self.device.type == 'cuda')
                input_for_predict = torch.cat((crop_img, crop_interactions))
                has_only_neg_interact = torch.any(crop_interactions[[2, 4, 6]]) and not torch.any(crop_interactions[[1, 3, 5]])
                del crop_img, crop_interactions

                pred = self.network(input_for_predict[None])[0].argmax(0).detach().cpu()
                del input_for_predict

                # We cannot run prediction propagation if we only see negative interactions
                if not has_only_neg_interact:
                    previous_prediction = crop_and_pad_nd(self.interactions[0], bbox)

                    added_border_bboxes = prediction_propagation_add_bounding_boxes(
                        previous_prediction,
                        pred,
                        bbox,
                        new_bbox_distance=[p // 2 for p in self.configuration_manager.patch_size],
                        abs_pxl_change_threshold=1e3,
                        rel_pxl_change_threshold=0.2,
                        min_pxl_change_threshold=100
                    )

                    added_border_bboxes_filtered = filter_bboxes(
                        added_border_bboxes,
                        bboxes_for_prediction[idx:],
                        min_distance_to_selected=np.mean([p // 3 for p in self.configuration_manager.patch_size])
                    )

                    if not allow_revisiting:
                        added_border_bboxes_filtered = filter_bboxes(
                            added_border_bboxes_filtered,
                            bboxes_for_prediction[:idx],
                            min_distance_to_selected=np.mean([p // 3 for p in self.configuration_manager.patch_size])
                        )

                    if self.verbose and len(added_border_bboxes_filtered) > 0:
                        print(f'Added {len(added_border_bboxes_filtered)} due to changed prediction at border')

                    bboxes_for_prediction += added_border_bboxes_filtered

                # I would love to run this in a separate thread but then we would get problems with coherency
                # because patches may overlap and we want successive patches to work on the most up to date
                # segmentation in self.interactions[0]
                insert_crop_into_image(self.interactions[0], pred, bbox)
                # place into target buffer
                bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in zip(bbox, self.preprocessed_props['bbox_used_for_cropping'])]
                insert_crop_into_image(self.target_buffer, pred if isinstance(self.target_buffer, torch.Tensor) else pred.numpy(), bbox)
                del pred
                idx += 1
        self.all_interaction_bboxes += self.new_interaction_bboxes
        self.new_interaction_bboxes = []
        empty_cache(self.device)
        # import IPython;IPython.embed()

    def _add_patch_for_point_interaction(self, coordinates):
        return self._generic_add_patch(coordinates)

    def _add_patch_for_bbox_interaction(self, bbox):
        bbox_center = [round((i[0] + i[1]) / 2) for i in bbox]
        return self._generic_add_patch(bbox_center)

    def _add_patch_for_scribble_interaction(self, scribble_image):
        return self._generic_add_patch_from_image(scribble_image)

    def _add_patch_for_lasso_interaction(self, lasso_image):
        return self._generic_add_patch_from_image(lasso_image)

    def _add_patch_for_initial_seg_interaction(self, initial_seg):
        return self._generic_add_patch_from_image(initial_seg)

    # @benchmark_decorator
    def _generic_add_patch_from_image(self, image: torch.Tensor):
        if not torch.any(image):
            print('Received empty image prompt. Cannot add patches for prediction')
            return
        nonzero_indices = torch.nonzero(image, as_tuple=False)
        mn = torch.min(nonzero_indices, dim=0)[0]
        mx = torch.max(nonzero_indices, dim=0)[0]
        roi = [[i.item(), x.item() + 1] for i, x in zip(mn, mx)]
        bboxes, prios = get_bboxes_and_prios_from_image(image.to(torch.float32),
                                                        roi,
                                                        [round(i // (1/0.7)) for i in self.configuration_manager.patch_size],
                                                        self.configuration_manager.patch_size)
        bboxes_ordered = cover_structure_with_bboxes(prios, bboxes,
                                                     np.mean(self.configuration_manager.patch_size) / 2,
                                                     verbose=self.verbose)
        if self.verbose:
            print(f'Added {len(bboxes_ordered)} new bounding boxes for prediction from this interaction')
        self.new_interaction_bboxes += bboxes_ordered

    # @benchmark_decorator
    def _generic_add_patch(self, coordinates):
        """
        coordinates must be given for internal representation!
        """
        # if self.verbose:
        #     print(f'Coordinates before cleansing: {coordinates}')
        # coordinates must be int
        image_shape = self.preprocessed_image.shape
        cleansed_center = []
        for d in range(len(coordinates)):
            # if image_shape[d+1] <= self.configuration_manager.patch_size[d]:
            #     cleansed_center.append(round(image_shape[d+1] / 2))
            # else:
            #     ps_half = self.configuration_manager.patch_size[d] // 2
            #     is_uneven = self.configuration_manager.patch_size[d] % 2
            #     coord = max(ps_half, coordinates[d])
            #     coord = min(coord, image_shape[d+1] - ps_half - is_uneven)
            #     cleansed_center.append(coord)
            cleansed_center.append(min(image_shape[d + 1], max(0, coordinates[d])))
        bbox = [[c - p // 2, c + p // 2 + p % 2]
                for c, p in zip(cleansed_center, self.configuration_manager.patch_size)]
        self.new_interaction_bboxes.append(bbox)

        if self.verbose:
            print(f'Added the following bbox for prediction:\n'
                  f'{self.new_interaction_bboxes[-1]}')
            print(f'We now have {len(self.new_interaction_bboxes)} bboxes in the list')


    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_fold: Union[int, str],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        # load trainer specific settings
        expected_json_file = join(model_training_output_dir, 'inference_session_class.json')
        json_content = load_json(expected_json_file)
        if isinstance(json_content, str):
            # old convention where we only specified the inference class in this file. Set defaults for stuff
            point_interaction_radius = 4
            point_interaction_use_etd = True
            self.preferred_scribble_thickness = [2, 2, 2]
            self.point_interaction = PointInteraction(
                point_interaction_radius,
                1,
                point_interaction_use_etd)
            self.pad_mode_data = "constant"
        else:
            point_interaction_radius = json_content['point_radius']
            self.preferred_scribble_thickness = json_content['preferred_scribble_thickness']
            if not isinstance(self.preferred_scribble_thickness, (tuple, list)):
                self.preferred_scribble_thickness = [self.preferred_scribble_thickness] * 3
            point_interaction_use_etd = True # so far this is not defined in that file so we stick with default
            self.point_interaction = PointInteraction(point_interaction_radius, 1, point_interaction_use_etd)
            # padding mode for data. See nnInteractiveTrainerV2_nodelete_reflectpad
            self.pad_mode_data = json_content['pad_mode_image'] if 'pad_mode_image' in json_content.keys() else "constant"

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        use_fold = int(use_fold) if use_fold != 'all' else use_fold
        checkpoint = torch.load(join(model_training_output_dir, f'fold_{use_fold}', checkpoint_name),
                                map_location=self.device, weights_only=False)
        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']

        parameters = checkpoint['network_weights']

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        ).to(self.device)
        network.load_state_dict(parameters)

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if self.use_torch_compile and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager,
                              dataset_json: dict, trainer_name: str):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if self.use_torch_compile and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

        if not self.use_torch_compile and isinstance(self.network, OptimizedModule):
            self.network = self.network._orig_mod

        self.network = self.network.to(self.device)


def transform_coordinates_noresampling(
        coords_orig: Union[List[int], Tuple[int, ...]],
        nnunet_preprocessing_crop_bbox: List[Tuple[int, int]]
) -> Tuple[int, ...]:
    """
    converts coordinates in the original uncropped image to the internal cropped representation. Man I really hate
    nnU-Net's crop to nonzero!
    """
    return tuple([coords_orig[d] - nnunet_preprocessing_crop_bbox[d][0] for d in range(len(coords_orig))])



if __name__ == '__main__':
    img, props = NibabelIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTr/liver_0_0000.nii.gz')])
    session = nnInteractiveInferenceSessionV2(
        device=torch.device('cuda:0'),
        use_torch_compile=False,
        verbose=True,
        torch_n_threads=8,
        interaction_decay=0.9,
        point_interaction_radius=4,
        point_interaction_use_etd=True,
        use_background_preprocessing=False
    )
    session.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset224_nnInteractive/nnInteractiveTrainerV2_2000ep__nnUNetResEncUNetLPlans_noResampling__3d_fullres_ps192_bs24'),
        5,
        'checkpoint_final.pth'
    )
    seg_buffer = torch.zeros(img.shape[1:], dtype=torch.uint8)

    session.set_image(img, props)
    session.set_target_buffer(seg_buffer)

    # scribble
    # scribble_0 = NibabelIO().read_seg('/home/isensee/temp/instanceseg/liver_0_0000-labels.nrrd')
    # scribble not tested because I cannot export segmentation from MITK and then read it with nibabel/sitk

    #################################
    #### bbox ####
    session.reset_interactions()
    bbox = [[58, 59], [169, 283], [90, 198]]
    slicer = bounding_box_to_slice(bbox)
    img2 = np.copy(img[0])
    img2[slicer] = 10000
    # view_batch(img2)

    session._finish_preprocessing_and_initialize_interactions()

    session.add_bbox_interaction(bbox, True)
    # view_batch(
    #     session.preprocessed_image,
    #     session.interactions[1],
    #     session.preprocessed_image[0]+session.interactions[1]*4, session.interactions[0]
    # )
    # view_batch(img, session.target_buffer.numpy(), img[0] + session.target_buffer.numpy() * 3000)
    #################################

    #################################
    #### points ####
    session.reset_interactions()
    point_0 = (61, img.shape[2] - 184, img.shape[3] - 350)
    point_1 = (47, img.shape[2] - 189, img.shape[3] - 277)
    point_2 = (65, img.shape[2] - 242, img.shape[3] - 330)

    # place first point
    times = []
    from time import time
    st = time()
    session.add_point_interaction(point_0, include_interaction=True)
    end = time()
    times.append(end - st)
    st = end
    session.add_point_interaction(point_1, include_interaction=True)
    end = time()
    times.append(end - st)
    st = end
    session.add_point_interaction(point_2, include_interaction=True)
    end = time()
    times.append(end - st)
    st = end

    print(f"Times for 3 interactions: {times}")

    # view_batch(
    #     session.preprocessed_image,
    #     session.interactions[3],
    #     session.preprocessed_image[0]+session.interactions[3]*4, session.interactions[0]
    # )
    #################################

