import os, sys

import SimpleITK
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
from typing import *
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import copy
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *


class ProstateNodeModelBuilder(BaseModelBuilder):
    def __init__(self, image_key='image', model_path=None, Bilinear_model=None, loss=None, loss_weights=None,
                 model_paths: Optional[List[str]] = None):
        super().__init__(image_key, model_path, Bilinear_model, loss, loss_weights)
        self.image_key = image_key
        self.model_path = model_path
        self.Bilinear_model = Bilinear_model
        self.loss = loss
        self.loss_weights = loss_weights
        self.paths = []
        self.image_processors = []
        self.prediction_processors = []
        self.model_paths = model_paths
        self.models = []

    def build_model(self, graph=None, session=None, model_name='modelname'):
        for model_path in self.model_paths:
            print(f"Loading model from: {model_path}")
            model = tf.keras.models.load_model(model_path)
            model.trainable = False
            self.models.append(model)

    def predict(self, input_features):
        image_shape = np.squeeze(input_features[self.image_key]).shape
        mask = np.zeros(image_shape)
        mask[:, image_shape[1]//2, image_shape[2]//2, ...] = 1
        input_features['center_array'] = mask
        box_processor = Processors.Box_Images(image_keys=('image',), annotation_key='center_array',
                                              wanted_vals_for_bbox=[1], bounding_box_expansion=(0, 0, 0),
                                              power_val_z=128, power_val_c=512, power_val_r=512,
                                              post_process_keys=('image', 'prediction'))
        x = np.squeeze(input_features[self.image_key])

        # Define the chunking parameters
        step = 128  # size of data input
        chunk_size = 96  # size of the inner section to keep +/-
        inner_section_start = step // 2 - chunk_size // 2  # start index of the inner section
        inner_section_end = step // 2 + chunk_size // 2  # end index of the inner section
        shift = inner_section_end - inner_section_start  # move by the full size of the chunk to get the next center

        # Padding the array to ensure the inner section covers the entire original array
        left_pad = inner_section_start
        right_pad = (inner_section_end - inner_section_start) + step - (x.shape[0] + left_pad) % step
        x_padded = np.pad(x, ((left_pad, right_pad), (0, 0), (0, 0)), mode='edge')
        mask_padded = np.pad(mask, ((left_pad, right_pad), (0, 0), (0, 0)), mode='edge')

        for i, model in enumerate(self.models):
            # Initialize the prediction array
            pred = np.zeros(x_padded.shape + (2,))

            # Start the chunking process
            start = 0
            while start < x_padded.shape[0] - step + 1:  # Ensure we don't go out of bounds
                # Extract the current chunk from the padded arrays
                image_cube = x_padded[start:start + step, ...]
                mask_cube = copy.deepcopy(mask_padded[start:start + step, ...])

                # Skip processing if the mask is empty
                mask_cube[:inner_section_start] = 0
                mask_cube[inner_section_end:] = 0
                if np.max(mask_cube) == 0:
                    start += shift
                    continue

                # Prepare input features for prediction
                temp_input_features = {'image': image_cube, 'center_array': mask_cube}
                box_processor.pre_process(temp_input_features)
                image = temp_input_features['image'][None, ..., None]

                # Make predictions
                pred_cube = model.predict(tf.convert_to_tensor(image))
                temp_input_features['prediction'] = np.squeeze(pred_cube)
                box_processor.post_process(temp_input_features)
                pred_cube = temp_input_features['prediction']

                # Extract the inner section of the predicted chunk (center region)
                pred_cube = pred_cube[inner_section_start:inner_section_end, ...]

                # Insert the processed prediction into the correct location in the output array
                pred[start + inner_section_start:start + inner_section_end, ...] = pred_cube

                # Move the start position for the next chunk by the size of the shift
                start += shift
            # Remove the padding from the prediction array to match the original size
            pred = pred[left_pad:-right_pad, ...] if right_pad > 0 else pred[left_pad:, ...]
            input_features[f"prediction_{i}"] = pred[None, ...]
        return input_features


class DicomReaderWriter(TemplateDicomReader):
    def __init__(self, roi_names, prediction_keys):
        super().__init__(roi_names)
        self.prediction_keys = prediction_keys

    def write_predictions(self, input_features, out_path):
        self.reader.template = 1
        image: SimpleITK.Image
        image = self.reader.dicom_handle
        sitk.WriteImage(image, os.path.join(out_path, 'Image.nii'))
        image_array = sitk.GetArrayFromImage(image)
        np.save(os.path.join(out_path, f'Image.npy'), image_array)
        truth_files = [i for i in os.listdir(out_path) if i.endswith('.mhd')]
        for file in truth_files:
            handle = SimpleITK.ReadImage(os.path.join(out_path, file))
            np.save(os.path.join(out_path, file.replace('.mhd', '.npy')),
                    SimpleITK.GetArrayFromImage(handle).astype('bool'))
        for i, pred_key in enumerate(self.prediction_keys):
            annotations = input_features[pred_key]
            contour_values = np.max(annotations, axis=0)
            while len(contour_values.shape) > 1:
                contour_values = np.max(contour_values, axis=0)
            contour_values[0] = 1
            annotations = annotations[..., contour_values == 1]
            ROI_Names = [self.roi_names[i]]
            np.save(os.path.join(out_path, f'{self.roi_names[i]}.npy'), annotations[..., 1].astype('bool'))
            pred_handle = sitk.GetImageFromArray(annotations[..., 1].astype('int'))
            pred_handle.SetOrigin(image.GetOrigin())
            pred_handle.SetDirection(image.GetDirection())
            pred_handle.SetSpacing(image.GetSpacing())
            pred_handle = sitk.Cast(pred_handle, sitk.sitkUInt8)
            sitk.WriteImage(pred_handle, os.path.join(out_path, f'{self.roi_names[i]}.nii'))
            self.reader.prediction_array_to_RT(prediction_array=annotations,
                                               output_dir=out_path,
                                               ROI_Names=ROI_Names,
                                               write_file=False)
        fid = open(os.path.join(out_path, 'Completed.txt'), 'w+')
        fid.close()


def return_prostate_nodes_model():
    local_path = return_paths()
    model_path_base = os.path.join(local_path, 'Models', 'ProstateNodes')
    models = [
        os.path.join(model_path_base, 'Model_82', 'model.keras'),
        os.path.join(model_path_base, 'Model_85', 'model.keras'),
        os.path.join(model_path_base, 'Model_88', 'model.keras'),
        os.path.join(model_path_base, 'Model_89', 'model.keras'),
        os.path.join(model_path_base, 'Model_94', 'model.keras')
              ]
    prostate_nodes_model = ProstateNodeModelBuilder(image_key='image',
                                                    model_paths=models)
    prostate_nodes_model.set_paths([os.path.join(local_path, 'DICOM', 'ProstateNodes', 'Input'),
                                    r'\\vscifs1\PhysicsQAdata\BMA\Predictions\ProstateNodes\Input'])
    roi_base_name = 'CTV_Pelvis_AI_Prediction'
    prediction_keys = tuple([f'prediction_{i}' for i in range(len(models))])
    template_reader = DicomReaderWriter(roi_names=[
        f"{roi_base_name}_UNC",
        f"{roi_base_name}_A", # Pearl
        f"{roi_base_name}_B", # Rep
        f"{roi_base_name}_C", # Shiv
        f"{roi_base_name}_D" # Wij
    ],
                                        prediction_keys=prediction_keys)
    prostate_nodes_model.set_dicom_reader(template_reader)
    dicom_handle_key = ('primary_handle',)
    mean_value = -35
    standard_deviation_value = 60

    image_processors = [
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.DeepCopyKey(from_keys=('primary_handle',), to_keys=('primary_handle_ref',)),
        Processors.Resampler(desired_output_spacing=(1.25, 1.25, 3), resample_keys=dicom_handle_key,
                             resample_interpolators=('Linear',), post_process_resample_keys=prediction_keys,
                             post_process_original_spacing_keys=tuple(['primary_handle_ref' for _ in range(len(prediction_keys))]),
                             post_process_interpolators=tuple(['Linear' for _ in range(len(prediction_keys))])),
        Processors.SimpleITKImageToArray(nifti_keys=('primary_handle',),
                                         out_keys=('image',),
                                         dtypes=['float32',]),
        Processors.AddByValues(image_keys=('image',), values=(-mean_value,)),
        # Put them on a scale of ~ 0 to max
        Processors.MultiplyByValues(image_keys=('image',),
                                    values=(1 / standard_deviation_value,)),
        Processors.Threshold_Images(image_keys=('image',), lower_bound=-5,
                                    upper_bound=5, divide=False),
        Processors.ExpandDimensions(axis=-1, image_keys=('image',), post_process_keys=('image',)),
        Processors.ExpandDimensions(axis=0, image_keys=('image',), post_process_keys=('image',) + prediction_keys)
    ]
    prostate_nodes_model.set_image_processors(image_processors)
    prediction_processors = [
        # Turn_Two_Class_Three(),
        Processors.Threshold_and_Expand(seed_threshold_values=(0.95, 0.95, 0.95, 0.95, 0.95),
                                        lower_threshold_values=(.45, 0.4, 0.25, 0.45, 0.45),
                                        prediction_keys=prediction_keys),
        Processors.Fill_Binary_Holes(prediction_keys=prediction_keys, dicom_handle_key='primary_handle_ref'),
        Processors.MinimumVolumeandAreaPrediction(prediction_keys=prediction_keys, min_volume=50.0,
                                                  min_area=2.0, max_area=np.inf, pred_axis=[1],
                                                  dicom_handle_key='primary_handle',
                                                  largest_only=True),
    ]
    prostate_nodes_model.set_prediction_processors(prediction_processors)
    return prostate_nodes_model


if __name__ == '__main__':
    pass
