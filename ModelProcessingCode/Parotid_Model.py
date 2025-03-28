import os, sys
sys.path.insert(0, os.path.abspath('..'))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import copy
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *


class ParotidModelBuilder(BaseModelBuilder):
    def predict(self, input_features):
        centroid_processor = Processors.ConvertBodyContourToCentroidLine(body_handle_key='body_handle',
                                                                         out_key='center_handle',
                                                                         extent_evaluated=1)
        box_processor = Processors.Box_Images(image_keys=('image',), annotation_key='body_array',
                                              wanted_vals_for_bbox=[1], bounding_box_expansion=(0, 0, 0),
                                              power_val_z=64, power_val_c=256, power_val_r=256, pad_value=-5,
                                              post_process_keys=('image', 'prediction'))
        x = np.squeeze(input_features[self.image_key])
        mask = sitk.GetArrayFromImage(input_features['body_handle'])

        # Define the chunking parameters
        step = 64  # size of data input
        chunk_size = 4  # size of the inner section to keep +/-
        inner_section_start = step // 2 - chunk_size  # start index of the inner section
        inner_section_end = step // 2 + chunk_size  # end index of the inner section
        shift = inner_section_end - inner_section_start  # move by the full size of the chunk to get the next center

        # Padding the array to ensure the inner section covers the entire original array
        left_pad = inner_section_start
        right_pad = (inner_section_end - inner_section_start) + step - (x.shape[0] + left_pad) % step
        x_padded = np.pad(x, ((left_pad, right_pad), (0, 0), (0, 0)), mode='edge')
        mask_padded = np.pad(mask, ((left_pad, right_pad), (0, 0), (0, 0)), mode='edge')

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
            temp_input_features = {'image': image_cube, 'body_handle': sitk.GetImageFromArray(mask_cube)}
            centroid_processor.pre_process(temp_input_features)
            temp_input_features['body_array'] = sitk.GetArrayFromImage(temp_input_features['center_handle'])
            box_processor.pre_process(temp_input_features)
            image = temp_input_features['image'][None, ..., None]

            # Make predictions
            pred_cube = self.model.predict(image)
            temp_input_features['prediction'] = np.squeeze(pred_cube)
            box_processor.post_process(temp_input_features)
            pred_cube = temp_input_features['prediction']

            # Extract the inner section of the predicted chunk (center region)
            pred_cube = pred_cube[inner_section_start:inner_section_end, ...]

            # Insert the processed prediction into the correct location in the output array
            pred[start + inner_section_start:start + inner_section_start + pred_cube.shape[0], ...] = pred_cube[
                None, ...]

            # Move the start position for the next chunk by the size of the shift
            start += shift
        # Remove the padding from the prediction array to match the original size
        pred = pred[left_pad:-right_pad, ...] if right_pad > 0 else pred[left_pad:, ...]
        input_features['prediction'] = pred[None, ...]
        return input_features


def return_parotid_model():
    local_path = return_paths()
    parotid_model = ParotidModelBuilder(image_key='image',
                                        model_path=os.path.join(local_path, 'Models', 'Parotid', 'Model_27',
                                                                'model.keras'))
    parotid_model.set_paths([os.path.join(local_path, 'DICOM', 'Parotid', 'Input'),
                             r'\\vscifs1\PhysicsQAdata\BMA\Predictions\Parotid\Input'])
    template_reader = TemplateDicomReader(roi_names=['Parotids_BMA_Program'])
    parotid_model.set_dicom_reader(template_reader)
    dicom_handle_key = ('primary_handle',)
    mean_value = 0
    standard_deviation_value = 40
    lower_bounds = (mean_value - 2 * standard_deviation_value,)
    upper_bounds = (mean_value + 2 * standard_deviation_value,)
    image_processors = [
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.DeepCopyKey(from_keys=('primary_handle',), to_keys=('primary_handle_ref',)),
        Processors.Resampler(desired_output_spacing=(0.9765, 0.9765, 3), resample_keys=dicom_handle_key,
                             resample_interpolators=('Linear',), post_process_resample_keys=('prediction',),
                             post_process_original_spacing_keys=('primary_handle_ref',),
                             post_process_interpolators=('Linear',)),
        Processors.IdentifyBodyContour(image_key=dicom_handle_key[0],
                                       lower_threshold=-100, upper_threshold=10000,
                                       out_label='body_handle'),
        Processors.SimpleITKImageToArray(nifti_keys=('primary_handle', 'body_handle',),
                                         out_keys=('image', 'body_array'),
                                         dtypes=['float32', 'int32']),
        Processors.AddByValues(image_keys=('image',), values=(-mean_value,)),
        # Put them on a scale of ~ 0 to max
        Processors.MultiplyByValues(image_keys=('image',),
                                    values=(1 / standard_deviation_value,)),
        Processors.Threshold_Images(image_keys=('image',), lower_bound=-5,
                                    upper_bound=5, divide=False),
        Processors.ExpandDimensions(axis=-1, image_keys=('image',), post_process_keys=('image',)),
        Processors.ExpandDimensions(axis=0, image_keys=('image',), post_process_keys=('image', 'prediction'))
    ]
    parotid_model.set_image_processors(image_processors)
    prediction_processors = [
        # Turn_Two_Class_Three(),
        Processors.Threshold_and_Expand(seed_threshold_values=(0.95,),
                                        lower_threshold_values=(.5,)),
        Processors.Fill_Binary_Holes(prediction_key='prediction', dicom_handle_key='primary_handle_ref')
    ]
    parotid_model.set_prediction_processors(prediction_processors)
    return parotid_model


if __name__ == '__main__':
    pass
