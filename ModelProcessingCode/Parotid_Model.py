import os, sys
sys.path.insert(0, os.path.abspath('..'))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *


class ParotidModelBuilder(BaseModelBuilder):
    def predict(self, input_features):
        box_processor = Processors.Box_Images(image_keys=('image',), annotation_key='center_array',
                                              wanted_vals_for_bbox=[1], bounding_box_expansion=(0, 0, 0),
                                              power_val_z=64, power_val_c=256, power_val_r=256, pad_value=-5,
                                              post_process_keys=('image', 'prediction'))
        step = 64
        shift = 32
        gap = 16
        x = np.squeeze(input_features[self.image_key])
        mask = input_features['center_array']
        pred = np.zeros((1,) + x.shape + (2,))
        start = 0
        while start < x.shape[0]:
            image_cube, mask_cube = x[start:start + step, ...], mask[start:start + step, ...]
            if np.max(mask_cube) == 0:
                start += shift
                continue
            temp_input_features = {'image': image_cube, 'center_array': mask_cube}
            box_processor.pre_process(temp_input_features)
            image = temp_input_features['image'][None, ..., None]
            pred_cube = self.model.predict(image)
            temp_input_features['prediction'] = np.squeeze(pred_cube)
            box_processor.post_process(temp_input_features)
            pred_cube = temp_input_features['prediction']
            start_gap = gap
            stop_gap = gap
            if start == 0:
                start_gap = 0
            elif start + step >= x[0].shape[1]:
                stop_gap = 0
            if stop_gap != 0:
                pred_cube = pred_cube[start_gap:-stop_gap, ...]
            else:
                pred_cube = pred_cube[start_gap:, ...]
            pred[:, start + start_gap:start + start_gap + pred_cube.shape[0], ...] = pred_cube[None, ...]
            start += shift
        input_features['prediction'] = pred
        x = 1


def return_parotid_model():
    local_path = return_paths()
    parotid_model = ParotidModelBuilder(image_key='image',
                                        model_path=os.path.join(local_path, 'Models', 'Parotid', 'Model_27'))
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
        Processors.ConvertBodyContourToCentroidLine(body_handle_key='body_handle', out_key='center_handle',
                                                    extent_evaluated=1),
        Processors.SimpleITKImageToArray(nifti_keys=('primary_handle', 'center_handle',),
                                         out_keys=('image', 'center_array'),
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
        Processors.Threshold_and_Expand(seed_threshold_value=0.9,
                                        lower_threshold_value=.05),
        Processors.Fill_Binary_Holes(prediction_key='prediction', dicom_handle_key='primary_handle_ref')
    ]
    parotid_model.set_prediction_processors(prediction_processors)
    return parotid_model


if __name__ == '__main__':
    pass
