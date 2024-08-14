import os, sys
sys.path.insert(0, os.path.abspath('..'))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *


def return_parotid_model():
    local_path = return_paths()
    parotid_model = BaseModelBuilder(image_key='image',
                                     model_path=os.path.join(local_path, 'Models', 'Parotid', 'Model_39'))
    parotid_model.set_paths([os.path.join(local_path, 'DICOM', 'Parotid', 'Input'),
                             r'\\vscifs1\PhysicsQAdata\BMA\Predictions\Parotid\Input'])
    template_reader = TemplateDicomReader(roi_names=['Parotids_BMA_Program'])
    parotid_model.set_dicom_reader(template_reader)
    dicom_handle_key = ('primary_handle',)
    mean_value = 3.89
    standard_deviation_value = 39.10
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
        Processors.ConvertBodyContourToCentroidLine(body_handle_key='body_handle', out_key='center_handle'),
        Processors.SimpleITKImageToArray(nifti_keys=('primary_handle', 'center_handle',),
                                         out_keys=('image', 'center_array'),
                                         dtypes=['float32', 'int32']),
        Processors.Box_Images(image_keys=('image',), annotation_key='center_array',
                              wanted_vals_for_bbox=[1], bounding_box_expansion=(0, 0, 0),
                              power_val_z=64, power_val_c=256, power_val_r=256, pad_value=-1000,
                              post_process_keys=('image', 'prediction')),
        Processors.Threshold_Images(image_keys=('image',), lower_bound=lower_bounds[0],
                                    upper_bound=upper_bounds[0], divide=False),
        Processors.AddByValues(image_keys=('image',), values=tuple([-i for i in lower_bounds])),
        # Put them on a scale of ~ 0 to max
        Processors.MultiplyByValues(image_keys=('image',),
                                    values=([2 / (upper_bounds[i] - lower_bounds[i])
                                             for i in range(len(upper_bounds))],)),
        Processors.AddByValues(image_keys=('image',), values=tuple([-1.0 for _ in lower_bounds])),
        Processors.ExpandDimensions(axis=-1, image_keys=('image',))
    ]
    parotid_model.set_image_processors(image_processors)
    prediction_processors = [
        # Turn_Two_Class_Three(),
        Processors.Threshold_and_Expand(seed_threshold_value=0.9,
                                        lower_threshold_value=.25),
        Processors.Fill_Binary_Holes(prediction_key='prediction', dicom_handle_key='primary_handle')
    ]
    parotid_model.set_prediction_processors(prediction_processors)
    return parotid_model


if __name__ == '__main__':
    pass
