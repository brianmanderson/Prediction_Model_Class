import os, sys
sys.path.insert(0, os.path.abspath('..'))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *


def return_liver_pb3D_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    required_size = (24, 256, 256)
    liver_model = PredictWindowSliding(image_key='image',
                                          model_path=os.path.join(model_load_path,
                                                                  'Liver_3D',
                                                                  'BasicUNet3D_Trial_12.hdf5'),
                                          model_template=BasicUnet3D(input_tensor=None,
                                                                     input_shape=required_size + (1,),
                                                                     classes=2, classifier_activation="softmax",
                                                                     activation="leakyrelu",
                                                                     normalization="group", nb_blocks=3,
                                                                     nb_layers=4, dropout='standard',
                                                                     filters=32, dropout_rate=0.1,
                                                                     skip_type='concat',
                                                                     bottleneck='standard').get_net(),
                                          nb_label=2, required_size=required_size, gaussiance_map=True,
                                          sigma_scale=0.250
                                          )
    paths = [
        os.path.join(shared_drive_path, 'Liver_3D_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'Liver_3D_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'Liver_3D_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'Liver_3D_Auto_Contour', 'Input_3')
    ]
    liver_model.set_paths(paths)
    liver_model.set_image_processors([
        CreateExternal(image_key='image', output_key='external', threshold_value=-250.0, mask_value=1),
        Processors.DeepCopyKey(from_keys=('external',), to_keys=('og_external',)),
        Processors.Threshold_Images(image_keys=('image',), lower_bounds=(-100,), upper_bounds=(300,), divides=(False,)),
        Processors.Per_Image_MinMax_Normalization(image_keys=('image',), threshold_value=1.0),
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.Resampler(resample_keys=('image', 'external'),
                  resample_interpolators=('Linear', 'Nearest'),
                  desired_output_spacing=[None, None, 2.5],
                  post_process_resample_keys=('prediction',),
                  post_process_original_spacing_keys=('primary_handle',),
                  post_process_interpolators=('Linear',)),
        Box_Images(bounding_box_expansion=(0, 0, 0), image_keys=('image',),
                   annotation_key='external', wanted_vals_for_bbox=(1,),
                   power_val_z=required_size[0], power_val_r=required_size[1], power_val_c=required_size[2],
                   post_process_keys=('prediction',)),
        Processors.ExpandDimensions(image_keys=('image',), axis=-1),
        Processors.ExpandDimensions(image_keys=('image',), axis=0),
        Processors.SqueezeDimensions(post_prediction_keys=('prediction',))
    ])
    liver_model.set_prediction_processors([
        Processors.ExpandDimensions(image_keys=('og_external',), axis=-1),
        Processors.MaskOneBasedOnOther(guiding_keys=('og_external',),
                            changing_keys=('prediction',),
                            guiding_values=(0,),
                            mask_values=(0,)),
        ProcessPrediction(prediction_keys=('prediction',),
                          threshold={"1": 0.5},
                          connectivity={"1": True},
                          extract_main_comp={"1": False},
                          thread_count=1, dist={"1": None}, max_comp={"1": 1}, min_vol={"1": 5000}),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v0' for roi in ["Liver"]]
    else:
        roi_names = ['Liver']

    liver_model.set_dicom_reader(TemplateDicomReader(roi_names=roi_names))
    return liver_model


if __name__ == '__main__':
    pass
