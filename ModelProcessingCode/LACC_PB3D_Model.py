import os, sys
sys.path.insert(0, os.path.abspath('..'))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *


def return_lacc_pb3D_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    required_size = (48, 192, 192)
    lacc_model = PredictWindowSliding(image_key='image',
                                      model_path=os.path.join(model_load_path,
                                                              'LACC_3D',
                                                              'BasicUNet3D_Trial_24.hdf5'),
                                      model_template=BasicUnet3D(input_tensor=None, input_shape=required_size + (1,),
                                                                 classes=13, classifier_activation="softmax",
                                                                 activation="leakyrelu",
                                                                 normalization="group", nb_blocks=2,
                                                                 nb_layers=5, dropout='standard',
                                                                 filters=32, dropout_rate=0.1,
                                                                 skip_type='att', bottleneck='standard').get_net(),
                                      nb_label=13, required_size=required_size
                                      )
    paths = [
        os.path.join(shared_drive_path, 'LACC_3D_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'LACC_3D_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'LACC_3D_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'LACC_3D_Auto_Contour', 'Input_3')
    ]

    lacc_model.set_paths(paths)
    lacc_model.set_image_processors([
        Processors.Threshold_Images(image_keys=('image',), lower_bounds=(-1000,), upper_bounds=(1500,), divides=(False,)),
        CreateExternal(image_key='image', output_key='external', threshold_value=-250.0, mask_value=1),
        Processors.DeepCopyKey(from_keys=('external',), to_keys=('og_external',)),
        Processors.Per_Image_MinMax_Normalization(image_keys=('image',), threshold_value=1.0),
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.Resampler(resample_keys=('image', 'external'),
                  resample_interpolators=('Linear', 'Nearest'),
                  desired_output_spacing=[1.17, 1.17, 3.0],
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
    lacc_model.set_prediction_processors([
        Processors.ExpandDimensions(image_keys=('og_external',), axis=-1),
        Processors.MaskOneBasedOnOther(guiding_keys=tuple(['og_external' for i in range(1, 13)]),
                            changing_keys=tuple(['prediction' for i in range(1, 13)]),
                            guiding_values=tuple([0 for i in range(1, 13)]),
                            mask_values=tuple([0 for i in range(1, 13)])),
        ProcessPrediction(prediction_keys=('prediction',),
                          threshold={"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5, "5": 0.5, "6": 0.5, "7": 0.5, "8": 0.5,
                                     "9": 0.5, "10": 0.5, "11": 0.5, "12": 0.5},
                          connectivity={"1": False, "2": True, "3": True, "4": False, "5": True, "6": False,
                                        "7": True, "8": True, "9": True, "10": True, "11": True, "12": True},
                          extract_main_comp={"1": True, "2": False, "3": False, "4": True, "5": False, "6": True,
                                             "7": False, "8": False, "9": False, "10": False, "11": False, "12": False},
                          dist={"1": 50, "4": 100, "6": None}, max_comp={"1": 2, "4": 3, "6": 2},
                          min_vol={"1": 2000, "4": 2000, "6": 2000}, thread_count=12),
        CombinePredictions(prediction_keys=('prediction',), combine_ids=((7, 8),), closings=(False,)),
        CreateUpperVagina(prediction_keys=('prediction',), class_id=(5,), sup_margin=(20,)),
        CombinePredictions(prediction_keys=('prediction',), combine_ids=((1, 14, 6),), closings=(True,)),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v5' for roi in
                     ["UteroCervix", "Bladder", "Rectum", "Sigmoid", "Vagina", "Parametrium", "Femur_Head_R",
                      "Femur_Head_L", 'Kidney_R', 'Kidney_L', 'SpinalCord', 'BowelSpace', 'Femoral Heads',
                      'Upper_Vagina_2.0cm', 'CTVp']]
    else:
        roi_names = ["UteroCervix", "Bladder", "Rectum", "Sigmoid", "Vagina", "Parametrium", "Femur_Head_R",
                     "Femur_Head_L", 'Kidney_R', 'Kidney_L', 'SpinalCord', 'BowelSpace', 'Femoral Heads',
                     'Upper_Vagina_2.0cm', 'CTVp']

    lacc_model.set_dicom_reader(TemplateDicomReader(roi_names=roi_names))
    return lacc_model


if __name__ == '__main__':
    pass
