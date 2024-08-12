import os, sys
sys.path.insert(0, os.path.abspath('..'))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *


def return_ctvn_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    ctvn_model = ModelBuilderFromTemplate(image_key='image',
                                          model_path=os.path.join(model_load_path,
                                                                  'CTVN',
                                                                  'DLv3_model_CTVN_v2_Trial_89.hdf5'),
                                          model_template=deeplabv3plus(input_shape=(512, 512, 1),
                                                                       backbone="xception",
                                                                       classes=3, final_activation='softmax',
                                                                       windowopt_flag=True,
                                                                       normalization='batch', activation='relu',
                                                                       weights=None).Deeplabv3())
    paths = [
        os.path.join(shared_drive_path, 'CTVN_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'CTVN_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'CTVN_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'CTVN_Auto_Contour', 'Input_3')
    ]
    ctvn_model.set_paths(paths)
    ctvn_model.set_image_processors([
        # Normalize_Images(keys=('image',), mean_values=(-17.0,), std_values=(63.0,)),
        # Threshold_Images(image_keys=('image',), lower_bounds=(-3.55,), upper_bounds=(3.55,), divides=(False,)),
        # AddByValues(image_keys=('image',), values=(3.55,)),
        # DivideByValues(image_keys=('image', 'image'), values=(7.10, 1 / 255)),
        Processors.ExpandDimensions(axis=-1, image_keys=('image',)),
        # RepeatChannel(num_repeats=3, axis=-1, image_keys=('image',)),
        Processors.Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
                                 post_process_keys=('image', 'prediction')),
    ])
    ctvn_model.set_prediction_processors([
        ProcessPrediction(prediction_keys=('prediction',),
                          threshold={"1": 0.5, "2": 0.5},
                          connectivity={"1": False, "2": False},
                          extract_main_comp={"1": True, "2": True},
                          thread_count=2, dist={"1": 50, "2": 50}, max_comp={"1": 2, "2": 2},
                          min_vol={"1": 2000, "2": 2000}),
        CombinePredictions(prediction_keys=('prediction',), combine_ids=((1, 2),), closings=(True,)),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v2' for roi in ["CTVn", "CTV_PAN", "Nodal_CTV"]]
    else:
        roi_names = ["CTVn", "CTV_PAN", "Nodal_CTV"]

    ctvn_model.set_dicom_reader(TemplateDicomReader(roi_names=roi_names))
    return ctvn_model


if __name__ == '__main__':
    pass
