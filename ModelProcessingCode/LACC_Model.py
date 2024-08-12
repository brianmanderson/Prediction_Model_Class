import os, sys
sys.path.insert(0, os.path.abspath('..'))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *


def return_lacc_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    lacc_model = ModelBuilderFromTemplate(image_key='image',
                                          model_path=os.path.join(model_load_path,
                                                                  'LACC',
                                                                  '2D_DLv3_clr_multi_organs_v4.hdf5'),
                                          model_template=deeplabv3plus(input_shape=(512, 512, 1),
                                                                       backbone="xception",
                                                                       classes=13, final_activation='softmax',
                                                                       windowopt_flag=True,
                                                                       normalization='batch', activation='relu',
                                                                       weights=None).Deeplabv3())
    paths = [
        os.path.join(shared_drive_path, 'LACC_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'LACC_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'LACC_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'LACC_Auto_Contour', 'Input_3')
    ]
    lacc_model.set_paths(paths)
    lacc_model.set_image_processors([
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.ExpandDimensions(axis=-1, image_keys=('image',)),
        Focus_on_CT()])
    lacc_model.set_prediction_processors([
        ProcessPrediction(prediction_keys=('prediction',),
                          threshold={"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5, "5": 0.5, "6": 0.5, "7": 0.5, "8": 0.5,
                                     "9": 0.5, "10": 0.5, "11": 0.5, "12": 0.5},
                          connectivity={"1": False, "2": True, "3": True, "4": False, "5": True, "6": False,
                                        "7": True, "8": True, "9": True, "10": True, "11": False, "12": False},
                          extract_main_comp={"1": True, "2": False, "3": False, "4": False, "5": False, "6": False,
                                             "7": False, "8": False, "9": False, "10": False, "11": False, "12": False},
                          thread_count=12, dist={"1": 50}, max_comp={"1": 2}, min_vol={"1": 2000}),
        CombinePredictions(prediction_keys=('prediction',), combine_ids=((7, 8),), closings=(False,)),
        CreateUpperVagina(prediction_keys=('prediction',), class_id=(5,), sup_margin=(20,)),
        CombinePredictions(prediction_keys=('prediction',), combine_ids=((1, 14, 6),), closings=(True,)),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v4' for roi in
                     ["UteroCervix", "Bladder", "Rectum", "Sigmoid", "Vagina", "Parametrium", "Femur_Head_R",
                      "Femur_Head_L",
                      'Kidney_R', 'Kidney_L', 'SpinalCord', 'BowelSpace', 'Femoral Heads', 'Upper_Vagina_2.0cm',
                      'CTVp']]
    else:
        roi_names = ["UteroCervix", "Bladder", "Rectum", "Sigmoid", "Vagina", "Parametrium", "Femur_Head_R",
                     "Femur_Head_L", 'Kidney_R', 'Kidney_L', 'SpinalCord', 'BowelSpace', 'Femoral Heads',
                     'Upper_Vagina_2.0cm', 'CTVp']

    lacc_model.set_dicom_reader(TemplateDicomReader(roi_names=roi_names))
    return lacc_model


if __name__ == '__main__':
    pass
