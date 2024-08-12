import os, sys
sys.path.insert(0, os.path.abspath('..'))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *


def return_pancreas_model():
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    pancreas_model = ModelBuilderFromTemplate(image_key='image',
                                              model_path=os.path.join(model_load_path, 'Pancreas',
                                                                      '2D_DLv3_pancreas_ID2.hdf5'),
                                              model_template=deeplabv3plus(input_shape=(512, 512, 1),
                                                                           backbone="xception",
                                                                           classes=2, final_activation='softmax',
                                                                           windowopt_flag=True,
                                                                           normalization='batch', activation='relu',
                                                                           weights=None).Deeplabv3())
    paths = [
        os.path.join(shared_drive_path, 'Pancreas_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'Pancreas_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'Pancreas_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'Pancreas_Auto_Contour', 'Input_3')
    ]
    pancreas_model.set_paths(paths)
    pancreas_model.set_image_processors([
        Processors.ExpandDimensions(axis=-1, image_keys=('image',)),
        Processors.Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
                                 post_process_keys=('image', 'prediction')), ])
    pancreas_model.set_prediction_processors(
        [ProcessPrediction(prediction_keys=('prediction',), threshold={"1": 0.5}, connectivity={"1": False},
                           extract_main_comp={"1": False}, thread_count=1),
         Postprocess_Pancreas(prediction_keys=('prediction',))])
    pancreas_model.set_dicom_reader(
        TemplateDicomReader(roi_names=['Pancreas_MorfeusLab_v0'],
                            associations={'Pancreas_MorfeusLab_v0': 'Pancreas_MorfeusLab_v0',
                                          'Pancreas': 'Pancreas_MorfeusLab_v0'}))
    return pancreas_model


if __name__ == '__main__':
    pass
