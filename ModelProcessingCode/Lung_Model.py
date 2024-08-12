import os, sys
sys.path.insert(0, os.path.abspath('..'))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *


def return_lung_model():
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    lung_model = BaseModelBuilder(image_key='image',
                                  model_path=os.path.join(model_load_path, 'Lungs', 'Covid_Four_Model_50'),
                                  Bilinear_model=BilinearUpsampling, loss=None, loss_weights=None)
    lung_model.set_dicom_reader(TemplateDicomReader(roi_names=['Ground Glass_BMA_Program_2', 'Lung_BMA_Program_2']))
    lung_model.set_paths([
        # r'H:\AutoModels\Lung\Input_4',
        os.path.join(shared_drive_path, 'Lungs_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'Lungs', 'Input_3'),
        os.path.join(raystation_clinical_path, 'Lungs_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'Lungs_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'BMAnderson', 'Test', 'Input_3')
    ])
    lung_model.set_image_processors([
        Processors.AddByValues(image_keys=('image',), values=(751,)),
        Processors.DivideByValues(image_keys=('image',), values=(200,)),
        Processors.Threshold_Images(image_keys=('image',), lower_bounds=(-5,), upper_bounds=(5,), divides=(False,)),
        Processors.DivideByValues(image_keys=('image',), values=(5,)),
        Processors.ExpandDimensions(axis=-1, image_keys=('image',)),
        Processors.RepeatChannel(num_repeats=3, axis=-1, image_keys=('image',)),
        Processors.Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
                                 post_process_keys=('image', 'prediction')),
    ])
    lung_model.set_prediction_processors([
        Processors.ArgMax(image_keys=('prediction',), axis=-1),
        Processors.To_Categorical(num_classes=3, annotation_keys=('prediction',)),
        Processors.CombineLungLobes(prediction_key='prediction', dicom_handle_key='primary_handle')
    ])
    return lung_model


if __name__ == '__main__':
    pass
