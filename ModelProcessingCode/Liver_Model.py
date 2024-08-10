import os, sys
sys.path.insert(0, os.path.abspath('..'))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *


def return_liver_model():
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    liver_model = BaseModelBuilder(image_key='image',
                                   model_path=os.path.join(model_load_path,
                                                           'Liver',
                                                           'weights-improvement-512_v3_model_xception-36.hdf5'),
                                   Bilinear_model=BilinearUpsampling, loss=None, loss_weights=None)

    # liver_model = BaseModelBuilderGraph(image_key='image',
    #                                    model_path=os.path.join(model_load_path,
    #                                                            'Liver',
    #                                                            'weights-improvement-512_v3_model_xception-36.hdf5'),
    #                                    Bilinear_model=BilinearUpsampling, loss=None, loss_weights=None)
    paths = [
        r'H:\AutoModels\Liver\Input_4',
        os.path.join(morfeus_path, 'BMAnderson', 'Test', 'Input_4'),
        os.path.join(shared_drive_path, 'Liver_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'Liver_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'Liver_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'Liver_Auto_Contour', 'Input_3')
    ]
    liver_model.set_paths(paths)
    liver_model.set_image_processors([
        Processors.Threshold_Images(image_keys=('image',), lower_bounds=(-100,), upper_bounds=(300,), divides=(False,)),
        Processors.AddByValues(image_keys=('image',), values=(100,)),
        Processors.DivideByValues(image_keys=('image', 'image'), values=(400, 1 / 255)),
        Processors.ExpandDimensions(axis=-1, image_keys=('image',)),
        Processors.RepeatChannel(num_repeats=3, axis=-1, image_keys=('image',)),
        Processors.Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
                                 post_process_keys=('image', 'prediction')),
        Processors.VGGNormalize(image_keys=('image',))])
    liver_model.set_prediction_processors([
        Processors.Threshold_Prediction(threshold=0.5, single_structure=True, is_liver=True, prediction_keys=('prediction',))])
    liver_model.set_dicom_reader(TemplateDicomReader(roi_names=['Liver_BMA_Program_4'],
                                                     associations={'Liver_BMA_Program_4': 'Liver', 'Liver': 'Liver'}))
    return liver_model


if __name__ == '__main__':
    pass
