import shutil, os, sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from functools import partial
from Image_Processors_Module.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Image_Processors_Module.src.Processors.MakeTFRecordProcessors import *
from Dicom_RT_and_Images_to_Mask.src.DicomRTTool import DicomReaderWriter
import tensorflow as tf
from Bilinear_Dsc import BilinearUpsampling

from Image_Processors_Utils.Image_Processor_Utils import ProcessPrediction, Postprocess_Pancreas, Normalize_Images, \
    Threshold_Images, DilateBinary, Focus_on_CT, CombinePredictions, CreateUpperVagina, CreateExternal

import SimpleITK as sitk

# this submodule is private (ask @guatavita Github)
from networks.DeepLabV3plus import *
from networks.UNet3D import *


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = tf.compat.v1.keras.backend.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.compat.v1.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.compat.v1.keras.backend.clip(y_pred, tf.compat.v1.keras.backend.epsilon(),
                                                 1 - tf.compat.v1.keras.backend.epsilon())
        # calc
        loss = y_true * tf.compat.v1.keras.backend.log(y_pred) * weights
        loss = -tf.compat.v1.keras.backend.sum(loss, -1)
        return loss

    return loss


def dice_coef_3D(y_true, y_pred, smooth=0.0001):
    intersection = tf.keras.backend.sum(y_true[..., 1:] * y_pred[..., 1:])
    union = tf.keras.backend.sum(y_true[..., 1:]) + tf.keras.backend.sum(y_pred[..., 1:])
    return (2. * intersection + smooth) / (union + smooth)


def find_base_dir():
    base_path = '.'
    for _ in range(20):
        if 'Morfeus' in os.listdir(base_path):
            break
        else:
            base_path = os.path.join(base_path, '..')
    return base_path


def return_paths():
    try:
        os.listdir('\\\\mymdafiles\\di_data1\\')
        morfeus_path = '\\\\mymdafiles\\di_data1\\Morfeus\\'
        shared_drive_path = '\\\\mymdafiles\\ro-ADMIN\\SHARED\\Radiation physics\\BMAnderson\\Auto_Contour_Sites\\'
        raystation_clinical_path = '\\\\mymdafiles\\ou-radonc\\Raystation\\Clinical\\Auto_Contour_Sites\\'
        raystation_research_path = '\\\\mymdafiles\\ou-radonc\\Raystation\\Research\\Auto_Contour_Sites\\'
    except:
        desktop_path = find_base_dir()
        morfeus_path = os.path.join(desktop_path, 'Morfeus')
        shared_drive_path = os.path.abspath(os.path.join(desktop_path, 'Shared_Drive', 'Auto_Contour_Sites'))
        raystation_clinical_path = os.path.abspath(
            os.path.join(desktop_path, 'Raystation_LDrive', 'Clinical', 'Auto_Contour_Sites'))
        raystation_research_path = os.path.abspath(
            os.path.join(desktop_path, 'Raystation_LDrive', 'Research', 'Auto_Contour_Sites'))
    model_load_path = os.path.join('.', 'Models')
    if not os.path.exists(model_load_path):
        model_load_path = os.path.join(morfeus_path, 'Auto_Contour_Sites', 'Models')
    return morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path


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
        Threshold_Images(image_keys=('image',), lower_bounds=(-100,), upper_bounds=(300,), divides=(False,)),
        AddByValues(image_keys=('image',), values=(100,)),
        DivideByValues(image_keys=('image', 'image'), values=(400, 1 / 255)),
        ExpandDimensions(axis=-1, image_keys=('image',)),
        RepeatChannel(num_repeats=3, axis=-1, image_keys=('image',)),
        Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
                                 post_process_keys=('image', 'prediction')),
        VGGNormalize(image_keys=('image',))])
    liver_model.set_prediction_processors([
        Threshold_Prediction(threshold=0.5, single_structure=True, is_liver=True, prediction_keys=('prediction',))])
    liver_model.set_dicom_reader(TemplateDicomReader(roi_names=['Liver_BMA_Program_4'],
                                                     associations={'Liver_BMA_Program_4': 'Liver', 'Liver': 'Liver'}))
    return liver_model


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
        AddByValues(image_keys=('image',), values=(751,)),
        DivideByValues(image_keys=('image',), values=(200,)),
        Threshold_Images(image_keys=('image',), lower_bounds=(-5,), upper_bounds=(5,), divides=(False,)),
        DivideByValues(image_keys=('image',), values=(5,)),
        ExpandDimensions(axis=-1, image_keys=('image',)),
        RepeatChannel(num_repeats=3, axis=-1, image_keys=('image',)),
        Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
                                 post_process_keys=('image', 'prediction')),
    ])
    lung_model.set_prediction_processors([
        ArgMax(image_keys=('prediction',), axis=-1),
        To_Categorical(num_classes=3, annotation_keys=('prediction',)),
        CombineLungLobes(prediction_key='prediction', dicom_handle_key='primary_handle')
    ])
    return lung_model


def return_liver_lobe_model():
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    liver_lobe_model = PredictLobes(image_key='image', loss=partial(weighted_categorical_crossentropy),
                                    loss_weights=[0.14, 10, 7.6, 5.2, 4.5, 3.8, 5.1, 4.4, 2.7],
                                    model_path=os.path.join(model_load_path, 'Liver_Lobes', 'Model_397'),
                                    Bilinear_model=BilinearUpsampling)
    liver_lobe_model.set_dicom_reader(EnsureLiverPresent(wanted_roi='Liver_BMA_Program_4',
                                                         liver_folder=os.path.join(raystation_clinical_path,
                                                                                   'Liver_Auto_Contour', 'Input_3'),
                                                         associations={'Liver_BMA_Program_4': 'Liver_BMA_Program_4',
                                                                       'Liver': 'Liver_BMA_Program_4'},
                                                         roi_names=['Liver_Segment_{}_BMAProgram3'.format(i)
                                                                    for i in range(1, 5)] +
                                                                   ['Liver_Segment_5-8_BMAProgram3']))
    liver_lobe_model.set_paths([
        # r'H:\AutoModels\Lobes\Input_4',
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'Liver_Segments_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'Liver_Segments_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'Liver_Segments_Auto_Contour', 'Input_3'),
    ])
    liver_lobe_model.set_image_processors([
        Normalize_to_annotation(image_key='image', annotation_key='annotation', annotation_value_list=(1,)),
        Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image', 'annotation')),
        CastData(image_keys=('image', 'annotation'), dtypes=('float32', 'int')),
        AddSpacing(spacing_handle_key='primary_handle'),
        DeepCopyKey(from_keys=('annotation',), to_keys=('og_annotation',)),
        Resampler(resample_keys=('image', 'annotation'), resample_interpolators=('Linear', 'Nearest'),
                  desired_output_spacing=[None, None, 5.0], post_process_resample_keys=('prediction',),
                  post_process_original_spacing_keys=('primary_handle',), post_process_interpolators=('Linear',)),
        Box_Images(bounding_box_expansion=(10, 10, 10), image_keys=('image',), annotation_key='annotation',
                   wanted_vals_for_bbox=(1,), power_val_z=64, power_val_r=320, power_val_c=384,
                   post_process_keys=('image', 'annotation', 'prediction'), pad_value=0),
        ExpandDimensions(image_keys=('image', 'annotation'), axis=0),
        ExpandDimensions(image_keys=('image', 'annotation', 'og_annotation'), axis=-1),
        Threshold_Images(image_keys=('image',), lower_bounds=(-5,), upper_bounds=(5,), divides=(False,)),
        DivideByValues(image_keys=('image',), values=(10,)),
        MaskOneBasedOnOther(guiding_keys=('annotation',), changing_keys=('image',), guiding_values=(0,),
                            mask_values=(0,)),
        CreateTupleFromKeys(image_keys=('image', 'annotation'), output_key='combined'),
        SqueezeDimensions(post_prediction_keys=('image', 'annotation', 'prediction'))
    ])
    liver_lobe_model.set_prediction_processors([
        MaskOneBasedOnOther(guiding_keys=('og_annotation',), changing_keys=('prediction',), guiding_values=(0,),
                            mask_values=(0,)),
        SqueezeDimensions(image_keys=('og_annotation',)),
        Threshold_and_Expand_New(seed_threshold_value=[.9, .9, .9, .9, .9],
                                 lower_threshold_value=[.75, .9, .25, .2, .75],
                                 prediction_key='prediction', ground_truth_key='og_annotation',
                                 dicom_handle_key='primary_handle')
    ])
    return liver_lobe_model


def return_liver_disease_model():
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    liver_disease = PredictDiseaseAblation(image_key='combined',
                                           model_path=os.path.join(model_load_path,
                                                                   'Liver_Disease_Ablation',
                                                                   'Model_42'),
                                           Bilinear_model=BilinearUpsampling, loss_weights=None, loss=None)
    liver_disease.set_paths([
        r'H:\AutoModels\Liver_Disease\Input_3',
        os.path.join(morfeus_path, 'Auto_Contour_Sites',
                     'Liver_Disease_Ablation_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'Liver_Disease_Ablation_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'Liver_Disease_Ablation_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'BMAnderson', 'Test', 'Input_5')
    ])
    liver_disease.set_image_processors([
        DeepCopyKey(from_keys=('annotation',), to_keys=('og_annotation',)),
        Normalize_to_annotation(image_key='image', annotation_key='annotation',
                                annotation_value_list=(1,), mirror_max=True),
        AddSpacing(spacing_handle_key='primary_handle'),
        Resampler(resample_keys=('image', 'annotation'),
                  resample_interpolators=('Linear', 'Nearest'),
                  desired_output_spacing=[None, None, 1.0],
                  post_process_resample_keys=('prediction',),
                  post_process_original_spacing_keys=('primary_handle',),
                  post_process_interpolators=('Linear',)),
        Box_Images(bounding_box_expansion=(5, 20, 20), image_keys=('image',),
                   annotation_key='annotation', wanted_vals_for_bbox=(1,),
                   power_val_z=2 ** 4, power_val_r=2 ** 5, power_val_c=2 ** 5),
        Threshold_Images(lower_bounds=(-10,), upper_bounds=(10,), divides=(True,), image_keys=('image',)),
        ExpandDimensions(image_keys=('image', 'annotation'), axis=0),
        ExpandDimensions(image_keys=('image', 'annotation'), axis=-1),
        MaskOneBasedOnOther(guiding_keys=('annotation',),
                            changing_keys=('image',),
                            guiding_values=(0,),
                            mask_values=(0,)),
        CombineKeys(image_keys=('image', 'annotation'), output_key='combined'),
        SqueezeDimensions(post_prediction_keys=('image', 'annotation', 'prediction'))
    ])
    liver_disease.set_dicom_reader(EnsureLiverPresent(wanted_roi='Liver_BMA_Program_4',
                                                      roi_names=['Liver_Disease_Ablation_BMA_Program_0'],
                                                      liver_folder=os.path.join(raystation_clinical_path,
                                                                                'Liver_Auto_Contour', 'Input_3'),
                                                      associations={'Liver_BMA_Program_4': 'Liver_BMA_Program_4',
                                                                    'Liver': 'Liver_BMA_Program_4'}))
    liver_disease.set_prediction_processors([
        Threshold_and_Expand(seed_threshold_value=0.55, lower_threshold_value=.3, prediction_key='prediction'),
        Fill_Binary_Holes(prediction_key='prediction', dicom_handle_key='primary_handle'),
        ExpandDimensions(image_keys=('og_annotation',), axis=-1),
        MaskOneBasedOnOther(guiding_keys=('og_annotation',), changing_keys=('prediction',),
                            guiding_values=(0,), mask_values=(0,)),
        MinimumVolumeandAreaPrediction(min_volume=0.25, prediction_key='prediction',
                                       dicom_handle_key='primary_handle')
    ])
    return liver_disease


def return_parotid_model():
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    partotid_model = {'model_path': os.path.join(model_load_path, 'Parotid', 'whole_model'),
                      'roi_names': ['Parotid_L_BMA_Program_4', 'Parotid_R_BMA_Program_4'],
                      'dicom_paths': [  # os.path.join(shared_drive_path,'Liver_Auto_Contour','Input_3')
                          os.path.join(morfeus_path, 'Auto_Contour_Sites', 'Parotid_Auto_Contour', 'Input_3'),
                          os.path.join(raystation_clinical_path, 'Parotid_Auto_Contour', 'Input_3'),
                          os.path.join(raystation_research_path, 'Parotid_Auto_Contour', 'Input_3')
                      ],
                      'file_loader': TemplateDicomReader(roi_names=None),
                      'image_processors': [NormalizeParotidMR(image_keys=('image',)),
                                           ExpandDimensions(axis=-1, image_keys=('image',)),
                                           RepeatChannel(num_repeats=3, axis=-1, image_keys=('image',)),
                                           Ensure_Image_Proportions(image_rows=256, image_cols=256,
                                                                    image_keys=('image',),
                                                                    post_process_keys=('image', 'prediction')),
                                           ],
                      'prediction_processors': [
                          # Turn_Two_Class_Three(),
                          Threshold_and_Expand(seed_threshold_value=0.9,
                                               lower_threshold_value=.5),
                          Fill_Binary_Holes(prediction_key='prediction', dicom_handle_key='primary_handle')]
                      }
    return partotid_model


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
        ExpandDimensions(axis=-1, image_keys=('image',)),
        Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
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


def return_cyst_model():
    # TODO Add connectivity to MaskOneBasedOnOther to keep Cyst that is outside Pancreas

    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    pancreas_cyst = PredictCyst(image_key='combined', model_path=os.path.join(model_load_path, 'Cyst',
                                                                              'HybridDLv3_model_Trial_62.hdf5'),
                                model_template=deeplabv3plus(nb_blocks=9, nb_layers=2,
                                                             backbone='mobilenetv2',
                                                             input_shape=(32, 128, 128, 1),
                                                             classes=2, final_activation='softmax',
                                                             activation='swish', normalization='group',
                                                             windowopt_flag=False, nb_output=3,
                                                             add_squeeze=True, add_mask=True,
                                                             dense_decoding=False,
                                                             transition_pool=False, ds_conv=True,
                                                             weights=os.path.join(model_load_path, 'Cyst',
                                                                                  'HybridDLv3_model_Trial_62.hdf5'),
                                                             ).HybridDeeplabv3())
    paths = [
        os.path.join(shared_drive_path, 'Cyst_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'Cyst_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'Cyst_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'Cyst_Auto_Contour', 'Input_3')
    ]
    pancreas_cyst.set_paths(paths)
    pancreas_cyst.set_image_processors([
        DeepCopyKey(from_keys=('annotation',), to_keys=('og_annotation',)),
        Normalize_Images(keys=('image',), mean_values=(21.0,), std_values=(24.0,)),
        Threshold_Images(image_keys=('image',), lower_bounds=(-3.55,), upper_bounds=(3.55,), divides=(False,)),
        AddByValues(image_keys=('image',), values=(3.55,)),
        DivideByValues(image_keys=('image', 'image'), values=(7.10, 1 / 255)),
        AddSpacing(spacing_handle_key='primary_handle'),
        Resampler(resample_keys=('image', 'annotation'),
                  resample_interpolators=('Linear', 'Nearest'),
                  desired_output_spacing=[1.0, 1.0, 3.0],
                  post_process_resample_keys=('prediction',),
                  post_process_original_spacing_keys=('primary_handle',),
                  post_process_interpolators=('Linear',)),
        DilateBinary(image_keys=('annotation',), radius=(5,)),
        Box_Images(bounding_box_expansion=(5, 20, 20), image_keys=('image',),
                   annotation_key='annotation', wanted_vals_for_bbox=(1,),
                   power_val_z=2 ** 4, power_val_r=2 ** 5, power_val_c=2 ** 5),
        ExpandDimensions(image_keys=('image', 'annotation'), axis=0),
        ExpandDimensions(image_keys=('image', 'annotation'), axis=-1),
        CombineKeys(image_keys=('image', 'annotation'), output_key='combined'),
        SqueezeDimensions(post_prediction_keys=('image', 'annotation', 'prediction'))
    ])
    pancreas_cyst.set_dicom_reader(EnsureLiverPresent(wanted_roi='Pancreas_DLv3_v0',
                                                      roi_names=['Cyst_HybridDLv3_v0'],
                                                      liver_folder=os.path.join(morfeus_path, 'Bastien', 'RayStation',
                                                                                'Pancreas', 'Input_3'),
                                                      associations={'Pancreas_Ezgi': 'Pancreas_DLv3_v0',
                                                                    'Pancreas': 'Pancreas_DLv3_v0',
                                                                    'Pancreas_DLv3_v0': 'Pancreas_DLv3_v0',
                                                                    'Pancreas_MONAI_v0': 'Pancreas_DLv3_v0',
                                                                    'Pancreas_RSDL_v0': 'Pancreas_DLv3_v0',
                                                                    }))
    pancreas_cyst.set_prediction_processors([
        Threshold_and_Expand(seed_threshold_value=0.55, lower_threshold_value=.3, prediction_key='prediction'),
        ExpandDimensions(image_keys=('og_annotation',), axis=-1),
        MaskOneBasedOnOther(guiding_keys=('og_annotation',), changing_keys=('prediction',),
                            guiding_values=(0,), mask_values=(0,)),
    ])
    return pancreas_cyst


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
        AddSpacing(spacing_handle_key='primary_handle'),
        ExpandDimensions(axis=-1, image_keys=('image',)),
        Focus_on_CT()])
    lacc_model.set_prediction_processors([
        ProcessPrediction(prediction_keys=('prediction',),
                          threshold={"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5, "5": 0.5, "6": 0.5, "7": 0.5, "8": 0.5,
                                     "9": 0.5, "10": 0.5, "11": 0.5, "12": 0.5},
                          connectivity={"1": False, "2": True, "3": True, "4": False, "5": True, "6": False,
                                        "7": True, "8": True, "9": True, "10": True, "11": False, "12": False},
                          extract_main_comp={"1": True, "2": False, "3": False, "4": False, "5": False, "6": False,
                                             "7": False, "8": False, "9": False, "10": False, "11": False, "12": False},
                          thread_count=12, dist=20, max_comp=2),
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


def return_lacc_pb3D_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    # lacc_model = PredictLACC(image_key='image',
    #                          model_path=os.path.join(model_load_path,
    #                                                  'LACC_3D',
    #                                                  'pb3D_model_Trial_6_test.hdf5'),
    #                          model_template=DenseNet3D(input_tensor=None, input_shape=(32, 192, 192, 1),
    #                                                    classes=13,
    #                                                    classifier_activation="softmax",
    #                                                    activation="relu",
    #                                                    normalization="group", nb_blocks=3,
    #                                                    nb_layers=3, dense_decoding=False,
    #                                                    transition_pool=False,
    #                                                    ds_conv=False, atrous_rate=1).get_net())

    lacc_model = PredictLACC(image_key='image',
                             model_path=os.path.join(model_load_path,
                                                     'LACC_3D',
                                                     'BasicUNet3D_Trial_2_test.hdf5'),
                             model_template=BasicUnet3D(input_tensor=None, input_shape=(32, 192, 192, 1),
                                                        classes=13, classifier_activation="softmax",
                                                        activation="leakyrelu",
                                                        normalization="instance", nb_blocks=2,
                                                        nb_layers=5, dropout='standard',
                                                        filters=32, dropout_rate=0.1).get_net())
    paths = [
        os.path.join(shared_drive_path, 'LACC_3D_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'LACC_3D_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'LACC_3D_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'LACC_3D_Auto_Contour', 'Input_3')
    ]

    # TODO add create external contour in dictionnary
    # multiply pred by to remove unwanted weird prediction in background
    # cf focus
    # TODO extract bounding box using that external contour (cf liver disease)
    # BinaryFillholeImageFilter

    lacc_model.set_paths(paths)
    lacc_model.set_image_processors([
        Threshold_Images(image_keys=('image',), lower_bounds=(-1000,), upper_bounds=(1500,), divides=(False,)),
        CreateExternal(image_key='image', output_key='external', threshold_value=-250.0, mask_value=1),
        DeepCopyKey(from_keys=('external',), to_keys=('og_external',)),
        Normalize_Images(keys=('image',), mean_values=(20.0,), std_values=(30.0,)),
        Threshold_Images(image_keys=('image',), lower_bounds=(-3.55,), upper_bounds=(3.55,), divides=(False,)),
        AddByValues(image_keys=('image',), values=(3.55,)),
        DivideByValues(image_keys=('image',), values=(7.10,)),
        AddSpacing(spacing_handle_key='primary_handle'),
        # post_process_interpolators is Nearest with argmax!
        # Resampler(resample_keys=('image', 'external'),
        #           resample_interpolators=('Linear','Nearest'),
        #           desired_output_spacing=[1.17, 1.17, 3.0],
        #           post_process_resample_keys=('prediction',),
        #           post_process_original_spacing_keys=('primary_handle',),
        #           post_process_interpolators=('Nearest',)),
        Resampler(resample_keys=('image', 'external'),
                  resample_interpolators=('Linear', 'Nearest'),
                  desired_output_spacing=[1.17, 1.17, 3.0],
                  post_process_resample_keys=('prediction',),
                  post_process_original_spacing_keys=('primary_handle',),
                  post_process_interpolators=('Linear',)),
        # PadImages(bounding_box_expansion=(0, 0, 0), power_val_z=32, power_val_x=192,
        #           power_val_y=192, min_val=None, image_keys=('image',),
        #           post_process_keys=('image', 'prediction')),
        Box_Images(bounding_box_expansion=(0, 0, 0), image_keys=('image',),
                   annotation_key='external', wanted_vals_for_bbox=(1,),
                   power_val_z=32, power_val_r=192, power_val_c=192,
                   post_process_keys=('prediction',)),
        ExpandDimensions(image_keys=('image',), axis=-1),
        ExpandDimensions(image_keys=('image',), axis=0),
        SqueezeDimensions(post_prediction_keys=('prediction',))
    ])
    lacc_model.set_prediction_processors([
        ExpandDimensions(image_keys=('og_external',), axis=-1),
        MaskOneBasedOnOther(guiding_keys=tuple(['og_external' for i in range(0,13)]), changing_keys=tuple(['prediction' for i in range(0,13)]),
                            guiding_values=tuple([0 for i in range(0,13)]), mask_values=tuple([i for i in range(0,13)])),
        ProcessPrediction(prediction_keys=('prediction',),
                          threshold={"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5, "5": 0.5, "6": 0.5, "7": 0.5, "8": 0.5,
                                     "9": 0.5, "10": 0.5, "11": 0.5, "12": 0.5},
                          connectivity={"1": False, "2": True, "3": True, "4": False, "5": True, "6": False,
                                        "7": True, "8": True, "9": True, "10": True, "11": False, "12": False},
                          extract_main_comp={"1": True, "2": False, "3": False, "4": False, "5": False, "6": False,
                                             "7": False, "8": False, "9": False, "10": False, "11": False, "12": False},
                          thread_count=12, dist=20, max_comp=2),
        CombinePredictions(prediction_keys=('prediction',), combine_ids=((7, 8),), closings=(False,)),
        CreateUpperVagina(prediction_keys=('prediction',), class_id=(5,), sup_margin=(20,)),
        CombinePredictions(prediction_keys=('prediction',), combine_ids=((1, 14, 6),), closings=(True,)),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v5' for roi in
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


def return_ctvn_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    ctvn_model = ModelBuilderFromTemplate(image_key='image',
                                          model_path=os.path.join(model_load_path,
                                                                  'CTVN',
                                                                  'DLv3_model_CTVN_v1_Trial_34.hdf5'),
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
        ExpandDimensions(axis=-1, image_keys=('image',)),
        # RepeatChannel(num_repeats=3, axis=-1, image_keys=('image',)),
        Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
                                 post_process_keys=('image', 'prediction')),
    ])
    ctvn_model.set_prediction_processors([
        ProcessPrediction(prediction_keys=('prediction',),
                          threshold={"1": 0.5, "2": 0.5},
                          connectivity={"1": False, "2": False},
                          extract_main_comp={"1": True, "2": True},
                          thread_count=2, dist=5, max_comp=2),
        CombinePredictions(prediction_keys=('prediction',), combine_ids=((1, 2),), closings=(True,)),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v1' for roi in ["CTVn", "CTV_PAN", "Nodal_CTV"]]
    else:
        roi_names = ["CTVn", "CTV_PAN", "Nodal_CTV"]

    ctvn_model.set_dicom_reader(TemplateDicomReader(roi_names=roi_names))
    return ctvn_model


def return_duodenum_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    duodenum_model = ModelBuilderFromTemplate(image_key='image',
                                              model_path=os.path.join(model_load_path,
                                                                      'Duodenum',
                                                                      'DLv3_model_Duodenum_v0_Trial_38.hdf5'),
                                              model_template=deeplabv3plus(input_shape=(512, 512, 3),
                                                                           backbone="xception",
                                                                           classes=2, final_activation='softmax',
                                                                           windowopt_flag=False,
                                                                           normalization='batch', activation='relu',
                                                                           weights=None).Deeplabv3())
    paths = [
        os.path.join(shared_drive_path, 'Duodenum_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'Duodenum_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'Duodenum_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'Duodenum_Auto_Contour', 'Input_3')
    ]
    duodenum_model.set_paths(paths)
    duodenum_model.set_image_processors([
        Normalize_Images(keys=('image',), mean_values=(25.0,), std_values=(129.0,)),
        Threshold_Images(image_keys=('image',), lower_bounds=(-3.55,), upper_bounds=(3.55,), divides=(False,)),
        AddByValues(image_keys=('image',), values=(3.55,)),
        DivideByValues(image_keys=('image', 'image'), values=(7.10, 1 / 255)),
        ExpandDimensions(axis=-1, image_keys=('image',)),
        RepeatChannel(num_repeats=3, axis=-1, image_keys=('image',)),
        Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
                                 post_process_keys=('image', 'prediction')),
    ])
    duodenum_model.set_prediction_processors([
        ProcessPrediction(prediction_keys=('prediction',),
                          threshold={"1": 0.5},
                          connectivity={"1": False},
                          extract_main_comp={"1": False},
                          thread_count=1, dist=50, max_comp=2),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v0' for roi in ["Duodenum"]]
    else:
        roi_names = ['Duodenum']

    duodenum_model.set_dicom_reader(TemplateDicomReader(roi_names=roi_names))
    return duodenum_model


class BaseModelBuilder(object):
    def __init__(self, image_key='image', model_path=None, Bilinear_model=None, loss=None, loss_weights=None):
        self.image_key = image_key
        self.model_path = model_path
        self.Bilinear_model = Bilinear_model
        self.loss = loss
        self.loss_weights = loss_weights
        self.paths = []
        self.image_processors = []
        self.prediction_processors = []
        self.dicom_reader = None

    def set_paths(self, paths_list):
        self.paths = paths_list

    def set_image_processors(self, image_processors_list):
        self.image_processors = image_processors_list

    def set_prediction_processors(self, prediction_processors_list):
        self.prediction_processors = prediction_processors_list

    def set_dicom_reader(self, dicom_reader):
        self.dicom_reader = dicom_reader

    def build_model(self, graph=None, session=None, model_name='modelname'):
        if self.loss is not None and self.loss_weights is not None:
            self.loss = self.loss(self.loss_weights)
        print("Loading model from: {}".format(self.model_path))
        self.model = tf.keras.models.load_model(self.model_path,
                                                custom_objects={'BilinearUpsampling': self.Bilinear_model,
                                                                'dice_coef_3D': dice_coef_3D,
                                                                'loss': self.loss},
                                                compile=False)
        self.model.trainable = False
        # self.model.load_weights(self.model_path, by_name=True, skip_mismatch=False)
        # avoid forbidden character from tf1.14 model (for ex: DeepLabV3+)
        # also allocate a scope per model name
        self.model._name = model_name

    def load_images(self, input_features):
        input_features = self.dicom_reader.load_images(input_features=input_features)
        return input_features

    def return_series_instance_dictionary(self):
        return self.dicom_reader.reader.series_instances_dictionary[0]

    def return_status(self):
        return self.dicom_reader.return_status()

    def pre_process(self, input_features):
        for processor in self.image_processors:
            print('Performing pre process {}'.format(processor))
            input_features = processor.pre_process(input_features=input_features)
        return input_features

    def post_process(self, input_features):
        for processor in self.image_processors[::-1]:  # In reverse order now
            print('Performing post process {}'.format(processor))
            input_features = processor.post_process(input_features=input_features)
        return input_features

    def prediction_process(self, input_features):
        for processor in self.prediction_processors:  # In reverse order now
            print('Performing prediction process {}'.format(processor))
            input_features = processor.pre_process(input_features=input_features)
        return input_features

    def predict(self, input_features):
        input_features['prediction'] = self.model.predict(input_features[self.image_key])
        return input_features

    def write_predictions(self, input_features):
        self.dicom_reader.write_predictions(input_features=input_features)


class BaseModelBuilderGraph(BaseModelBuilder):
    # keep for legacy
    # see test_graph_liver for how to use graph/session

    def build_model(self, graph=None, session=None, model_name='modelname'):
        with graph.as_default():
            with session.as_default():
                if self.loss is not None and self.loss_weights is not None:
                    self.loss = self.loss(self.loss_weights)
                print("Loading model from: {}".format(self.model_path))
                if tf.__version__ == '1.14.0':
                    print('loading VGG Pretrained')
                    self.model = tf.keras.models.load_model(self.model_path,
                                                            custom_objects={'BilinearUpsampling': self.Bilinear_model,
                                                                            'dice_coef_3D': dice_coef_3D,
                                                                            'loss': self.loss})
                else:
                    self.model = tf.keras.models.load_model(self.model_path,
                                                            custom_objects={'BilinearUpsampling': self.Bilinear_model,
                                                                            'dice_coef_3D': dice_coef_3D,
                                                                            'loss': self.loss},
                                                            compile=False)
                if os.path.isdir(self.model_path):
                    session.run(tf.compat.v1.global_variables_initializer())


class ModelBuilderFromTemplate(BaseModelBuilder):
    def __init__(self, image_key='image', model_path=None, model_template=None):
        super().__init__(image_key, model_path)
        self.image_key = image_key
        self.model_path = model_path
        self.paths = []
        self.image_processors = []
        self.prediction_processors = []
        self.dicom_reader = None
        self.model_template = model_template

    def build_model(self, graph=None, session=None, model_name='modelname'):
        if self.model_template:
            self.model = self.model_template
            if os.path.isfile(self.model_path):
                print("Loading weights from: {}".format(self.model_path))
                self.model.load_weights(self.model_path, by_name=True, skip_mismatch=False)
                # avoid forbidden character from tf1.14 model
                # also allocate a scope per model name
                self.model._name = model_name
            else:
                raise ValueError("Model path {} is not a file or cannot be found!".format(self.model_path))


class PredictLobes(BaseModelBuilder):
    def predict(self, input_features):
        pred = self.model.predict(input_features['combined'])
        input_features['prediction'] = np.squeeze(pred)
        return input_features


class PredictDiseaseAblation(BaseModelBuilder):
    def predict(self, input_features):
        x = input_features['combined']
        step = 64
        shift = 32
        gap = 8
        if x[..., 0].shape[1] > step:
            pred = np.zeros(x[..., 0].shape[1:] + (2,))
            start = 0
            while start < x[0].shape[1]:
                image_cube, mask_cube = x[..., 0][:, start:start + step, ...], x[..., -1][:, start:start + step, ...]
                image_cube, mask_cube = image_cube[..., None], mask_cube[..., None]
                difference = image_cube.shape[1] % 32
                if difference != 0:
                    image_cube = np.pad(image_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
                    mask_cube = np.pad(mask_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
                pred_cube = self.model.predict([image_cube, mask_cube])
                pred_cube = pred_cube[:, difference:, ...]
                start_gap = gap
                stop_gap = gap
                if start == 0:
                    start_gap = 0
                elif start + step >= x[0].shape[1]:
                    stop_gap = 0
                if stop_gap != 0:
                    pred_cube = pred_cube[:, start_gap:-stop_gap, ...]
                else:
                    pred_cube = pred_cube[:, start_gap:, ...]
                pred[start + start_gap:start + start_gap + pred_cube.shape[1], ...] = pred_cube[0, ...]
                start += shift
        else:
            image_cube, mask_cube = x[..., 0][..., None], x[..., -1][..., None]
            difference = image_cube.shape[1] % 32
            if difference != 0:
                image_cube = np.pad(image_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
                mask_cube = np.pad(mask_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
            pred_cube = self.model.predict([image_cube, mask_cube])
            pred = pred_cube[:, difference:, ...]
        input_features['prediction'] = np.squeeze(pred)
        return input_features


class TemplateDicomReader(object):
    def __init__(self, roi_names, associations=None):
        self.status = True
        self.associations = associations
        self.roi_names = roi_names
        self.reader = DicomReaderWriter(associations=self.associations)

    def load_images(self, input_features):
        input_path = input_features['input_path']
        self.reader.__reset__()
        self.reader.walk_through_folders(input_path)
        self.reader.get_images()
        input_features['image'] = self.reader.ArrayDicom
        input_features['primary_handle'] = self.reader.dicom_handle
        return input_features

    def return_status(self):
        return self.status

    def write_predictions(self, input_features):
        self.reader.template = 1
        true_outpath = input_features['out_path']
        annotations = input_features['prediction']
        contour_values = np.max(annotations, axis=0)
        while len(contour_values.shape) > 1:
            contour_values = np.max(contour_values, axis=0)
        contour_values[0] = 1
        annotations = annotations[..., contour_values == 1]
        contour_values = contour_values[1:]
        ROI_Names = list(np.asarray(self.roi_names)[contour_values == 1])
        if ROI_Names:
            self.reader.prediction_array_to_RT(prediction_array=annotations,
                                               output_dir=true_outpath,
                                               ROI_Names=ROI_Names)
        else:
            no_prediction = os.path.join(true_outpath, 'Status_No Prediction created.txt')
            fid = open(no_prediction, 'w+')
            fid.close()
            fid = open(os.path.join(true_outpath, 'Failed.txt'), 'w+')
            fid.close()


class EnsureLiverPresent(TemplateDicomReader):
    def __init__(self, roi_names=None, associations=None, wanted_roi='Liver', liver_folder=None):
        super(EnsureLiverPresent, self).__init__(associations=associations, roi_names=roi_names)
        self.wanted_roi = wanted_roi
        self.liver_folder = liver_folder
        self.reader = DicomReaderWriter(associations=self.associations, Contour_Names=[self.wanted_roi])

    def check_ROIs_In_Checker(self):
        self.roi_name = None
        for roi in self.reader.rois_in_case:
            if roi.lower() == self.wanted_roi.lower():
                self.roi_name = roi
                return None
        for roi in self.reader.rois_in_case:
            if roi in self.associations:
                if self.associations[roi] == self.wanted_roi.lower():
                    self.roi_name = roi
                    break

    def load_images(self, input_features):
        input_path = input_features['input_path']
        self.reader.__reset__()
        self.reader.walk_through_folders(input_path)
        self.check_ROIs_In_Checker()
        go = False
        if self.roi_name is None and go:
            liver_input_path = os.path.join(self.liver_folder, self.reader.ds.PatientID,
                                            self.reader.ds.SeriesInstanceUID)
            liver_out_path = liver_input_path.replace('Input_3', 'Output')
            if os.path.exists(liver_out_path):
                files = [i for i in os.listdir(liver_out_path) if i.find('.dcm') != -1]
                for file in files:
                    self.reader.lstRSFile = os.path.join(liver_out_path, file)
                    self.reader.get_rois_from_RT()
                    self.check_ROIs_In_Checker()
                    if self.roi_name:
                        print('Previous liver contour found at ' + liver_out_path + '\nCopying over')
                        shutil.copy(os.path.join(liver_out_path, file), os.path.join(input_path, file))
                        break
        if self.roi_name is None:
            self.status = False
            print('No liver contour found')
        if self.roi_name:
            self.reader.get_images_and_mask()
            input_features['image'] = self.reader.ArrayDicom
            input_features['primary_handle'] = self.reader.dicom_handle
            input_features['annotation'] = self.reader.mask
        return input_features


class PredictLACC(ModelBuilderFromTemplate):

    def predict(self, input_features):
        # This function follows on monai.inferers.SlidingWindowInferer implementations
        x = input_features['image']
        nb_label = 13
        required_size = (32, 192, 192)
        sw_batch_size = 8
        batch_size = 1
        image_size = x[0, ..., 0].shape
        sigma_scale = 0.125
        sw_overlap = 0.50
        scan_interval = _get_scan_interval(image_size, required_size, 3, sw_overlap)

        # Store all slices in list
        slices = dense_patch_slices(image_size, required_size, scan_interval)
        num_win = len(slices)  # number of windows per image
        total_slices = num_win * batch_size  # total number of windows

        # Create window-level importance map (can be changed to remove border effect for example)
        # importance_map = np.ones(required_size + (nb_label,))
        GaussianSource = sitk.GaussianSource(size=required_size[::-1],
                                             mean=tuple([x // 2 for x in required_size[::-1]]),
                                             sigma=tuple([sigma_scale * x for x in required_size[::-1]]), scale=1.0,
                                             spacing=(1.0, 1.0, 1.0), normalized=False)
        importance_map = sitk.GetArrayFromImage(GaussianSource)
        importance_map = np.repeat(importance_map[..., None], repeats=nb_label, axis=-1)

        # Perform predictions
        # output_image, count_map = np.array([]), np.array([])
        _initialized = False
        for slice_g in range(0, total_slices, sw_batch_size):
            slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
            unravel_slice = [
                [slice(int(idx / num_win), int(idx / num_win) + 1)] + list(slices[idx % num_win]) + [slice(None)]
                for idx in slice_range
            ]
            window_data = np.concatenate([x[tuple(win_slice)] for win_slice in unravel_slice], axis=0)
            seg_prob = self.model.predict(window_data)

            if not _initialized:  # init. buffer at the first iteration
                output_shape = [batch_size] + list(image_size) + [nb_label]
                # allocate memory to store the full output and the count for overlapping parts
                output_image = np.zeros(output_shape, dtype=np.float32)
                count_map = np.zeros(output_shape, dtype=np.float32)
                _initialized = True

            # store the result in the proper location of the full output. Apply weights from importance map.
            for idx, original_idx in zip(slice_range, unravel_slice):
                output_image[tuple(original_idx)] += importance_map * seg_prob[idx - slice_g]
                count_map[tuple(original_idx)] += importance_map

        # account for any overlapping sections
        # input_features['prediction'] = to_categorical(argmax_keepdims(np.squeeze(output_image / count_map), axis=-1),
        #                                               num_classes=nb_label)
        input_features['prediction'] = np.squeeze(output_image / count_map)
        return input_features

    def predict_np(self, input_features):
        # this function needs a input image with compatible number of required_size
        # otherwise predict will not be performed on the entire FOV resulting on NaN recovered patches
        x = input_features['image']
        nb_label = 13
        required_size = (32, 192, 192)
        shift = (16, 96, 96)
        batch_size = 4

        x_patches = patch_extract_3D(input=x[0, ..., 0], patch_shape=required_size, xstep=shift[0], ystep=shift[1],
                                     zstep=shift[2])
        pred_patches = np.zeros(x_patches.shape + (nb_label,))

        for index in np.arange(0, x_patches.shape[0], batch_size):
            pred_patches[index:index + batch_size, ...] = self.model.predict(
                x_patches[index:index + batch_size, ...][..., None])

        pred = np.zeros(x[0, ..., 0].shape + (nb_label,))

        for label in range(1, pred_patches.shape[-1]):
            print(label)
            pred[..., label] = recover_patches_3D(out_shape=x[0, ..., 0].shape, patches=pred_patches[..., label],
                                                  xstep=shift[0], ystep=shift[1], zstep=shift[2])

        input_features['prediction'] = pred
        return input_features

    def predict_std(self, input_features):

        # extracting patches using numpy broadcasting
        # for 1d
        # def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
        #     nrows = ((a.size - L) // S) + 1
        #     return a[S * np.arange(nrows)[:, None] + np.arange(L)]

        # for 3D
        # nslices = ((img.shape[0] - 32) // 16) + 1
        # nrows = ((img.shape[1] - 192) // 96) + 1
        # ncols = ((img.shape[2] - 192) // 96) + 1
        # patches = img[16 * np.arange(nslices)[:, None] + np.arange(32), ...][:, :, 96 * np.arange(nrows)[:, None] + np.arange(192), ...][:, :, :,:, 96 * np.arange(ncols)[:, None] + np.arange(192)]

        x = input_features['image']

        nb_label = 13
        required_size = (32, 192, 192)
        step = (32, 192, 192)
        shift = (16, 96, 96)
        start = [0, 0, 0]

        if x.shape[0] == 1:
            x_shape = x[0].shape
        else:
            x_shape = x.shape

        pred_count = np.zeros(x_shape)
        pred = np.zeros(x[0, ..., 0].shape + (nb_label,))
        while start[0] < x_shape[0]:
            start[1] = 0
            while start[1] < x_shape[1]:
                start[2] = 0
                while start[2] < x_shape[2]:
                    print("{}".format(start))

                    image_cube = x[:, start[0]:start[0] + step[0],
                                 start[1]:start[1] + step[1],
                                 start[2]:start[2] + step[2], ...]

                    remain_z, remain_r, remain_c = required_size[0] - image_cube.shape[1], \
                                                   required_size[1] - image_cube.shape[2], \
                                                   required_size[2] - image_cube.shape[3]

                    image_cube = np.pad(image_cube,
                                        [[0, 0], [floor(remain_z / 2), ceil(remain_z / 2)],
                                         [floor(remain_r / 2), ceil(remain_r / 2)],
                                         [floor(remain_c / 2), ceil(remain_c / 2)], [0, 0]],
                                        mode='constant', constant_values=np.min(image_cube))

                    pred_cube = self.model.predict(image_cube)
                    pred_cube = pred_cube[:, floor(remain_z / 2):step[0] - ceil(remain_z / 2),
                                floor(remain_r / 2):step[1] - ceil(remain_r / 2),
                                floor(remain_c / 2):step[2] - ceil(remain_c / 2), ...]

                    pred[start[0]:start[0] + step[0], start[1]:start[1] + step[1], start[2]:start[2] + step[2],
                    ...] += pred_cube[0, ...]
                    pred_count[start[0]:start[0] + step[0], start[1]:start[1] + step[1], start[2]:start[2] + step[2],
                    ...] += 1

                    start[2] += shift[2]
                start[1] += shift[1]
            start[0] += shift[0]

        pred /= np.repeat(pred_count, repeats=nb_label, axis=-1)

        input_features['prediction'] = pred
        return input_features


def argmax_keepdims(x, axis):
    """
    Returns the indices of the maximum values along an axis.

    The axis which is reduced is left in the result as dimension with size one.
    The result will broadcast correctly against the input array.

    Original numpy.argmax() implementation does not currently support the keepdims parameter.
    See https://github.com/numpy/numpy/issues/8710 for further information.
    """
    output_shape = list(x.shape)
    output_shape[axis] = 1
    return np.argmax(x, axis=axis).reshape(output_shape)


def gaussian_blur(img, kernel_size=11, sigma=5):
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1], padding='SAME', data_format='NHWC')


class PredictCyst(ModelBuilderFromTemplate):

    def predict(self, input_features):
        x = input_features['combined']
        required_size = (32, 128, 128)
        step = (32, 128, 128)
        shift = (24, 32, 32)
        start = [0, 0, 0]

        if x[0].shape != (32, 128, 128, 2):
            pred_count = np.zeros(x[..., 0].shape[1:])
            pred = np.zeros(x[..., 0].shape[1:] + (2,))
            while start[0] < x[0].shape[0]:
                start[1] = 0
                while start[1] < x[0].shape[1]:
                    start[2] = 0
                    while start[2] < x[0].shape[2]:
                        image_cube, mask_cube = x[..., 0][:, start[0]:start[0] + step[0], start[1]:start[1] + step[1],
                                                start[2]:start[2] + step[2], ...], x[..., -1][:,
                                                                                   start[0]:start[0] + step[0],
                                                                                   start[1]:start[1] + step[1],
                                                                                   start[2]:start[2] + step[2], ...]
                        image_cube, mask_cube = image_cube[..., None], mask_cube[..., None]

                        remain_z, remain_r, remain_c = required_size[0] - image_cube.shape[1], required_size[1] - \
                                                       image_cube.shape[2], required_size[2] - image_cube.shape[3]

                        image_cube = np.pad(image_cube,
                                            [[0, 0], [floor(remain_z / 2), ceil(remain_z / 2)],
                                             [floor(remain_r / 2), ceil(remain_r / 2)],
                                             [floor(remain_c / 2), ceil(remain_c / 2)], [0, 0]],
                                            mode='reflect')
                        mask_cube = np.pad(mask_cube, [[0, 0], [floor(remain_z / 2), ceil(remain_z / 2)],
                                                       [floor(remain_r / 2), ceil(remain_r / 2)],
                                                       [floor(remain_c / 2), ceil(remain_c / 2)],
                                                       [0, 0]], mode='constant', constant_values=0)

                        pred_cube = self.model.predict([image_cube, mask_cube])
                        pred_cube = pred_cube[:, floor(remain_z / 2):step[0] - ceil(remain_z / 2),
                                    floor(remain_r / 2):step[1] - ceil(remain_r / 2),
                                    floor(remain_c / 2):step[2] - ceil(remain_c / 2), ...]

                        pred[start[0]:start[0] + step[0], start[1]:start[1] + step[1], start[2]:start[2] + step[2],
                        ...] += pred_cube[0, ...]
                        pred_count[start[0]:start[0] + step[0], start[1]:start[1] + step[1],
                        start[2]:start[2] + step[2], ...] += 1
                        start[2] += shift[2]
                    start[1] += shift[1]
                start[0] += shift[0]

            pred /= np.repeat(pred_count[..., None], repeats=2, axis=-1)
        else:
            image_cube, mask_cube = x[..., 0][..., None], x[..., -1][..., None]
            difference = image_cube.shape[1] % 32
            if difference != 0:
                image_cube = np.pad(image_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
                mask_cube = np.pad(mask_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
            pred_cube = self.model.predict([image_cube, mask_cube])
            pred = pred_cube[:, difference:, ...]
        input_features['prediction'] = np.squeeze(pred)
        return input_features


def ensure_tuple_size(tup, dim, pad_val=(0,)):
    """
    Returns a copy of `tup` with `dim` values by either shortened or padded with `pad_val` as necessary.
    """
    tup = tup + (pad_val,) * dim
    return tuple(tup[:dim])


def get_valid_patch_size(image_size, patch_size):
    """
    Given an image of dimensions `image_size`, return a patch size tuple taking the dimension from `patch_size` if this is
    not 0/None. Otherwise, or if `patch_size` is shorter than `image_size`, the dimension from `image_size` is taken. This ensures
    the returned patch size is within the bounds of `image_size`. If `patch_size` is a single number this is interpreted as a
    patch of the same dimensionality of `image_size` with that size in each dimension.
    """
    ndim = len(image_size)
    patch_size_ = ensure_tuple_size(patch_size, ndim)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size_))


def first(iterable, default=None):
    """
    Returns the first item in the given iterable or `default` if empty, meaningful mostly with 'for' expressions.
    """
    for i in iterable:
        return i
    return default


def _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap):
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def dense_patch_slices(image_size, patch_size, scan_interval):
    """
    Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image.

    Args:
        image_size: dimensions of image to iterate over
        patch_size: size of patches to generate slices
        scan_interval: dense patch sampling interval

    Returns:
        a list of slice objects defining each patch

    """
    num_spatial_dims = len(image_size)
    patch_size = get_valid_patch_size(image_size, patch_size)
    scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)

    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i])
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)

    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + patch_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)
    out = np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T
    return [tuple(slice(s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]


def patch_extract_3D(input, patch_shape, xstep=1, ystep=1, zstep=1):
    patches_3D = np.lib.stride_tricks.as_strided(input, (
        (input.shape[0] - patch_shape[0] + 1) // xstep, (input.shape[1] - patch_shape[1] + 1) // ystep,
        (input.shape[2] - patch_shape[2] + 1) // zstep, patch_shape[0], patch_shape[1], patch_shape[2]),
                                                 (input.strides[0] * xstep, input.strides[1] * ystep,
                                                  input.strides[2] * zstep, input.strides[0], input.strides[1],
                                                  input.strides[2]))
    patches_3D = patches_3D.reshape(patches_3D.shape[0] * patches_3D.shape[1] * patches_3D.shape[2],
                                    patch_shape[0], patch_shape[1], patch_shape[2])
    return patches_3D


def recover_patches_3D(out_shape, patches, xstep=12, ystep=12, zstep=12):
    out = np.zeros(out_shape, patches.dtype)
    denom = np.zeros(out_shape, patches.dtype)
    patch_shape = patches.shape[-3:]
    patches_6D = np.lib.stride_tricks.as_strided(out, (
        (out.shape[0] - patch_shape[0] + 1) // xstep, (out.shape[1] - patch_shape[1] + 1) // ystep,
        (out.shape[2] - patch_shape[2] + 1) // zstep, patch_shape[0], patch_shape[1], patch_shape[2]),
                                                 (out.strides[0] * xstep, out.strides[1] * ystep,
                                                  out.strides[2] * zstep, out.strides[0], out.strides[1],
                                                  out.strides[2]))
    denom_6D = np.lib.stride_tricks.as_strided(denom, (
        (denom.shape[0] - patch_shape[0] + 1) // xstep, (denom.shape[1] - patch_shape[1] + 1) // ystep,
        (denom.shape[2] - patch_shape[2] + 1) // zstep, patch_shape[0], patch_shape[1], patch_shape[2]),
                                               (denom.strides[0] * xstep, denom.strides[1] * ystep,
                                                denom.strides[2] * zstep, denom.strides[0], denom.strides[1],
                                                denom.strides[2]))
    np.add.at(patches_6D, tuple(x.ravel() for x in np.indices(patches_6D.shape)), patches.ravel())
    np.add.at(denom_6D, tuple(x.ravel() for x in np.indices(patches_6D.shape)), 1)
    return out / denom


def main():
    pass


if __name__ == '__main__':
    main()
