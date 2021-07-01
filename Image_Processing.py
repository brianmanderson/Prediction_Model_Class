import shutil, os, sys

sys.path.insert(0, os.path.abspath('.'))
from functools import partial
from Image_Processors_Module.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Image_Processors_Module.src.Processors.MakeTFRecordProcessors import *
from Dicom_RT_and_Images_to_Mask.src.DicomRTTool import DicomReaderWriter
import tensorflow as tf
from Bilinear_Dsc import BilinearUpsampling

from Image_Processors_Utils.Image_Processor_Utils import ProcessPrediction, Postprocess_Pancreas, Normalize_Images, \
    Threshold_Images, DilateBinary, Focus_on_CT, CombinePredictions, CreateUpperVagina

# this submodule is private (ask @guatavita Github)
from networks.DeepLabV3plus import *


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
    # Add connectivity to MaskOneBasedOnOther

    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    pancreas_cyst = PredictCyst(image_key='combined', model_path=os.path.join(model_load_path, 'Cyst',
                                                                              'HybridDLv3_Cyst_model_debug_119.hdf5'),
                                model_template=deeplabv3plus(nb_blocks=3, nb_layers=2,
                                                             backbone='xception', input_shape=(32, 128, 128, 1),
                                                             classes=2, final_activation='softmax',
                                                             activation='relu', normalization='group',
                                                             windowopt_flag=False, nb_output=3,
                                                             add_squeeze=True, add_mask=True, dense_decoding=False,
                                                             transition_pool=False, ds_conv=True,
                                                             weights=os.path.join(model_load_path, 'Cyst',
                                                                                  'HybridDLv3_Cyst_model_debug_119.hdf5'),
                                                             ).HybridDeeplabv3())
    paths = [
        os.path.join(morfeus_path, 'Bastien', 'Auto_seg', 'RayStation', 'Cyst', 'Input_3'),
    ]
    pancreas_cyst.set_paths(paths)
    pancreas_cyst.set_image_processors([
        DeepCopyKey(from_keys=('annotation',), to_keys=('og_annotation',)),
        Normalize_Images(keys=('image',), mean_values=(21.0,), std_values=(25.0,)),
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
                                                      associations={'Pancreas_DLv3_v0': 'Pancreas_DLv3_v0'}))
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
                          thread_count=12, dist=50, max_comp=2),
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


def return_ctvn_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    ctvn_model = ModelBuilderFromTemplate(image_key='image',
                                          model_path=os.path.join(model_load_path,
                                                                  'CTVN',
                                                                  'DLv3_model_CTVN_v1.hdf5'),
                                          model_template=deeplabv3plus(input_shape=(512, 512, 3),
                                                                       backbone="xception",
                                                                       classes=3, final_activation='softmax',
                                                                       windowopt_flag=False,
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
        Normalize_Images(keys=('image',), mean_values=(-17.0,), std_values=(63.0,)),
        Threshold_Images(image_keys=('image',), lower_bounds=(-3.55,), upper_bounds=(3.55,), divides=(False,)),
        AddByValues(image_keys=('image',), values=(3.55,)),
        DivideByValues(image_keys=('image', 'image'), values=(7.10, 1 / 255)),
        ExpandDimensions(axis=-1, image_keys=('image',)),
        RepeatChannel(num_repeats=3, axis=-1, image_keys=('image',)),
        Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
                                 post_process_keys=('image', 'prediction')),
    ])
    ctvn_model.set_prediction_processors([
        ProcessPrediction(prediction_keys=('prediction',),
                          threshold={"1": 0.5, "2":0.5},
                          connectivity={"1": False, "2": False},
                          extract_main_comp={"1": False, "2": False},
                          thread_count=1, dist=50, max_comp=2),
        CombinePredictions(prediction_keys=('prediction',), combine_ids=((1, 2),), closings=(True,)),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v0' for roi in ["CTVn", "CTV_PAN", "Nodal_CTV"]]
    else:
        roi_names = ["CTVn", "CTV_PAN", "Nodal_CTV"]

    ctvn_model.set_dicom_reader(TemplateDicomReader(roi_names=roi_names))
    return ctvn_model


def return_duodenum_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    duodenum_model = ModelBuilderFromTemplate(image_key='image',
                                              model_path=os.path.join(model_load_path,
                                                                      'Duodenum',
                                                                      'DLv3_model_Duodenum_v0.hdf5'),
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


class PredictCyst(ModelBuilderFromTemplate):

    def predict(self, input_features):
        x = input_features['combined']

        required_size = (32, 128, 128)
        step = (32, 128, 128)
        shift = (32, 32, 32)
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


def main():
    pass


if __name__ == '__main__':
    main()
