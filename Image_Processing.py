import shutil, os, sys
from math import floor, ceil

sys.path.insert(0, os.path.abspath('.'))
from functools import partial
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *
import tensorflow as tf

import SimpleITK as sitk

# this submodule is private (ask @guatavita Github)
try:
    from networks.DeepLabV3plus import *
    from networks.UNet3D import *
except:
    print('Cannot load from networks submodule, ask @guatavita Github if you want this functionality')


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


def find_base_dir():
    base_path = '.'
    for _ in range(20):
        if 'Morfeus' in os.listdir(base_path):
            break
        else:
            base_path = os.path.join(base_path, '..')
    return base_path


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
        Processors.Normalize_to_annotation(image_key='image', annotation_key='annotation', annotation_value_list=(1,)),
        Processors.Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image', 'annotation')),
        Processors.CastData(image_keys=('image', 'annotation'), dtypes=('float32', 'int')),
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.DeepCopyKey(from_keys=('annotation',), to_keys=('og_annotation',)),
        Processors.Resampler(resample_keys=('image', 'annotation'), resample_interpolators=('Linear', 'Nearest'),
                  desired_output_spacing=[None, None, 5.0], post_process_resample_keys=('prediction',),
                  post_process_original_spacing_keys=('primary_handle',), post_process_interpolators=('Linear',)),
        Box_Images(bounding_box_expansion=(10, 10, 10), image_keys=('image',), annotation_key='annotation',
                   wanted_vals_for_bbox=(1,), power_val_z=64, power_val_r=320, power_val_c=384,
                   post_process_keys=('image', 'annotation', 'prediction'), pad_value=0),
        Processors.ExpandDimensions(image_keys=('image', 'annotation'), axis=0),
        Processors.ExpandDimensions(image_keys=('image', 'annotation', 'og_annotation'), axis=-1),
        Processors.Threshold_Images(image_keys=('image',), lower_bounds=(-5,), upper_bounds=(5,), divides=(False,)),
        Processors.DivideByValues(image_keys=('image',), values=(10,)),
        Processors.MaskOneBasedOnOther(guiding_keys=('annotation',), changing_keys=('image',), guiding_values=(0,),
                            mask_values=(0,)),
        Processors.CreateTupleFromKeys(image_keys=('image', 'annotation'), output_key='combined'),
        Processors.SqueezeDimensions(post_prediction_keys=('image', 'annotation', 'prediction'))
    ])
    liver_lobe_model.set_prediction_processors([
        Processors.MaskOneBasedOnOther(guiding_keys=('og_annotation',), changing_keys=('prediction',), guiding_values=(0,),
                            mask_values=(0,)),
        Processors.SqueezeDimensions(image_keys=('og_annotation',)),
        Processors.Threshold_and_Expand_New(seed_threshold_value=[.9, .9, .9, .9, .9],
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
        Processors.DeepCopyKey(from_keys=('annotation',), to_keys=('og_annotation',)),
        Processors.Normalize_to_annotation(image_key='image', annotation_key='annotation',
                                annotation_value_list=(1,), mirror_max=True),
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.Resampler(resample_keys=('image', 'annotation'),
                  resample_interpolators=('Linear', 'Nearest'),
                  desired_output_spacing=[None, None, 1.0],
                  post_process_resample_keys=('prediction',),
                  post_process_original_spacing_keys=('primary_handle',),
                  post_process_interpolators=('Linear',)),
        Box_Images(bounding_box_expansion=(5, 20, 20), image_keys=('image',),
                   annotation_key='annotation', wanted_vals_for_bbox=(1,),
                   power_val_z=2 ** 4, power_val_r=2 ** 5, power_val_c=2 ** 5),
        Processors.Threshold_Images(lower_bounds=(-10,), upper_bounds=(10,), divides=(True,), image_keys=('image',)),
        Processors.ExpandDimensions(image_keys=('image', 'annotation'), axis=0),
        Processors.ExpandDimensions(image_keys=('image', 'annotation'), axis=-1),
        Processors.MaskOneBasedOnOther(guiding_keys=('annotation',),
                            changing_keys=('image',),
                            guiding_values=(0,),
                            mask_values=(0,)),
        Processors.CombineKeys(image_keys=('image', 'annotation'), output_key='combined'),
        Processors.SqueezeDimensions(post_prediction_keys=('image', 'annotation', 'prediction'))
    ])
    liver_disease.set_dicom_reader(EnsureLiverPresent(wanted_roi='Liver_BMA_Program_4',
                                                      roi_names=['Liver_Disease_Ablation_BMA_Program_0'],
                                                      liver_folder=os.path.join(raystation_clinical_path,
                                                                                'Liver_Auto_Contour', 'Input_3'),
                                                      associations={'Liver_MorfeusLab_v0': 'Liver_BMA_Program_4',
                                                                    'Liver_BMA_Program_4': 'Liver_BMA_Program_4',
                                                                    'Liver': 'Liver_BMA_Program_4'}))
    liver_disease.set_prediction_processors([
        Processors.Threshold_and_Expand(seed_threshold_value=0.55, lower_threshold_value=.3, prediction_key='prediction'),
        Processors.Fill_Binary_Holes(prediction_key='prediction', dicom_handle_key='primary_handle'),
        Processors.ExpandDimensions(image_keys=('og_annotation',), axis=-1),
        Processors.MaskOneBasedOnOther(guiding_keys=('og_annotation',), changing_keys=('prediction',),
                            guiding_values=(0,), mask_values=(0,)),
        Processors.MinimumVolumeandAreaPrediction(min_volume=0.25, prediction_key='prediction',
                                       dicom_handle_key='primary_handle')
    ])
    return liver_disease


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
        Processors.DeepCopyKey(from_keys=('annotation',), to_keys=('og_annotation',)),
        Processors.Normalize_Images(keys=('image',), mean_values=(21.0,), std_values=(24.0,)),
        Processors.Threshold_Images(image_keys=('image',), lower_bounds=(-3.55,), upper_bounds=(3.55,), divides=(False,)),
        Processors.AddByValues(image_keys=('image',), values=(3.55,)),
        Processors.DivideByValues(image_keys=('image', 'image'), values=(7.10, 1 / 255)),
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.Resampler(resample_keys=('image', 'annotation'),
                  resample_interpolators=('Linear', 'Nearest'),
                  desired_output_spacing=[1.0, 1.0, 3.0],
                  post_process_resample_keys=('prediction',),
                  post_process_original_spacing_keys=('primary_handle',),
                  post_process_interpolators=('Linear',)),
        DilateBinary(image_keys=('annotation',), radius=(5,)),
        Box_Images(bounding_box_expansion=(5, 20, 20), image_keys=('image',),
                   annotation_key='annotation', wanted_vals_for_bbox=(1,),
                   power_val_z=2 ** 4, power_val_r=2 ** 5, power_val_c=2 ** 5),
        Processors.ExpandDimensions(image_keys=('image', 'annotation'), axis=0),
        Processors.ExpandDimensions(image_keys=('image', 'annotation'), axis=-1),
        Processors.CombineKeys(image_keys=('image', 'annotation'), output_key='combined'),
        Processors.SqueezeDimensions(post_prediction_keys=('image', 'annotation', 'prediction'))
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
        Processors.Threshold_and_Expand(seed_threshold_value=0.55, lower_threshold_value=.3, prediction_key='prediction'),
        Processors.ExpandDimensions(image_keys=('og_annotation',), axis=-1),
        Processors.MaskOneBasedOnOther(guiding_keys=('og_annotation',), changing_keys=('prediction',),
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
        Threshold_Images(image_keys=('image',), lower_bounds=(-1000,), upper_bounds=(1500,), divides=(False,)),
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


def return_duodenum_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    required_size = (48, 192, 192)
    duodenum_model = PredictWindowSliding(image_key='image',
                                          model_path=os.path.join(model_load_path,
                                                                  'Duodenum',
                                                                  'BasicUNet3D_Duodenum_v3_Trial_45.hdf5'),
                                          model_template=BasicUnet3D(input_tensor=None,
                                                                     input_shape=required_size + (1,),
                                                                     classes=2, classifier_activation="softmax",
                                                                     activation="leakyrelu",
                                                                     normalization="group", nb_blocks=2,
                                                                     nb_layers=5, dropout='standard',
                                                                     filters=32, dropout_rate=0.1,
                                                                     skip_type='concat',
                                                                     bottleneck='standard').get_net(),
                                          nb_label=2, required_size=required_size
                                          )
    paths = [
        os.path.join(shared_drive_path, 'Duodenum_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'Duodenum_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'Duodenum_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'Duodenum_Auto_Contour', 'Input_3')
    ]
    duodenum_model.set_paths(paths)
    duodenum_model.set_image_processors([
        Processors.Threshold_Images(image_keys=('image',), lower_bounds=(-1000,), upper_bounds=(1500,), divides=(False,)),
        CreateExternal(image_key='image', output_key='external', threshold_value=-250.0, mask_value=1),
        Processors.DeepCopyKey(from_keys=('external',), to_keys=('og_external',)),
        Processors.Normalize_Images(keys=('image',), mean_values=(33.0,), std_values=(116.0,)),
        Processors.Threshold_Images(image_keys=('image',), lower_bounds=(-3.55,), upper_bounds=(3.55,), divides=(False,)),
        Processors.Per_Image_MinMax_Normalization(image_keys=('image',), threshold_value=1.0),
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.Resampler(resample_keys=('image', 'external'),
                  resample_interpolators=('Linear', 'Nearest'),
                  desired_output_spacing=[0.976, 0.976, 2.5],
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
    duodenum_model.set_prediction_processors([
        Processors.ExpandDimensions(image_keys=('og_external',), axis=-1),
        Processors.MaskOneBasedOnOther(guiding_keys=('og_external',),
                            changing_keys=('prediction',),
                            guiding_values=(0,),
                            mask_values=(0,)),
        ProcessPrediction(prediction_keys=('prediction',),
                          threshold={"1": 0.5},
                          connectivity={"1": False},
                          extract_main_comp={"1": True},
                          thread_count=1, dist={"1": 25}, max_comp={"1": 2}, min_vol={"1": 1500}),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v2' for roi in ["Duodenum"]]
    else:
        roi_names = ['Duodenum']

    duodenum_model.set_dicom_reader(TemplateDicomReader(roi_names=roi_names))
    return duodenum_model


def return_liver_ablation_3d_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    required_size = (32, 64, 64)
    ablation_3d_model = PredictWindowSliding(image_key='image', model_path=os.path.join(model_load_path,
                                                                                        'Liver_Ablation_3D',
                                                                                        'BasicUNet3D_Trial_15.hdf5'),
                                             model_template=BasicUnet3D(input_tensor=None,
                                                                        input_shape=required_size + (1,),
                                                                        classes=2, classifier_activation="softmax",
                                                                        activation="leakyrelu",
                                                                        normalization="batch", nb_blocks=3,
                                                                        nb_layers=5, dropout='standard',
                                                                        filters=32, dropout_rate=0.1,
                                                                        skip_type='concat',
                                                                        bottleneck='standard').get_net(),
                                             nb_label=2, required_size=required_size, sw_overlap=0.5, sw_batch_size=8,
                                             )
    paths = [
        os.path.join(shared_drive_path, 'Liver_Ablation_3D_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'Liver_Ablation_3D_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'Liver_Ablation_3D_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'Liver_Ablation_3D_Auto_Contour', 'Input_3')
    ]
    ablation_3d_model.set_paths(paths)
    ablation_3d_model.set_image_processors([
        Processors.DeepCopyKey(from_keys=('annotation',), to_keys=('og_annotation',)),
        Processors.Threshold_Images(image_keys=('image',), lower_bounds=(-1000,), upper_bounds=(1500,), divides=(False,)),
        ZNorm_By_Annotation(image_key='image', annotation_key='annotation'),
        Processors.Threshold_Images(image_keys=('image',), lower_bounds=(-3.55,), upper_bounds=(3.55,), divides=(False,)),
        Processors.Per_Image_MinMax_Normalization(image_keys=('image',), threshold_value=1.0),
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.Resampler(resample_keys=('image', 'annotation'),
                  resample_interpolators=('Linear', 'Nearest'),
                  desired_output_spacing=[None, None, 1],
                  post_process_resample_keys=('prediction',),
                  post_process_original_spacing_keys=('primary_handle',),
                  post_process_interpolators=('Linear',)),
        Box_Images(bounding_box_expansion=(0, 0, 0), image_keys=('image',),
                   annotation_key='annotation', wanted_vals_for_bbox=(1,),
                   power_val_z=required_size[0], power_val_r=required_size[1], power_val_c=required_size[2],
                   post_process_keys=('prediction',)),
        Processors.MaskOneBasedOnOther(guiding_keys=('annotation',),
                            changing_keys=('image',),
                            guiding_values=(0,),
                            mask_values=(0,)),
        Processors.ExpandDimensions(image_keys=('image',), axis=-1),
        Processors.ExpandDimensions(image_keys=('image',), axis=0),
        Processors.SqueezeDimensions(post_prediction_keys=('prediction',))
    ])

    ablation_3d_model.set_prediction_processors([
        Processors.ExpandDimensions(image_keys=('og_annotation',), axis=-1),
        Processors.MaskOneBasedOnOther(guiding_keys=('og_annotation',),
                            changing_keys=('prediction',),
                            guiding_values=(0,),
                            mask_values=(0,)),
        Duplicate_Prediction(prediction_key='prediction'),
        # ProcessPrediction(prediction_keys=('prediction',),
        #                   threshold={"1": 0.5},
        #                   connectivity={"1": False},
        #                   extract_main_comp={"1": False},
        #                   thread_count=1, dist={"1": None}, max_comp={"1": 2}, min_vol={"1": 5000}),
        Processors.Threshold_and_Expand(seed_threshold_value=[0.55, 0.50], lower_threshold_value=[.3, 0.5],
                             prediction_key='prediction'),
        Processors.Fill_Binary_Holes(prediction_key='prediction', dicom_handle_key='primary_handle'),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v0' for roi in ["Disease", "Ablation"]]
    else:
        roi_names = ["Disease", "Ablation"]

    ablation_3d_model.set_dicom_reader(EnsureLiverPresent(wanted_roi='Liver_BMA_Program_4',
                                                          roi_names=roi_names,
                                                          liver_folder=os.path.join(raystation_clinical_path,
                                                                                    'Liver_Auto_Contour', 'Input_3'),
                                                          associations={'Liver_MorfeusLab_v0': 'Liver_BMA_Program_4',
                                                                        'Liver_BMA_Program_4': 'Liver_BMA_Program_4',
                                                                        'Liver': 'Liver_BMA_Program_4'}))

    return ablation_3d_model


def return_psma_pb3D_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    required_size = (48, 192, 192)
    psma_model = PredictWindowSliding(image_key='image',
                                      model_path=os.path.join(model_load_path,
                                                              'PSMA_3D',
                                                              'BasicUNet3D_Trial_31.hdf5'),
                                      model_template=BasicUnet3D(input_tensor=None, input_shape=required_size + (1,),
                                                                 classes=5, classifier_activation="softmax",
                                                                 activation="leakyrelu",
                                                                 normalization="group", nb_blocks=2,
                                                                 nb_layers=5, dropout='standard',
                                                                 filters=32, dropout_rate=0.1,
                                                                 skip_type='concat', bottleneck='standard',
                                                                 last_conv_name='Output_3D_Conv_custom').get_net(),
                                      nb_label=5, required_size=required_size, sw_overlap=0.50, gaussiance_map=True,
                                      )
    paths = [
        os.path.join(shared_drive_path, 'PSMA_3D_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'PSMA_3D_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'PSMA_3D_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'PSMA_3D_Auto_Contour', 'Input_3')
    ]

    psma_model.set_paths(paths)
    psma_model.set_image_processors([
        Processors.DeepCopyKey(from_keys=('annotation',), to_keys=('og_annotation',)),
        CreateExternal(image_key='image', output_key='external', threshold_value=-250.0, mask_value=1),
        Processors.Threshold_Images(image_keys=('image',), lower_bounds=(-1000,), upper_bounds=(1500,), divides=(False,)),
        Processors.Per_Image_MinMax_Normalization(image_keys=('image',), threshold_value=1.0),
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.Resampler(resample_keys=('image', 'annotation'),
                  resample_interpolators=('Linear', 'Nearest'),
                  desired_output_spacing=[0.9765625, 0.9765625, 1.5],
                  post_process_resample_keys=('prediction',),
                  post_process_original_spacing_keys=('primary_handle',),
                  post_process_interpolators=('Linear',)),
        Box_Images(bounding_box_expansion=(0, 120, 0, 0, 0, 0), image_keys=('image',),
                   annotation_key='annotation', wanted_vals_for_bbox=(1,),
                   power_val_z=required_size[0], power_val_r=required_size[1], power_val_c=required_size[2],
                   post_process_keys=('prediction',), extract_comp=False),
        Processors.ExpandDimensions(image_keys=('image',), axis=-1),
        Processors.ExpandDimensions(image_keys=('image',), axis=0),
        Processors.SqueezeDimensions(post_prediction_keys=('prediction',))
    ])
    psma_model.set_prediction_processors([
        Processors.ExpandDimensions(image_keys=('external',), axis=-1),
        Processors.MaskOneBasedOnOther(guiding_keys=tuple(['external' for i in range(1, 5)]),
                            changing_keys=tuple(['prediction' for i in range(1, 5)]),
                            guiding_values=tuple([0 for i in range(1, 5)]),
                            mask_values=tuple([0 for i in range(1, 5)])),
        ProcessPrediction(prediction_keys=('prediction',),
                          threshold={"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
                          connectivity={"1": False, "2": False, "3": False, "4": False},
                          extract_main_comp={"1": False, "2": False, "3": False, "4": False},
                          dist={}, max_comp={}, min_vol={}, thread_count=4),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v4' for roi in
                     ['Bladder', 'Rectum', 'Iliac Veins', 'Iliac Arteries']]
    else:
        roi_names = ['Bladder', 'Rectum', 'Iliac Veins', 'Iliac Arteries']

    psma_model.set_dicom_reader(EnsureLiverPresent(wanted_roi='Femoral Heads',
                                                   roi_names=roi_names,
                                                   liver_folder=os.path.join(raystation_clinical_path,
                                                                             'FemHeads_Auto_Contour', 'Input_3'),
                                                   associations={'Femoral Heads_MorfeusLab_v0': 'Femoral Heads',
                                                                 'Femoral Heads': 'Femoral Heads'}))

    return psma_model


def return_psma_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    psma_model = ModelBuilderFromTemplate(image_key='image',
                                          model_path=os.path.join(model_load_path,
                                                                  'PSMA',
                                                                  'DLv3_model_Trial_0.hdf5'),
                                          model_template=deeplabv3plus(input_shape=(512, 512, 1),
                                                                       backbone="xception",
                                                                       classes=5, final_activation='softmax',
                                                                       windowopt_flag=True,
                                                                       normalization='batch', activation='relu',
                                                                       weights=None).Deeplabv3())
    paths = [
        os.path.join(shared_drive_path, 'PSMA_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'PSMA_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'PSMA_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'PSMA_Auto_Contour', 'Input_3')
    ]
    # TODO get spacing and clip by 60*3mm dist
    # see Clip_Images_By_Extension
    psma_model.set_paths(paths)
    psma_model.set_image_processors([
        Processors.DeepCopyKey(from_keys=('annotation',), to_keys=('og_annotation',)),
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.ExpandDimensions(axis=-1, image_keys=('image',)),
        Processors.Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
                                 post_process_keys=('image', 'prediction')),
    ])
    psma_model.set_prediction_processors([
        ProcessPrediction(prediction_keys=('prediction',),
                          threshold={"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
                          connectivity={"1": False, "2": False, "3": False, "4": False},
                          extract_main_comp={"1": False, "2": False, "3": False, "4": False},
                          dist={}, max_comp={}, min_vol={}, thread_count=4),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v3' for roi in
                     ['Bladder', 'Rectum', 'Iliac Veins', 'Iliac Arteries']]
    else:
        roi_names = ['Bladder', 'Rectum', 'Iliac Veins', 'Iliac Arteries']

    psma_model.set_dicom_reader(EnsureLiverPresent(wanted_roi='Femoral Heads',
                                                   roi_names=roi_names,
                                                   liver_folder=os.path.join(raystation_clinical_path,
                                                                             'FemHeads_Auto_Contour', 'Input_3'),
                                                   associations={'Femoral Heads_MorfeusLab_v0': 'Femoral Heads',
                                                                 'Femoral Heads': 'Femoral Heads'}))
    return psma_model


def return_femheads_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    femheads_model = ModelBuilderFromTemplate(image_key='image',
                                              model_path=os.path.join(model_load_path,
                                                                      'FemHeads',
                                                                      'DLv3_model_Trial_1.hdf5'),
                                              model_template=deeplabv3plus(input_shape=(512, 512, 1),
                                                                           backbone="xception",
                                                                           classes=2, final_activation='softmax',
                                                                           windowopt_flag=True,
                                                                           normalization='batch', activation='relu',
                                                                           weights=None).Deeplabv3())
    paths = [
        os.path.join(shared_drive_path, 'FemHeads_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'FemHeads_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'FemHeads_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'FemHeads_Auto_Contour', 'Input_3')
    ]
    femheads_model.set_paths(paths)
    femheads_model.set_image_processors([
        Processors.ExpandDimensions(axis=-1, image_keys=('image',)),
        Processors.Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
                                 post_process_keys=('image', 'prediction')),
    ])
    femheads_model.set_prediction_processors([
        ProcessPrediction(prediction_keys=('prediction',),
                          threshold={"1": 0.5},
                          connectivity={"1": False},
                          extract_main_comp={"1": True},
                          thread_count=1, dist={"1": None}, max_comp={"1": 2},
                          min_vol={"1": 2000}),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v0' for roi in ["Femoral Heads"]]
    else:
        roi_names = ["Femoral Heads"]

    femheads_model.set_dicom_reader(TemplateDicomReader(roi_names=roi_names))
    return femheads_model


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


def main():
    pass


if __name__ == '__main__':
    main()
