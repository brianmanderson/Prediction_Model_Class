import os, sys
sys.path.insert(0, os.path.abspath('..'))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *
import shutil
from functools import partial


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


class PredictLobes(BaseModelBuilder):
    def predict(self, input_features):
        pred = self.model.predict(input_features['combined'])
        input_features['prediction'] = np.squeeze(pred)
        return input_features


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
