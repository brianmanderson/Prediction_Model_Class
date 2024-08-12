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


if __name__ == '__main__':
    pass
