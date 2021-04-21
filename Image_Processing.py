import copy, shutil, os, time, sys
sys.path.insert(0, os.path.abspath('.'))
from math import ceil, floor
from tensorflow.python.keras.utils.np_utils import to_categorical
from Resample_Class.src.NiftiResampler.ResampleTools import Resample_Class_Object, sitk
from Utils import np, get_bounding_box_indexes, remove_non_liver, plot_scroll_Image, plt, variable_remove_non_liver
# from DicomRTTool import DicomReaderWriter
from Dicom_RT_and_Images_to_Mask.src.DicomRTTool import DicomReaderWriter
from Fill_Missing_Segments.Fill_In_Segments_sitk import Fill_Missing_Segments
from skimage import morphology
import tensorflow as tf
import cv2


def dice_coef_3D(y_true, y_pred, smooth=0.0001):
    intersection = tf.keras.backend.sum(y_true[..., 1:] * y_pred[..., 1:])
    union = tf.keras.backend.sum(y_true[..., 1:]) + tf.keras.backend.sum(y_pred[..., 1:])
    return (2. * intersection + smooth) / (union + smooth)


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

    def build_model(self, graph, session):
        self.graph = graph
        self.session = session
        with graph.as_default():
            with self.session.as_default():
                if self.loss is not None and self.loss_weights is not None:
                    self.loss = self.loss(self.loss_weights)
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

    def load_images(self, input_features):
        self.dicom_reader.load(input_features=input_features)
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


class Base_Predictor(object):
    def __init__(self, model_path, graph, session, Bilinear_model=None, loss=None, loss_weights=None, image_key='image',
                 **kwargs):
        print('loaded vgg model ' + model_path)
        self.graph = graph
        self.session = session
        self.image_key = image_key
        with graph.as_default():
            with self.session.as_default():
                if tf.__version__ == '1.14.0':
                    if loss is not None and loss_weights is not None:
                        loss = loss(loss_weights)
                    print('loading VGG Pretrained')
                    self.model = tf.keras.models.load_model(model_path,
                                                            custom_objects={'BilinearUpsampling': Bilinear_model,
                                                                            'dice_coef_3D': dice_coef_3D,
                                                                            'loss': loss})
                else:
                    if loss is not None and loss_weights is not None:
                        loss = loss(loss_weights)
                    self.model = tf.keras.models.load_model(model_path,
                                                            custom_objects={'BilinearUpsampling': Bilinear_model,
                                                                            'dice_coef_3D': dice_coef_3D, 'loss': loss},
                                                            compile=False)

    def predict(self, input_features):
        input_features['prediction'] = self.model.predict(input_features[self.image_key])
        return input_features


class Predict_Lobes(Base_Predictor):
    def predict(self, input_features):
        pred = self.model.predict(input_features['combined'])
        input_features['prediction'] = np.squeeze(pred)
        return input_features


class Predict_Disease(Base_Predictor):
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


class template_dicom_reader(object):
    def __init__(self, roi_names, associations={'Liver_BMA_Program_4': 'Liver', 'Liver': 'Liver'}):
        self.status = True
        self.associations = associations
        self.roi_names = roi_names
        self.reader = DicomReaderWriter(associations=self.associations)

    def load(self, input_features):
        input_path = input_features['input_path']
        self.reader.__reset__()
        self.reader.walk_through_folders(input_path)
        self.reader.get_images()
        input_features['image'] = self.reader.ArrayDicom
        input_features['primary_handle'] = self.reader.dicom_handle
        return input_features

    def return_status(self):
        return self.status

    def pre_process(self, input_features):
        self.reader.get_images()
        input_features['image'] = self.reader.ArrayDicom
        input_features['primary_handle'] = self.reader.dicom_handle
        return input_features

    def post_process(self, input_features):
        return input_features

    def write_predicitons(self, input_features):
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


class Ensure_Liver_Disease_Segmentation(template_dicom_reader):
    def __init__(self, roi_names=None, associations=None, wanted_roi='Liver', liver_folder=None):
        super(Ensure_Liver_Disease_Segmentation, self).__init__(associations=associations, roi_names=roi_names)
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

    def process(self, input_features):
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
            self.reader.get_images()
            input_features['image'] = self.reader.ArrayDicom
            input_features['primary_handle'] = self.reader.dicom_handle
        return input_features

    def pre_process(self, input_features):
        self.dicom_handle = self.reader.dicom_handle
        self.reader.get_mask()
        input_features['image'] = self.reader.ArrayDicom
        input_features['primary_handle'] = self.reader.dicom_handle
        input_features['annotation'] = self.reader.mask
        return input_features

    def post_process(self, input_features):
        return input_features


class Ensure_Liver_Segmentation(Ensure_Liver_Disease_Segmentation):
    def __init__(self, roi_names=None, associations=None, wanted_roi='Liver', liver_folder=None):
        super(Ensure_Liver_Segmentation, self).__init__(associations=associations, roi_names=roi_names,
                                                        wanted_roi=wanted_roi, liver_folder=liver_folder)


def main():
    pass


if __name__ == '__main__':
    main()
