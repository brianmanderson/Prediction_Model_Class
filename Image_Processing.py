import copy, shutil, os, time, sys
sys.path.insert(0, os.path.abspath('.'))
from math import ceil, floor
from tensorflow.python.keras.utils.np_utils import to_categorical
from Resample_Class.Resample_Class import Resample_Class_Object, sitk
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


class Base_Predictor(object):
    def __init__(self, model_path, graph, session, Bilinear_model=None, loss=None, loss_weights=None, **kwargs):
        print('loaded vgg model ' + model_path)
        self.graph = graph
        self.session = session
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

    def predict(self, images):
        return self.model.predict(images)


class Predict_Disease(Base_Predictor):
    def predict(self, images):
        x = images
        step = 64
        shift = 32
        gap = 8
        if x[0].shape[1] > step:
            pred = np.zeros(x[0][0].shape[:-1] + (2,))
            start = 0
            while start < x[0].shape[1]:
                image_cube, mask_cube = x[0][:, start:start + step, ...], x[1][:, start:start + step, ...]
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
            image_cube, mask_cube = x[0], x[1]
            difference = image_cube.shape[1] % 32
            if difference != 0:
                image_cube = np.pad(image_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
                mask_cube = np.pad(mask_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
            pred_cube = self.model.predict([image_cube, mask_cube])
            pred = pred_cube[:, difference:, ...]
        # pred = self.model.predict(x)
        return pred


class template_dicom_reader(object):
    def __init__(self, associations={'Liver_BMA_Program_4': 'Liver', 'Liver': 'Liver'}):
        self.status = True
        self.associations = associations
        self.reader = DicomReaderWriter(associations=self.associations)

    def process(self, dicom_folder):
        self.reader.__reset__()
        self.reader.walk_through_folders(dicom_folder)
        self.reader.get_images()
        self.dicom_handle = self.reader.dicom_handle

    def return_status(self):
        return self.status

    def pre_process(self):
        return self.reader.ArrayDicom, None

    def post_process(self, images, pred, ground_truth=None):
        return images, pred, ground_truth


class Image_Processor(object):

    def get_path(self, PathDicom):
        self.PathDicom = PathDicom

    def get_niftii_info(self, niftii_handle):
        self.dicom_handle = niftii_handle

    def pre_process(self, input_features):
        return input_features

    def post_process(self, input_features):
        return input_features


class Iterate_Overlap(Image_Processor):
    def __init__(self, on_liver_lobes=True, max_iterations=10, prediction_key='pred', ground_truth_key='annotations'):
        self.max_iterations = max_iterations
        self.on_liver_lobes = on_liver_lobes
        MauererDistanceMap = sitk.SignedMaurerDistanceMapImageFilter()
        MauererDistanceMap.SetInsideIsPositive(True)
        MauererDistanceMap.UseImageSpacingOn()
        MauererDistanceMap.SquaredDistanceOff()
        self.MauererDistanceMap = MauererDistanceMap
        self.Remove_Smallest_Structure = Remove_Smallest_Structures()
        self.Smooth_Annotation = SmoothingPredictionRecursiveGaussian()
        self.prediction_key = prediction_key
        self.ground_truth_key = ground_truth_key

    def remove_56_78(self, annotations):
        amounts = np.sum(annotations, axis=(1, 2))
        indexes = np.where((np.max(amounts[:, (5, 6)], axis=-1) > 0) & (np.max(amounts[:, (7, 8)], axis=-1) > 0))
        if indexes:
            indexes = indexes[0]
            for i in indexes:
                if amounts[i, 5] < amounts[i, 8]:
                    annotations[i, ..., 8] += annotations[i, ..., 5]
                    annotations[i, ..., 5] = 0
                else:
                    annotations[i, ..., 5] += annotations[i, ..., 8]
                    annotations[i, ..., 8] = 0
                if amounts[i, 6] < amounts[i, 7]:
                    annotations[i, ..., 7] += annotations[i, ..., 6]
                    annotations[i, ..., 6] = 0
                else:
                    annotations[i, ..., 6] += annotations[i, ..., 7]
                    annotations[i, ..., 7] = 0
        return annotations

    def iterate_annotations(self, annotations_out, ground_truth_out, spacing, allowed_differences=50, z_mult=1):
        '''
        :param annotations:
        :param ground_truth:
        :param spacing:
        :param allowed_differences:
        :param max_iteration:
        :param z_mult: factor by which to ensure slices don't bleed into ones above and below
        :return:
        '''
        self.Remove_Smallest_Structure.spacing = self.dicom_handle.GetSpacing()
        self.Smooth_Annotation.spacing = self.dicom_handle.GetSpacing()
        annotations_out[ground_truth_out == 0] = 0
        min_z, max_z, min_r, max_r, min_c, max_c = get_bounding_box_indexes(ground_truth_out)
        annotations = annotations_out[min_z:max_z, min_r:max_r, min_c:max_c, ...]
        ground_truth = ground_truth_out[min_z:max_z, min_r:max_r, min_c:max_c, ...]
        spacing[-1] *= z_mult
        differences = [np.inf]
        index = 0
        while differences[-1] > allowed_differences and index < self.max_iterations:
            index += 1
            print('Iterating {}'.format(index))
            # if self.on_liver_lobes:
            #     annotations = self.remove_56_78(annotations)
            previous_iteration = copy.deepcopy(np.argmax(annotations, axis=-1))
            for i in range(1, annotations.shape[-1]):
                annotation_handle = sitk.GetImageFromArray(annotations[..., i])
                annotation_handle.SetSpacing(self.dicom_handle.GetSpacing())
                pruned_handle = self.Remove_Smallest_Structure.remove_smallest_component(annotation_handle)
                annotations[..., i] = sitk.GetArrayFromImage(pruned_handle)
                slices = np.where(annotations[..., i] == 1)
                if slices:
                    slices = np.unique(slices[0])
                    for ii in range(len(slices)):
                        image_handle = sitk.GetImageFromArray(annotations[slices[ii], ..., i][None, ...])
                        pruned_handle = self.Remove_Smallest_Structure.remove_smallest_component(image_handle)
                        annotations[slices[ii], ..., i] = sitk.GetArrayFromImage(pruned_handle)

            annotations = self.make_distance_map(annotations, ground_truth, spacing=spacing)
            differences.append(np.abs(
                np.sum(previous_iteration[ground_truth == 1] - np.argmax(annotations, axis=-1)[ground_truth == 1])))
        annotations_out[min_z:max_z, min_r:max_r, min_c:max_c, ...] = annotations
        annotations_out[ground_truth_out == 0] = 0
        return annotations_out

    def run_distance_map(self, array, spacing):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing(spacing)
        output = self.MauererDistanceMap.Execute(image)
        output = sitk.GetArrayFromImage(output)
        return output

    def make_distance_map(self, pred, liver, reduce=True, spacing=(0.975, 0.975, 2.5)):
        '''
        :param pred: A mask of your predictions with N channels on the end, N=0 is background [# Images, rows, cols, N]
        :param liver: A mask of the desired region [# Images, rows, cols]
        :param reduce: Save time and only work on masked region
        :return:
        '''
        liver = np.squeeze(liver)
        pred = np.squeeze(pred)
        pred = np.round(pred).astype('int')
        min_z, min_r, min_c, max_z, max_r, max_c = 0, 0, 0, pred.shape[0], pred.shape[1], pred.shape[2]

        if reduce:
            min_z, max_z, min_r, max_r, min_c, max_c = get_bounding_box_indexes(liver)
        reduced_pred = pred[min_z:max_z, min_r:max_r, min_c:max_c]
        reduced_liver = liver[min_z:max_z, min_r:max_r, min_c:max_c]
        reduced_output = np.zeros(reduced_pred.shape)
        for i in range(1, pred.shape[-1]):
            temp_reduce = reduced_pred[..., i]
            output = self.run_distance_map(temp_reduce, spacing)
            reduced_output[..., i] = output
        reduced_output[reduced_output > 0] = 0
        reduced_output = np.abs(reduced_output)
        reduced_output[..., 0] = np.inf
        output = np.zeros(reduced_output.shape, dtype='int')
        mask = reduced_liver == 1
        values = reduced_output[mask]
        output[mask, np.argmin(values, axis=-1)] = 1
        pred[min_z:max_z, min_r:max_r, min_c:max_c] = output
        return pred

    def post_process(self, input_features):
        pred = input_features[self.prediction_key]
        ground_truth = input_features[self.ground_truth_key]
        pred = self.iterate_annotations(pred, ground_truth, spacing=list(self.dicom_handle.GetSpacing()), z_mult=1)
        input_features[self.prediction_key] = pred
        return input_features


class Remove_Smallest_Structures(Image_Processor):
    def __init__(self):
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        self.RelabelComponent.SortByObjectSizeOn()

    def remove_smallest_component(self, annotation_handle):
        label_image = self.Connected_Component_Filter.Execute(
            sitk.BinaryThreshold(sitk.Cast(annotation_handle, sitk.sitkFloat32), lowerThreshold=0.01,
                                 upperThreshold=np.inf))
        label_image = self.RelabelComponent.Execute(label_image)
        output = sitk.BinaryThreshold(sitk.Cast(label_image, sitk.sitkFloat32), lowerThreshold=0.1, upperThreshold=1.0)
        return output


class Threshold_and_Expand(Image_Processor):
    def __init__(self, seed_threshold_value=None, lower_threshold_value=None, prediction_key='pred'):
        self.seed_threshold_value = seed_threshold_value
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        self.Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        self.stats = sitk.LabelShapeStatisticsImageFilter()
        self.lower_threshold_value = lower_threshold_value
        self.Connected_Threshold.SetUpper(2)
        self.prediction_key = prediction_key

    def post_process(self, input_features):
        pred = input_features[self.prediction_key]
        for i in range(1, pred.shape[-1]):
            temp_pred = pred[..., i]
            output = np.zeros(temp_pred.shape)
            expanded = False
            if len(temp_pred.shape) == 4:
                temp_pred = temp_pred[0]
                expanded = True
            prediction = sitk.GetImageFromArray(temp_pred)
            if type(self.seed_threshold_value) is not list:
                seed_threshold = self.seed_threshold_value
            else:
                seed_threshold = self.seed_threshold_value[i - 1]
            if type(self.lower_threshold_value) is not list:
                lower_threshold = self.lower_threshold_value
            else:
                lower_threshold = self.lower_threshold_value[i - 1]
            overlap = temp_pred > seed_threshold
            if np.max(overlap) > 0:
                seeds = np.transpose(np.asarray(np.where(overlap > 0)))[..., ::-1]
                seeds = [[int(i) for i in j] for j in seeds]
                self.Connected_Threshold.SetLower(lower_threshold)
                self.Connected_Threshold.SetSeedList(seeds)
                output = sitk.GetArrayFromImage(self.Connected_Threshold.Execute(prediction))
                if expanded:
                    output = output[None, ...]
            pred[..., i] = output
        input_features[self.prediction_key] = pred
        return input_features


def createthreshold(predictionimage, seeds, thresholdvalue):
    Connected_Threshold = sitk.ConnectedThresholdImageFilter()
    Connected_Threshold.SetUpper(2)
    Connected_Threshold.SetSeedList(seeds)
    Connected_Threshold.SetLower(thresholdvalue)
    threshold_prediction = Connected_Threshold.Execute(sitk.Cast(predictionimage, sitk.sitkFloat32))
    del Connected_Threshold, predictionimage, seeds, thresholdvalue
    return threshold_prediction


def createseeds(predictionimage, seed_value):
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    stats = sitk.LabelShapeStatisticsImageFilter()
    thresholded_image = sitk.BinaryThreshold(sitk.Cast(predictionimage, sitk.sitkFloat32), lowerThreshold=seed_value)
    connected_image = Connected_Component_Filter.Execute(thresholded_image)
    stats.Execute(connected_image)
    seeds = [stats.GetCentroid(l) for l in stats.GetLabels()]
    seeds = [thresholded_image.TransformPhysicalPointToIndex(i) for i in seeds]
    del stats, Connected_Component_Filter, connected_image, predictionimage, seed_value
    return seeds


class Iterate_Lobe_Annotations(object):
    def __init__(self):
        self.remove_smallest = Remove_Smallest_Structures()
        MauererDistanceMap = sitk.SignedMaurerDistanceMapImageFilter()
        MauererDistanceMap.SetInsideIsPositive(True)
        MauererDistanceMap.UseImageSpacingOn()
        MauererDistanceMap.SquaredDistanceOff()
        self.BinaryfillFilter = sitk.BinaryFillholeImageFilter()
        self.BinaryfillFilter.SetFullyConnected(True)
        self.BinaryfillFilter = sitk.BinaryMorphologicalClosingImageFilter()
        self.BinaryfillFilter.SetKernelRadius((3, 3, 1))
        self.BinaryfillFilter.SetKernelType(sitk.sitkBall)
        self.MauererDistanceMap = MauererDistanceMap

    def remove_56_78(self, annotations):
        amounts = np.sum(annotations, axis=(1, 2))
        indexes = np.where((np.max(amounts[:, (5, 6)], axis=-1) > 0) & (np.max(amounts[:, (7, 8)], axis=-1) > 0))
        if indexes:
            indexes = indexes[0]
            for i in indexes:
                if amounts[i, 5] < amounts[i, 8]:
                    annotations[i, ..., 8] += annotations[i, ..., 5]
                    annotations[i, ..., 5] = 0
                else:
                    annotations[i, ..., 5] += annotations[i, ..., 8]
                    annotations[i, ..., 8] = 0
                if amounts[i, 6] < amounts[i, 7]:
                    annotations[i, ..., 7] += annotations[i, ..., 6]
                    annotations[i, ..., 6] = 0
                else:
                    annotations[i, ..., 6] += annotations[i, ..., 7]
                    annotations[i, ..., 7] = 0
        return annotations

    def iterate_annotations(self, annotations_base, ground_truth_base, spacing, allowed_differences=50,
                            max_iteration=15, reduce2D=True):
        '''
        :param annotations:
        :param ground_truth:
        :param spacing:
        :param allowed_differences:
        :param max_iteration:
        :param z_mult: factor by which to ensure slices don't bleed into ones above and below
        :return:
        '''
        differences = [np.inf]
        index = 0
        liver = np.squeeze(ground_truth_base)
        min_z_s, max_z_s, min_r_s, max_r_s, min_c_s, max_c_s = get_bounding_box_indexes(liver)
        annotations = annotations_base[min_z_s:max_z_s, min_r_s:max_r_s, min_c_s:max_c_s]
        ground_truth = ground_truth_base[min_z_s:max_z_s, min_r_s:max_r_s, min_c_s:max_c_s]
        while differences[-1] > allowed_differences and index < max_iteration:
            previous_iteration = copy.deepcopy(np.argmax(annotations, axis=-1))
            for i in range(1, annotations.shape[-1]):
                annotation = annotations[..., i]
                if reduce2D:
                    # start = time.time()
                    slices = np.where(np.max(annotation, axis=(1, 2)) > 0)
                    for slice in slices[0]:
                        annotation[slice] = sitk.GetArrayFromImage(self.remove_smallest.remove_smallest_component(
                            sitk.GetImageFromArray(annotation[slice].astype('float32')) > 0))
                    # print('Took {} seconds'.format(time.time()-start))
                # start = time.time()
                annotations[..., i] = sitk.GetArrayFromImage(self.remove_smallest.remove_smallest_component(
                    sitk.GetImageFromArray(annotation.astype('float32')) > 0))
                # print('Took {} seconds'.format(time.time() - start))
            # annotations = self.remove_56_78(annotations)
            summed = np.sum(annotations, axis=-1)
            annotations[summed > 1] = 0
            annotations[annotations > 0] = 1
            annotations[..., 0] = 1 - ground_truth
            # start = time.time()
            annotations = self.make_distance_map(annotations, ground_truth, spacing=spacing)
            differences.append(np.abs(
                np.sum(previous_iteration[ground_truth == 1] - np.argmax(annotations, axis=-1)[ground_truth == 1])))
            index += 1
        annotations_base[min_z_s:max_z_s, min_r_s:max_r_s, min_c_s:max_c_s] = annotations
        annotations_base[..., 0] = 1 - ground_truth_base
        return annotations_base

    def run_distance_map(self, array, spacing):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing(spacing)
        output = self.MauererDistanceMap.Execute(image)
        output = sitk.GetArrayFromImage(output)
        return output

    def make_distance_map(self, pred, liver, spacing=(0.975, 0.975, 2.5)):
        '''
        :param pred: A mask of your predictions with N channels on the end, N=0 is background [# Images, 512, 512, N]
        :param liver: A mask of the desired region [# Images, 512, 512]
        :param reduce: Save time and only work on masked region
        :return:
        '''
        liver = np.squeeze(liver)
        pred = np.squeeze(pred)
        pred = np.round(pred).astype('int')
        min_z, min_r, min_c = 0, 0, 0
        max_z, max_r, max_c = pred.shape[:3]
        reduced_pred = pred[min_z:max_z, min_r:max_r, min_c:max_c]
        reduced_liver = liver[min_z:max_z, min_r:max_r, min_c:max_c]
        reduced_output = np.zeros(reduced_pred.shape)
        for i in range(1, pred.shape[-1]):
            temp_reduce = reduced_pred[..., i]
            output = self.run_distance_map(temp_reduce, spacing)
            reduced_output[..., i] = output
        reduced_output[reduced_output > 0] = 0
        reduced_output = np.abs(reduced_output)
        reduced_output[..., 0] = np.inf
        output = np.zeros(reduced_output.shape, dtype='int')
        mask = reduced_liver == 1
        values = reduced_output[mask]
        output[mask, np.argmin(values, axis=-1)] = 1
        pred[min_z:max_z, min_r:max_r, min_c:max_c] = output
        return pred


class Threshold_and_Expand_New(Image_Processor):
    def __init__(self, seed_threshold_value=None, lower_threshold_value=None, prediction_key='prediction',
                 ground_truth_key='annotation'):
        self.seed_threshold_value = seed_threshold_value
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        self.Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        self.stats = sitk.LabelShapeStatisticsImageFilter()
        self.lower_threshold_value = lower_threshold_value
        self.Connected_Threshold.SetUpper(2)
        self.prediction_key = prediction_key
        self.ground_truth_key = ground_truth_key
        self.Iterate_Lobe_Annotations_Class = Iterate_Lobe_Annotations()

    def post_process(self, input_features):
        pred = input_features[self.prediction_key]
        ground_truth = input_features[self.ground_truth_key]
        out_prediction = np.zeros(pred.shape).astype('float32')
        for i in range(1, out_prediction.shape[-1]):
            out_prediction[..., i] = sitk.GetArrayFromImage(
                createthreshold(sitk.GetImageFromArray(pred[..., i].astype('float32')),
                                createseeds(sitk.GetImageFromArray(pred[..., i].astype('float32')),
                                            self.seed_threshold_value[i - 1]),
                                self.lower_threshold_value[i - 1]))
        summed_image = np.sum(out_prediction, axis=-1)
        # stop = time.time()
        out_prediction[summed_image > 1] = 0
        out_prediction = self.Iterate_Lobe_Annotations_Class.iterate_annotations(
            out_prediction, ground_truth > 0,
            spacing=self.dicom_handle.GetSpacing(),
            max_iteration=10, reduce2D=False)
        input_features[self.prediction_key] = out_prediction
        return input_features


class Mask_within_Liver(Image_Processor):
    def __init__(self, prediction_key, ground_truth_key):
        self.prediction_key = prediction_key
        self.ground_truth_key = ground_truth_key

    def post_process(self, input_features):
        pred = input_features[self.prediction_key]
        ground_truth = input_features[self.ground_truth_key]
        pred[ground_truth == 0] = 0
        input_features[self.prediction_key] = pred
        return input_features


class Fill_Binary_Holes(Image_Processor):
    def __init__(self, prediction_key, dicom_handle_key):
        self.BinaryfillFilter = sitk.BinaryFillholeImageFilter()
        self.BinaryfillFilter.SetFullyConnected(True)
        self.prediction_key = prediction_key
        self.dicom_handle_key = dicom_handle_key

    def post_process(self, input_features):
        pred = input_features[self.prediction_key]
        dicom_handle = input_features[self.dicom_handle_key]
        for class_num in range(1, pred.shape[-1]):
            temp_pred = pred[..., class_num]
            k = sitk.GetImageFromArray(temp_pred.astype('int'))
            k.SetSpacing(dicom_handle.GetSpacing())
            output = self.BinaryfillFilter.Execute(k)
            output_array = sitk.GetArrayFromImage(output)
            pred[..., class_num] = output_array
        input_features[self.prediction_key] = pred
        return input_features


class Minimum_Volume_and_Area_Prediction(Image_Processor):
    '''
    This should come after prediction thresholding
    '''

    def __init__(self, min_volume=0.0, min_area=0.0, max_area=np.inf, pred_axis=[1]):
        '''
        :param min_volume: Minimum volume of structure allowed, in cm3
        :param min_area: Minimum area of structure allowed, in cm2
        :param max_area: Max area of structure allowed, in cm2
        :return: Masked annotation
        '''
        self.min_volume = min_volume * 1000  # cm3 to mm3
        self.min_area = min_area * 100
        self.max_area = max_area * 100
        self.pred_axis = pred_axis
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()

    def post_process(self, images, pred, ground_truth=None):
        for axis in self.pred_axis:
            temp_pred = pred[..., axis]
            if self.min_volume != 0:
                label_image = self.Connected_Component_Filter.Execute(sitk.GetImageFromArray(temp_pred) > 0)
                self.RelabelComponent.SetMinimumObjectSize(
                    int(self.min_volume / np.prod(self.dicom_handle.GetSpacing())))
                label_image = self.RelabelComponent.Execute(label_image)
                temp_pred = sitk.GetArrayFromImage(label_image > 0)
            if self.min_area != 0 or self.max_area != np.inf:
                slice_indexes = np.where(np.sum(temp_pred, axis=(1, 2)) > 0)
                if slice_indexes:
                    slice_spacing = np.prod(self.dicom_handle.GetSpacing()[:-1])
                    for slice_index in slice_indexes[0]:
                        labels = morphology.label(temp_pred[slice_index], connectivity=1)
                        for i in range(1, labels.max() + 1):
                            new_area = labels[labels == i].shape[0]
                            temp_area = slice_spacing * new_area
                            if temp_area > self.max_area:
                                labels[labels == i] = 0
                                continue
                            elif temp_area < self.min_area:
                                labels[labels == i] = 0
                                continue
                        labels[labels > 0] = 1
                        temp_pred[slice_index] = labels
            if self.min_volume != 0:
                label_image = self.Connected_Component_Filter.Execute(sitk.GetImageFromArray(temp_pred) > 0)
                self.RelabelComponent.SetMinimumObjectSize(
                    int(self.min_volume / np.prod(self.dicom_handle.GetSpacing())))
                label_image = self.RelabelComponent.Execute(label_image)
                temp_pred = sitk.GetArrayFromImage(label_image > 0)
            pred[..., axis] = temp_pred
        return images, pred, ground_truth


class SmoothingPredictionRecursiveGaussian(Image_Processor):
    def __init__(self, sigma=(0.1, 0.1, 0.0001), pred_axis=[1]):
        self.sigma = sigma
        self.pred_axis = pred_axis

    def smooth(self, handle):
        return sitk.BinaryThreshold(sitk.SmoothingRecursiveGaussian(handle), lowerThreshold=.01, upperThreshold=np.inf)

    def post_process(self, images, pred, ground_truth=None):
        for axis in self.pred_axis:
            k = sitk.GetImageFromArray(pred[..., axis])
            k.SetSpacing(self.dicom_handle.GetSpacing())
            k = self.smooth(k)
            pred[..., axis] = sitk.GetArrayFromImage(k)
        return images, pred, ground_truth


class To_Categorical(Image_Processor):
    def __init__(self, num_classes, is_preprocessing=True, is_post_processing=False):
        self.num_classes = num_classes
        self.is_preprocessing, self.is_post_processing = is_preprocessing, is_post_processing

    def pre_process(self, images, annotations=None):
        if self.is_preprocessing:
            annotations = to_categorical(annotations, self.num_classes)
        return images, annotations

    def post_process(self, images, pred, ground_truth=None):
        if self.is_post_processing:
            pred = to_categorical(pred, self.num_classes)
        return images, pred, ground_truth


class Normalize_to_Liver_New(Image_Processor):
    def __init__(self, mirror_max=False, lower_percentile=None, upper_percentile=None):
        '''
        :param annotation_value: mask values to normalize over, [1]
        '''
        self.mirror_max = mirror_max
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def pre_process(self, images, annotations=None):
        liver = annotations == 1
        data = images[liver == 1].flatten()
        counts, bins = np.histogram(data, bins=100)
        bins = bins[:-1]
        count_index = np.where(counts == np.max(counts))[0][-1]
        peak = bins[count_index]
        data_reduced = data[np.where((data > peak - 150) & (data < peak + 150))]
        counts, bins = np.histogram(data_reduced, bins=1000)
        bins = bins[:-1]
        count_index = np.where(counts == np.max(counts))[0][-1]
        half_counts = counts - np.max(counts) // 2
        half_upper = np.abs(half_counts[count_index + 1:])
        max_50 = np.where(half_upper == np.min(half_upper))[0][0]

        half_lower = np.abs(half_counts[:count_index - 1][-1::-1])
        min_50 = np.where(half_lower == np.min(half_lower))[0][0]

        min_values = bins[count_index - min_50]
        if self.mirror_max:
            min_values = bins[count_index - max_50]  # Good for non-normal distributions, just mirror the other FWHM
        max_values = bins[count_index + max_50]
        data = data[np.where((data >= min_values) & (data <= max_values))]
        mean_val, std_val = np.mean(data), np.std(data)
        images = (images - mean_val) / std_val
        return images, annotations


class Normalize_to_Liver_Old(Image_Processor):
    def __init__(self, lower_fraction=0, upper_fraction=1):
        '''
        This is a little tricky... We only want to perform this task once, since it requires potentially large
        computation time, but it also requires that all individual image slices already be loaded
        '''
        self.lower_fraction = lower_fraction
        self.upper_fraction = upper_fraction

    def pre_process(self, images, annotations=None):
        data = images[annotations == 1].flatten()
        data.sort()
        data = data[int(len(data) * self.lower_fraction):int(len(data) * self.upper_fraction)]
        mean_val = np.mean(data)
        std_val = np.std(data)
        images = (images - mean_val) / std_val
        return images, annotations


class Normalize_to_Liver(Image_Processor):
    def __init__(self, mirror_max=False):
        '''
        This is a little tricky... We only want to perform this task once, since it requires potentially large
        computation time, but it also requires that all individual image slices already be loaded
        '''
        # Now performing FWHM
        self.mirror_max = mirror_max

    def pre_process(self, images, annotations=None):
        liver = annotations == 1
        data = images[liver == 1].flatten()
        counts, bins = np.histogram(data, bins=100)
        bins = bins[:-1]
        count_index = np.where(counts == np.max(counts))[0][-1]
        peak = bins[count_index]
        data_reduced = data[np.where((data > peak - 150) & (data < peak + 150))]
        counts, bins = np.histogram(data_reduced, bins=1000)
        bins = bins[:-1]
        count_index = np.where(counts == np.max(counts))[0][-1]
        half_counts = counts - np.max(counts) // 2
        half_upper = np.abs(half_counts[count_index + 1:])
        max_50 = np.where(half_upper == np.min(half_upper))[0][0]

        half_lower = np.abs(half_counts[:count_index - 1][-1::-1])
        min_50 = np.where(half_lower == np.min(half_lower))[0][0]

        min_values = bins[count_index - min_50]
        if self.mirror_max:
            min_values = bins[count_index - max_50]  # Good for non-normal distributions, just mirror the other FWHM
        max_values = bins[count_index + max_50]
        data = data[np.where((data >= min_values) & (data <= max_values))]
        mean_val, std_val = np.mean(data), np.std(data)
        images = (images - mean_val) / std_val
        return images, annotations


class Mask_Prediction_New(Image_Processor):
    def pre_process(self, images, annotations=None):
        images[annotations == 0] = 0
        return [images, annotations], annotations

    def post_process(self, images, pred, ground_truth=None):
        return images[0], pred, ground_truth


class Mask_Prediction(Image_Processor):
    def __init__(self, num_repeats, liver_lower=None):
        self.num_repeats = num_repeats
        self.liver_lower = liver_lower

    def pre_process(self, images, annotations=None):
        mask = np.repeat(annotations, self.num_repeats, axis=-1)
        if self.liver_lower is not None:
            inside = images[mask[..., 1] == 1]
            inside[inside < self.liver_lower] = self.liver_lower
            images[mask[..., 1] == 1] = inside
        sum_vals = np.zeros(mask.shape)
        sum_vals[..., 0] = 1 - mask[..., 0]
        return [images, annotations], annotations

    def post_process(self, images, pred, ground_truth=None):
        return images[0], pred, ground_truth


class remove_potential_ends_threshold(Image_Processor):
    def __init__(self, threshold=-1000):
        self.threshold = threshold

    def post_process(self, images, pred, ground_truth=None):
        indexes = np.where(pred == 1)
        values = images[indexes]
        unique_values = np.unique(indexes[0])
        for i in unique_values:
            mean_val = np.mean(values[indexes[0] == i])
            if mean_val < self.threshold:
                pred[i] = 0
        return images, pred, ground_truth


class remove_potential_ends_size(Image_Processor):

    def post_process(self, images, pred, ground_truth=None):
        sum_slice = tuple(range(len(pred.shape))[1:])
        slices = np.where(sum_slice > 0)
        if slices and len(slices[0]) > 10:
            reduced_slices = sum_slice[slices[0]]
            local_min = (np.diff(np.sign(np.diff(reduced_slices))) > 0).nonzero()[0] + 1  # local min
            local_max = (np.diff(np.sign(np.diff(reduced_slices))) < 0).nonzero()[0] + 1  # local max
            total_slices = len(slices[0]) // 5 + 1
            for index in range(total_slices):
                if reduced_slices[index] > reduced_slices[index + 1]:
                    pred[reduced_slices[index]]
        indexes = np.where(pred == 1)
        values = images[indexes]
        unique_values = np.unique(indexes[0])
        for i in unique_values:
            mean_val = np.mean(values[indexes[0] == i])
            if mean_val < self.threshold:
                pred[i] = 0
        return images, pred, ground_truth


class Make_3D(Image_Processor):
    def pre_process(self, images, annotations=None):
        return images[None, ...], annotations

    def post_process(self, images, pred, ground_truth=None):
        return np.squeeze(images), np.squeeze(pred), ground_truth


class Reduce_Prediction(Image_Processor):
    def post_process(self, images, pred, ground_truth=None):
        pred[pred < 0.5] = 0
        return images, pred, ground_truth


class Box_Images(Image_Processor):
    def __init__(self, bbox=(5, 20, 20)):
        self.bbox = bbox

    def pre_process(self, images, annotations=None):
        self.boxed = False
        if annotations is None:
            return images, annotations
        self.boxed = True
        self.images_shape = images.shape
        self.z_start, z_stop, self.r_start, r_stop, self.c_start, c_stop = get_bounding_box_indexes(annotations,
                                                                                                    bbox=self.bbox)
        self.z_dif = images.shape[0] - z_stop
        self.r_dif = images.shape[1] - r_stop
        self.c_dif = images.shape[2] - c_stop
        images = images[self.z_start:z_stop, self.r_start:r_stop, self.c_start:c_stop]
        annotations = annotations[self.z_start:z_stop, self.r_start:r_stop, self.c_start:c_stop]
        return images, annotations

    def post_process(self, images, pred, ground_truth=None):
        images, pred = np.squeeze(images), np.squeeze(pred)
        if ground_truth is not None:
            ground_truth = np.squeeze(ground_truth)
        if len(images.shape) == 3:
            pad = [[self.z_start, self.z_dif], [self.r_start, self.r_dif], [self.c_start, self.c_dif]]
        elif len(images.shape) == 4:
            pad = [[0, 0], [self.z_start, self.z_dif], [self.r_start, self.r_dif], [self.c_start, self.c_dif]]
        else:
            pad = [[0, 0], [self.z_start, self.z_dif], [self.r_start, self.r_dif], [self.c_start, self.c_dif], [0, 0]]
        images = np.pad(images, pad)

        if len(pred.shape) == 4:
            pad = [[self.z_start, self.z_dif], [self.r_start, self.r_dif], [self.c_start, self.c_dif], [0, 0]]
        else:
            pad = [[0, 0], [self.z_start, self.z_dif], [self.r_start, self.r_dif], [self.c_start, self.c_dif], [0, 0]]
        pred = np.pad(pred, pad)

        if ground_truth is not None:
            if len(ground_truth.shape) == 3:
                pad = [[self.z_start, self.z_dif], [self.r_start, self.r_dif], [self.c_start, self.c_dif]]
            elif len(ground_truth.shape) == 4:
                pad = [[0, 0], [self.z_start, self.z_dif], [self.r_start, self.r_dif], [self.c_start, self.c_dif]]
            else:
                pad = [[0, 0], [self.z_start, self.z_dif], [self.r_start, self.r_dif], [self.c_start, self.c_dif],
                       [0, 0]]
            ground_truth = np.pad(ground_truth, pad)
        return images, pred, ground_truth


class Pad_Images(Image_Processor):
    def __init__(self, bounding_box_expansion=(10, 10, 10), power_val_z=1, power_val_x=1,
                 power_val_y=1, min_val=None):
        self.bounding_box_expansion = bounding_box_expansion
        self.min_val = min_val
        self.power_val_z, self.power_val_x, self.power_val_y = power_val_z, power_val_x, power_val_y

    def pre_process(self, images, annotations=None):
        images_shape = images.shape
        self.og_shape = images_shape
        z_start, r_start, c_start = 0, 0, 0
        z_stop, r_stop, c_stop = images_shape[0], images_shape[1], images_shape[2]
        z_total, r_total, c_total = z_stop - z_start, r_stop - r_start, c_stop - c_start
        self.remainder_z, self.remainder_r, self.remainder_c = self.power_val_z - z_total % self.power_val_z if z_total % self.power_val_z != 0 else 0, \
                                                               self.power_val_x - r_total % self.power_val_x if r_total % self.power_val_x != 0 else 0, \
                                                               self.power_val_y - c_total % self.power_val_y if c_total % self.power_val_y != 0 else 0
        pads = [self.remainder_z, self.remainder_r, self.remainder_c]
        self.pad = [[max([0, floor(i / 2)]), max([0, ceil(i / 2)])] for i in pads]
        if len(images_shape) > 3:
            self.pad = [[0, 0]] + self.pad
        if self.min_val is None:
            min_val = np.min(images)
        else:
            min_val = self.min_val
        images = np.pad(images, self.pad, constant_values=min_val)
        if annotations is not None:
            annotations = np.pad(annotations, pad_width=self.pad, constant_values=np.min(annotations))
        return images, annotations

    def post_process(self, images, pred, ground_truth=None):
        if max([self.remainder_z, self.remainder_r, self.remainder_c]) == 0:
            return images, pred, ground_truth
        if len(pred.shape) == 3 or len(pred.shape) == 4:
            pred = pred[self.pad[0][0]:, self.pad[1][0]:, self.pad[2][0]:]
            pred = pred[:self.og_shape[0], :self.og_shape[1], :self.og_shape[2]]
        elif len(pred.shape) == 5:
            pred = pred[:, self.pad[0][0]:, self.pad[1][0]:, self.pad[2][0]:]
            pred = pred[:, :self.og_shape[0], :self.og_shape[1], :self.og_shape[2]]
        if len(images.shape) == 3 or len(images.shape) == 4:
            images = images[self.pad[0][0]:, self.pad[1][0]:, self.pad[2][0]:]
            images = images[:self.og_shape[0], :self.og_shape[1], :self.og_shape[2]]
        elif len(images.shape) == 5:
            images = images[:, self.pad[0][0]:, self.pad[1][0]:, self.pad[2][0]:]
            images = images[:, :self.og_shape[0], :self.og_shape[1], :self.og_shape[2]]

        if ground_truth is not None:
            if len(ground_truth.shape) == 3 or len(ground_truth.shape) == 4:
                ground_truth = ground_truth[self.pad[0][0]:, self.pad[1][0]:, self.pad[2][0]:]
                ground_truth = ground_truth[:self.og_shape[0], :self.og_shape[1], :self.og_shape[2]]
            elif len(ground_truth.shape) == 5:
                ground_truth = ground_truth[:, self.pad[0][0]:, self.pad[1][0]:, self.pad[2][0]:]
                ground_truth = ground_truth[:, :self.og_shape[0], :self.og_shape[1], :self.og_shape[2]]
        return images, pred, ground_truth


class Image_Clipping_and_Padding(Image_Processor):
    def __init__(self, layers=3, return_mask=False, liver_box=False, mask_output=False):
        self.mask_output = mask_output
        self.patient_dict = {}
        self.liver_box = liver_box
        power_val_z, power_val_x, power_val_y = (1, 1, 1)
        pool_base = 2
        for layer in range(layers):
            pooling = [pool_base for _ in range(3)]
            power_val_z *= pooling[0]
            power_val_x *= pooling[1]
            power_val_y *= pooling[2]
        self.return_mask = return_mask
        self.power_val_z, self.power_val_x, self.power_val_y = power_val_z, power_val_x, power_val_y

    def pre_process(self, images, annotations=None):
        x, y = images, annotations
        if self.liver_box and y is not None:
            liver = np.argmax(y, axis=-1)
            z_start, z_stop, r_start, r_stop, c_start, c_stop = get_bounding_box_indexes(liver)
            z_start = max([0, z_start - 5])
            z_stop = min([z_stop + 5, x.shape[1]])
            r_start = max([0, r_start - 10])
            r_stop = min([x.shape[2], r_stop + 10])
            c_start = max([0, c_start - 10])
            c_stop = min([x.shape[3], c_stop + 10])
        else:
            z_start = 0
            z_stop = x.shape[0]
            r_start = 0
            r_stop = x.shape[1]
            c_start = 0
            c_stop = x.shape[2]
        z_total, r_total, c_total = z_stop - z_start, r_stop - r_start, c_stop - c_start
        remainder_z, remainder_r, remainder_c = self.power_val_z - z_total % self.power_val_z if z_total % self.power_val_z != 0 else 0, \
                                                self.power_val_x - r_total % self.power_val_x if r_total % self.power_val_x != 0 else 0, \
                                                self.power_val_y - c_total % self.power_val_y if c_total % self.power_val_y != 0 else 0
        self.z, self.r, self.c = remainder_z, remainder_r, remainder_c
        min_images, min_rows, min_cols = z_total + remainder_z, r_total + remainder_r, c_total + remainder_c
        out_images = np.ones([min_images, min_rows, min_cols, x.shape[-1]], dtype=x.dtype) * np.min(x)
        out_images[0:z_stop - z_start, :r_stop - r_start, :c_stop - c_start, :] = x[z_start:z_stop, r_start:r_stop,
                                                                                  c_start:c_stop, :]
        if annotations is not None:
            annotations = np.zeros([min_images, min_rows, min_cols], dtype=y.dtype)
            annotations[0:z_stop - z_start, :r_stop - r_start, :c_stop - c_start] = y[z_start:z_stop, r_start:r_stop,
                                                                                    c_start:c_stop]
            if self.return_mask:
                return [out_images, np.sum(annotations[..., 1:], axis=-1)[..., None]], annotations
        if self.mask_output:
            out_images[annotations == 0] = np.min(out_images)
        return out_images, annotations


class Turn_Two_Class_Three(Image_Processor):
    def post_process(self, images, pred, ground_truth=None):
        i_size = pred.shape[1]
        new_output = np.zeros([pred.shape[0], pred.shape[1], pred.shape[2], 3], dtype=pred.dtype)
        new_output[..., 0] = pred[..., 0]
        new_output[:, :, :i_size // 2, 1] = pred[:, :, :i_size // 2, 1]
        new_output[:, :, i_size // 2:, 2] = pred[:, :, i_size // 2:, 1]
        return images, new_output, ground_truth


class Expand_Dimension(Image_Processor):
    def __init__(self, axis=0):
        self.axis = axis

    def pre_process(self, images, annotations=None):
        images = np.expand_dims(images, axis=self.axis)
        if annotations is not None:
            annotations = np.expand_dims(annotations, axis=self.axis)
        return images, annotations


class Check_Size(Image_Processor):
    def __init__(self, image_size=512):
        self.image_size = image_size

    def pre_process(self, images, annotations=None):
        self.og_image_size = images.shape
        self.dif_r = images.shape[1] - self.image_size
        self.dif_c = images.shape[2] - self.image_size
        if self.dif_r == 0 and self.dif_c == 0:
            return images
        self.start_r = self.dif_r // 2
        self.start_c = self.dif_c // 2
        if self.start_r > 0:
            images = images[:, self.start_r:self.start_r + self.image_size, ...]
        if self.start_c > 0:
            images = images[:, :, self.start_c:self.start_c + self.image_size, ...]
        if self.start_r < 0 or self.start_c < 0:
            output_images = np.ones(images.shape, dtype=images.dtype) * np.min(images)
            output_images[:, abs(self.start_r):abs(self.start_r) + images.shape[1],
            abs(self.start_c):abs(self.start_c) + images.shape[2], ...] = images
        else:
            output_images = images
        return output_images, annotations

    def post_process(self, images, pred, ground_truth=None):
        out_pred = np.zeros([self.og_image_size[0], self.og_image_size[1], self.og_image_size[2], pred.shape[-1]])
        out_pred[:, self.start_r:pred.shape[1] + self.start_r, self.start_c:pred.shape[2] + self.start_c, ...] = pred
        return images, out_pred, ground_truth


class VGG_Normalize(Image_Processor):
    def pre_process(self, images, annotations=None):
        images[..., 0] -= 123.68
        images[..., 1] -= 116.78
        images[..., 2] -= 103.94
        return images, annotations


class Repeat_Channel(Image_Processor):
    def __init__(self, num_repeats=3, axis=-1):
        self.num_repeats = num_repeats
        self.axis = axis

    def pre_process(self, images, annotations=None):
        images = np.repeat(images, self.num_repeats, axis=self.axis)
        return images, annotations


class True_Threshold_Prediction(Image_Processor):
    def __init__(self, threshold=0.5, pred_axis=[1]):
        '''
        :param threshold:
        '''
        self.threshold = threshold
        self.pred_axis = pred_axis

    def post_process(self, images, pred, ground_truth=None):
        for axis in self.pred_axis:
            temp_pred = pred[..., axis]
            temp_pred[temp_pred > self.threshold] = 1
            temp_pred[temp_pred < 1] = 0
            pred[..., axis] = temp_pred
        return images, pred, ground_truth


class ArgMax_Pred(Image_Processor):
    def post_process(self, images, pred, ground_truth=None):
        out_classes = pred.shape[-1]
        pred = np.argmax(pred, axis=-1)
        pred = to_categorical(pred, out_classes)
        return images, pred, ground_truth


class Threshold_Prediction(Image_Processor):
    def __init__(self, threshold=0.0, single_structure=True, is_liver=False, min_volume=0.0):
        '''
        :param threshold:
        :param single_structure:
        :param is_liver:
        :param min_volume: in ccs
        '''
        self.threshold = threshold
        self.is_liver = is_liver
        self.min_volume = min_volume
        self.single_structure = single_structure

    def post_process(self, images, pred, ground_truth=None):
        if self.is_liver:
            pred[..., -1] = variable_remove_non_liver(pred[..., -1], threshold=0.2, is_liver=True)
        if self.threshold != 0.0:
            for i in range(1, pred.shape[-1]):
                pred[..., i] = remove_non_liver(pred[..., i], threshold=self.threshold, do_3D=self.single_structure,
                                                min_volume=self.min_volume)
        return images, pred, ground_truth


class Rename_Lung_Voxels(Iterate_Overlap):
    def post_process(self, images, pred, ground_truth=None):
        mask = np.sum(pred[..., 1:], axis=-1)
        pred = self.iterate_annotations(pred, mask, spacing=list(self.dicom_handle.GetSpacing()), z_mult=1)
        return images, pred, ground_truth


class Rename_Lung_Voxels_Ground_Glass(Iterate_Overlap):
    def post_process(self, images, pred, ground_truth=None):
        mask = np.sum(pred[..., 1:], axis=-1)
        lungs = np.stack([mask, mask], axis=-1)
        lungs = self.iterate_annotations(lungs, mask, spacing=list(self.dicom_handle.GetSpacing()), z_mult=1)
        lungs = lungs[..., 1]
        pred[lungs == 0] = 0
        pred[..., 2] = lungs # Just put lungs in as entirety
        return images, pred, ground_truth


class Normalize_JPG_HU(Image_Processor):
    def __init__(self, is_jpg=False):
        self.is_jpg = is_jpg

    def normalize_function(self, img, fat=-109):
        air = np.min(img)
        air_HU = -1000
        fat_HU = -100

        delta_air_fat_HU = abs(air_HU - fat_HU)
        delta_fat_air_rgb = abs(fat - air)
        ratio = delta_air_fat_HU / delta_fat_air_rgb

        img = img - air
        img = img * ratio
        img = img + air_HU
        return img

    def pre_process(self, images, annotations=None):
        images = self.normalize_function(images)
        return images, annotations


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if height is not None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    if width is not None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        if dim is None:
            dim = (width, int(h * r))
        else:
            dim = min([dim, (width, int(h * r))])

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


class Ensure_Image_Proportions(Image_Processor):
    def __init__(self, image_rows=512, image_cols=512):
        self.wanted_rows = image_rows
        self.wanted_cols = image_cols

    def pre_process(self, images, annotations=None):
        og_image_size = np.squeeze(images.shape)
        if len(og_image_size) == 4:
            self.og_rows, self.og_cols = og_image_size[-3], og_image_size[-2]
        else:
            self.og_rows, self.og_cols = og_image_size[-2], og_image_size[-1]
        self.resize = False
        self.pad = False
        if self.og_rows != self.wanted_rows or self.og_cols != self.wanted_cols:
            self.resize = True
            images = [image_resize(i, self.wanted_rows, self.wanted_cols, inter=cv2.INTER_LINEAR)[None, ...] for i in
                      images]
            images = np.concatenate(images, axis=0)
            print('Resizing {} to {}'.format(self.og_rows, images.shape[1]))
            if annotations is not None:
                annotations = [image_resize(i, self.wanted_rows, self.wanted_cols, inter=cv2.INTER_LINEAR)[None, ...]
                               for i in annotations.astype('float32')]
                annotations = np.concatenate(annotations, axis=0).astype('int')
            self.pre_pad_rows, self.pre_pad_cols = images.shape[1], images.shape[2]
            if self.wanted_rows != self.pre_pad_rows or self.wanted_cols != self.pre_pad_cols:
                print('Padding {} to {}'.format(self.pre_pad_rows, self.wanted_rows))
                self.pad = True
                images = [np.resize(i, new_shape=(self.wanted_rows, self.wanted_cols, images.shape[-1]))[None, ...] for
                          i in images]
                images = np.concatenate(images, axis=0)
                if annotations is not None:
                    annotations = [np.resize(i, new_shape=(self.wanted_rows, self.wanted_cols, annotations.shape[-1]))
                                   for i in
                                   annotations]
                    annotations = np.concatenate(annotations, axis=0)
        return images, annotations

    def post_process(self, images, pred, ground_truth=None):
        if not self.pad and not self.resize:
            return images, pred, ground_truth
        if self.pad:
            pred = [np.resize(i, new_shape=(self.pre_pad_rows, self.pre_pad_cols, pred.shape[-1])) for i in pred]
            pred = np.concatenate(pred, axis=0)

            images = [np.resize(i, new_shape=(self.pre_pad_rows, self.pre_pad_cols, images.shape[-1])) for i in images]
            images = np.concatenate(images, axis=0)

            if ground_truth is not None:
                ground_truth = [np.resize(i, new_shape=(self.pre_pad_rows, self.pre_pad_cols, ground_truth.shape[-1]))
                                for i in
                                ground_truth]
                ground_truth = np.concatenate(ground_truth, axis=0)

        if self.resize:
            pred = [image_resize(i, self.og_rows, self.og_cols, inter=cv2.INTER_LINEAR)[None, ...] for i in pred]
            pred = np.concatenate(pred, axis=0)

            images = [image_resize(i, self.og_rows, self.og_cols, inter=cv2.INTER_LINEAR)[None, ...] for i in images]
            images = np.concatenate(images, axis=0)
            if ground_truth is not None:
                ground_truth = [image_resize(i, self.og_rows, self.og_cols, inter=cv2.INTER_LINEAR)[None, ...] for i in
                                ground_truth.astype('float32')]
                ground_truth = np.concatenate(ground_truth, axis=0).astype('int')
        return images, pred, ground_truth


class Threshold_Images(Image_Processor):
    def __init__(self, lower_bound=-np.inf, upper_bound=np.inf, inverse_image=False, final_scale_value=None,
                 divide=False):
        '''
        :param lower_bound: Lower bound to threshold images, normally -3.55 if Normalize_Images is used previously
        :param upper_bound: Upper bound to threshold images, normally 3.55 if Normalize_Images is used previously
        :param inverse_image: Should the image be inversed after threshold?
        '''
        self.lower = lower_bound
        self.divide = divide
        self.upper = upper_bound
        self.inverse_image = inverse_image
        self.final_scale_value = final_scale_value

    def pre_process(self, images, annotations=None):
        images[images < self.lower] = self.lower
        images[images > self.upper] = self.upper
        if self.final_scale_value is not None:
            images = (images - self.lower) / (self.upper - self.lower) * self.final_scale_value
        if self.inverse_image:
            if self.upper != np.inf and self.lower != -np.inf:
                images = (self.upper + self.lower) - images
            else:
                images = -1 * images
        if self.divide:
            images /= (self.upper - self.lower)
        return images, annotations


class MultiplyImagesByConstant(Image_Processor):
    def __init__(self, multiply_value=255.):
        '''
        :param multiply_value: Value to multiply array by
        '''
        self.multiply_value = multiply_value

    def pre_process(self, images, annotations=None):
        images *= self.multiply_value
        return images, annotations


class Normalize_Parotid_MR(Image_Processor):
    def pre_process(self, images, annotations=None):
        data = images.flatten()
        counts, bins = np.histogram(data, bins=1000)
        count_index = 0
        count_value = 0
        while count_value / np.sum(counts) < .3:  # Throw out the bottom 30 percent of data, as that is usually just 0s
            count_value += counts[count_index]
            count_index += 1
        min_bin = bins[count_index]
        data = data[data > min_bin]
        mean_val, std_val = np.mean(data), np.std(data)
        images = (images - mean_val) / std_val
        return images, annotations


class Normalize_Images(Image_Processor):
    def __init__(self, mean_val=0, std_val=1, upper_threshold=None, lower_threshold=None, max_val=1):
        self.mean_val, self.std_val = mean_val, std_val
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.max_val = max_val

    def pre_process(self, images, annotations=None):
        self.raw_images = copy.deepcopy(images)
        if self.upper_threshold is not None:
            images[images > self.upper_threshold] = self.upper_threshold
        if self.lower_threshold is not None:
            images[images < self.lower_threshold] = self.lower_threshold
        if np.max(images) == 255:
            images -= 80
            images /= 30
            return images, annotations
        if self.mean_val != 0 or self.std_val != 1:
            images = (images - self.mean_val) / self.std_val
            images[images > 3.55] = 3.55
            images[images < -3.55] = -3.55
            self.mean_min, self.mean_max = -3.55, 3.55
        else:
            images = (images - self.lower_threshold) / (self.upper_threshold - self.lower_threshold) * self.max_val
            self.mean_min, self.mean_max = self.lower_threshold, self.upper_threshold
        return images, annotations

    def post_process(self, images, pred, ground_truth=None):
        return self.raw_images, pred, ground_truth


class Ensure_Liver_Segmentation(template_dicom_reader):
    def __init__(self, associations=None, wanted_roi='Liver', liver_folder=None):
        super(Ensure_Liver_Segmentation, self).__init__(associations=associations)
        self.wanted_roi = wanted_roi
        self.liver_folder = liver_folder
        self.reader = DicomReaderWriter(associations=self.associations, Contour_Names=[self.wanted_roi])

    def check_ROIs_In_Checker(self):
        self.roi_name = None
        for roi in self.reader.rois_in_case:
            if roi.lower() == self.wanted_roi.lower():
                self.roi_name = roi.lower()
                return None
        for roi in self.reader.rois_in_case:
            if roi in self.associations:
                if self.associations[roi] == self.wanted_roi.lower():
                    self.roi_name = roi.lower()
                    break

    def process(self, dicom_folder):
        self.reader.__reset__()
        self.reader.walk_through_folders(dicom_folder)
        self.reader.get_images()
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
                    self.check_ROIs_In_Checker()
                    if self.roi_name:
                        print('Previous liver contour found at ' + liver_out_path + '\nCopying over')
                        shutil.copy(os.path.join(liver_out_path, file), os.path.join(dicom_folder, file))
                        break
        if self.roi_name is None:
            self.status = False
            print('No liver contour, passing to liver model')

    def pre_process(self):
        self.dicom_handle = self.reader.dicom_handle
        self.reader.get_mask()
        return sitk.GetArrayFromImage(self.dicom_handle), self.reader.mask

    def post_process(self, images, pred, ground_truth=None):
        return images, pred, ground_truth


class Resample_Process(Image_Processor):
    def __init__(self, desired_output_dim=(None, None, 1.0)):
        self.desired_output_dim = desired_output_dim
        self.resampler = Resample_Class_Object()

    def pre_process(self, images, annotations=None):
        self.og_annotations = annotations
        self.desired_spacing = []
        self.resampled_images = False
        for i in range(3):
            if self.desired_output_dim[i] is None:
                self.desired_spacing.append(self.dicom_handle.GetSpacing()[i])
            else:
                if self.desired_output_dim[i] != self.dicom_handle.GetSpacing()[i]:
                    self.desired_spacing.append(self.desired_output_dim[i])
                    self.resampled_images = True
        if self.resampled_images:
            image_handle = sitk.GetImageFromArray(np.squeeze(images))
            image_handle.SetSpacing(self.dicom_handle.GetSpacing())
            self.image_handle = image_handle
            image_handle = self.resampler.resample_image(image_handle, output_spacing=self.desired_spacing)
            images = sitk.GetArrayFromImage(image_handle)
            if annotations is not None:
                temp_annotation = sitk.GetImageFromArray(np.squeeze(annotations).astype('int'))
                temp_annotation.SetSpacing(self.dicom_handle.GetSpacing())
                temp_annotation = self.resampler.resample_image(temp_annotation,
                                                                input_spacing=self.dicom_handle.GetSpacing(),
                                                                output_spacing=self.desired_spacing)
                temp_annotation = sitk.GetArrayFromImage(temp_annotation)
                temp_annotation[temp_annotation > 0] = 1
                annotations = temp_annotation.astype('int')
            if len(self.image_handle.GetSize()) > 3:
                images, annotations = images[None, ...], annotations[None, ...]
        return images, annotations

    def post_process(self, images, pred, ground_truth=None):
        if not self.resampled_images:
            return images, pred, ground_truth
        else:
            images = np.squeeze(images)
            image_handle = sitk.GetImageFromArray(images)
            image_handle.SetSpacing(self.desired_spacing)
            image_handle = self.resampler.resample_image(image_handle, ref_handle=self.image_handle)
            images = sitk.GetArrayFromImage(image_handle)

            pred = np.squeeze(pred)
            pred_out = []
            for class_num in range(1, pred.shape[-1]):
                pred_handle = sitk.GetImageFromArray(pred[..., class_num])
                pred_handle.SetSpacing(self.desired_spacing)
                pred_handle = self.resampler.resample_image(pred_handle, ref_handle=self.image_handle)
                pred_out.append(sitk.GetArrayFromImage(pred_handle)[..., None])
            pred_out = [np.zeros(pred_out[0].shape)] + pred_out  # Have to add in a background
            pred = np.concatenate(pred_out, axis=-1)
            if ground_truth is not None:
                ground_truth = self.og_annotations
        return images, pred, ground_truth


class Ensure_Liver_Disease_Segmentation(template_dicom_reader):
    def __init__(self, associations=None, wanted_roi='Liver', liver_folder=None):
        super(Ensure_Liver_Disease_Segmentation, self).__init__(associations=associations)
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

    def process(self, dicom_folder):
        self.reader.__reset__()
        self.reader.walk_through_folders(dicom_folder)
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
                        shutil.copy(os.path.join(liver_out_path, file), os.path.join(dicom_folder, file))
                        break
        if self.roi_name is None:
            self.status = False
            print('No liver contour found')
        if self.roi_name:
            self.reader.get_images()

    def pre_process(self):
        self.dicom_handle = self.reader.dicom_handle
        self.reader.get_mask()
        return sitk.GetArrayFromImage(self.dicom_handle), self.reader.mask

    def post_process(self, images, pred, ground_truth=None):
        return images, pred, ground_truth


def main():
    pass


if __name__ == '__main__':
    main()
