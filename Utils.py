import os
from math import floor, ceil
import SimpleITK as sitk
from typing import *
from PIL.features import features
from skimage import draw
import numpy as np
from Dicom_RT_and_Images_to_Mask.src.DicomRTTool import DicomReaderWriter
import tensorflow as tf
import time
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image


def dice_coef_3D(y_true, y_pred, smooth=0.0001):
    intersection = tf.keras.backend.sum(y_true[..., 1:] * y_pred[..., 1:])
    union = tf.keras.backend.sum(y_true[..., 1:]) + tf.keras.backend.sum(y_pred[..., 1:])
    return (2. * intersection + smooth) / (union + smooth)


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


class BaseModelBuilder(object):
    dicom_reader: TemplateDicomReader

    def __init__(self, image_key='image', model_path=None, Bilinear_model=None, loss=None, loss_weights=None):
        self.image_key = image_key
        self.model_path = model_path
        self.Bilinear_model = Bilinear_model
        self.loss = loss
        self.loss_weights = loss_weights
        self.paths = []
        self.image_processors = []
        self.prediction_processors = []
        self.out_path = ''
        self.input_features = {}

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
        self.model = tf.keras.models.load_model(self.model_path)
        self.model.trainable = False
        # self.model.load_weights(self.model_path, by_name=True, skip_mismatch=False)
        # avoid forbidden character from tf1.14 model (for ex: DeepLabV3+)
        # also allocate a scope per model name
        self.model._name = model_name

    def set_input_features(self, input_features):
        self.input_features = input_features
        if 'output_path' in self.input_features:
            self.set_out_path(self.input_features['output_path'])

    def run_load_images(self):
        self.load_images()
        self.define_image_out_path()
        self.write_status_file('Status_Preprocessing')
        time_flag = time.time()
        self.input_features = self.pre_process(self.input_features)
        print('Comp. time: pre_process {} seconds'.format(time.time() - time_flag))
        return True

    def run_prediction(self):
        self.write_status_file('Status_Predicting')
        time_flag = time.time()
        self.input_features = self.predict(self.input_features)
        print('Comp. time: predict {} seconds'.format(time.time() - time_flag))
        self.write_status_file('Status_Postprocessing')
        time_flag = time.time()
        self.input_features = self.post_process(self.input_features)
        print('Comp. time: post_process {} seconds'.format(time.time() - time_flag))
        self.write_status_file('Status_Writing RT Structure')
        self.write_predictions(self.input_features)
        self.write_status_file(None)

    def write_status_file(self, status: Optional[str] = None):
        for file in os.listdir(self.write_folder):
            if file.startswith('Status'):
                os.remove(os.path.join(self.write_folder, file))
        if status is not None:
            file_path = os.path.join(self.write_folder, status + '.txt')
            fid = open(file_path, 'w+')
            fid.close()

    def define_image_out_path(self):
        series_instances_dictionary = self.return_series_instance_dictionary()
        series_instance_uid = series_instances_dictionary['SeriesInstanceUID']
        patientID = series_instances_dictionary['PatientID']
        true_outpath = os.path.join(self.out_path, series_instance_uid)
        self.write_folder = true_outpath
        if not os.path.exists(true_outpath):
            os.makedirs(true_outpath)

    def set_out_path(self, out_path):
        self.out_path = out_path

    def load_images(self):
        self.input_features = self.dicom_reader.load_images(input_features=self.input_features)

    def return_series_instance_dictionary(self):
        return self.dicom_reader.reader.series_instances_dictionary[0]

    def return_status(self):
        return self.dicom_reader.return_status()

    def pre_process(self, input_features):
        for processor in self.image_processors:
            print('Performing pre process {}'.format(processor))
            processor.pre_process(input_features=input_features)
        return input_features

    def post_process(self, input_features):
        for processor in self.image_processors[::-1]:  # In reverse order now
            print('Performing post process {}'.format(processor))
            processor.post_process(input_features=input_features)
        return input_features

    def prediction_process(self, input_features):
        for processor in self.prediction_processors:
            print('Performing prediction process {}'.format(processor))
            processor.pre_process(input_features=input_features)
        return input_features

    def predict(self, input_features):
        input_features['prediction'] = self.model.predict(input_features[self.image_key])
        return input_features

    def write_predictions(self, input_features):
        self.dicom_reader.write_predictions(input_features=input_features)


class ModelBuilderFromTemplate(BaseModelBuilder):
    def __init__(self, image_key='image', model_path=None, model_template=None):
        super().__init__(image_key, model_path)
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


class PredictWindowSliding(ModelBuilderFromTemplate):
    def __init__(self, image_key='image', model_path=None, model_template=None, nb_label=13,
                 required_size=(32, 192, 192), sw_overlap=0.5, sw_batch_size=8, gaussiance_map=True, sigma_scale=0.125):
        super().__init__(image_key, model_path, model_template)
        self.nb_label = nb_label
        self.required_size = required_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size
        self.gaussiance_map = gaussiance_map
        self.sigma_scale = sigma_scale

    def predict(self, input_features):
        # This function follows on monai.inferers.SlidingWindowInferer implementations
        x = input_features['image']
        batch_size = 1
        image_size = x[0, ..., 0].shape
        scan_interval = _get_scan_interval(image_size, self.required_size, 3, self.sw_overlap)

        # Store all slices in list
        slices = dense_patch_slices(image_size, self.required_size, scan_interval)
        num_win = len(slices)  # number of windows per image
        total_slices = num_win * batch_size  # total number of windows

        # Create window-level importance map (can be changed to remove border effect for example)
        if not self.gaussiance_map:
            importance_map = np.ones(self.required_size + (self.nb_label,))
        else:
            GaussianSource = sitk.GaussianSource(size=self.required_size[::-1],
                                                 mean=tuple([x // 2 for x in self.required_size[::-1]]),
                                                 sigma=tuple([self.sigma_scale * x for x in self.required_size[::-1]]),
                                                 scale=1.0,
                                                 spacing=(1.0, 1.0, 1.0), normalized=False)
            importance_map = sitk.GetArrayFromImage(GaussianSource)
            importance_map = np.repeat(importance_map[..., None], repeats=self.nb_label, axis=-1)

        # Perform predictions
        # output_image, count_map = np.array([]), np.array([])
        _initialized = False
        for slice_g in range(0, total_slices, self.sw_batch_size):
            slice_range = range(slice_g, min(slice_g + self.sw_batch_size, total_slices))
            unravel_slice = [
                [slice(int(idx / num_win), int(idx / num_win) + 1)] + list(slices[idx % num_win]) + [slice(None)]
                for idx in slice_range
            ]
            window_data = np.concatenate([x[tuple(win_slice)] for win_slice in unravel_slice], axis=0)
            seg_prob = self.model.predict(window_data)

            if not _initialized:  # init. buffer at the first iteration
                output_shape = [batch_size] + list(image_size) + [self.nb_label]
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
        #                                               num_classes=self.nb_label)
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


def first(iterable, default=None):
    """
    Returns the first item in the given iterable or `default` if empty, meaningful mostly with 'for' expressions.
    """
    for i in iterable:
        return i
    return default


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


def return_paths():
    local_path = os.path.join('..', "Mounting", "Modular_Projects")
    if not os.path.exists(local_path):
        local_path = os.path.join('..')
    return local_path


def down_folder(input_path,output):
    files = []
    dirs = []
    for root, dirs, files in os.walk(input_path):
        break
    if 'Completed.txt' in files:
        output.append(input_path)
    for dir_val in dirs:
        output = down_folder(os.path.join(input_path,dir_val),output)
    return output


def cleanout_folder(path_origin, dicom_dir, delete_folders=True):
    files = []
    for _, _, files in os.walk(dicom_dir):
        break
    for file in files:
        os.remove(os.path.join(dicom_dir, file))
    while delete_folders and len(dicom_dir) > len(os.path.abspath(path_origin)):
        if len(os.listdir(dicom_dir)) == 0:
            os.rmdir(dicom_dir)
        dicom_dir = os.path.abspath(os.path.join(dicom_dir, '..'))
    return None


def poly2mask(vertex_row_coords, vertex_col_coords):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, [512,512])
    mask = np.zeros([512,512], dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def main():
    pass


if __name__ == "__main__":
    main()
