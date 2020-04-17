import copy, shutil, os, time
from tensorflow.python.keras.utils.np_utils import to_categorical
from Resample_Class.Resample_Class import Resample_Class_Object, sitk
from Utils import np, get_bounding_box_indexes, remove_non_liver, plot_scroll_Image, variable_remove_non_liver
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack
from Fill_Missing_Segments.Fill_In_Segments_sitk import Fill_Missing_Segments
from skimage import morphology


class template_dicom_reader(object):
    def __init__(self, template_dir, channels=3, get_images_mask=True, associations={'Liver_BMA_Program_4':'Liver','Liver':'Liver'}):
        self.status = True
        self.reader = Dicom_to_Imagestack(template_dir=template_dir, channels=channels,
                                          get_images_mask=get_images_mask, associations=associations)

    def define_channels(self, channels):
        self.reader.channels = channels

    def define_threshold(self, threshold):
        self.reader.threshold = threshold

    def process(self, dicom_folder):
        self.reader.make_array(dicom_folder)
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
        self.spacing = niftii_handle.GetSpacing()

    def pre_process(self, images, annotations=None):
        return images, annotations

    def post_process(self, images, pred, ground_truth=None):
        return images, pred, ground_truth


class Iterate_Lobe_Annotations(Image_Processor):
    def __init__(self):
        MauererDistanceMap = sitk.SignedMaurerDistanceMapImageFilter()
        MauererDistanceMap.SetInsideIsPositive(True)
        MauererDistanceMap.UseImageSpacingOn()
        MauererDistanceMap.SquaredDistanceOff()
        self.MauererDistanceMap = MauererDistanceMap
        self.Remove_Smallest_Structure = Remove_Smallest_Structures()
        self.Smooth_Annotation = SmoothingPredictionRecursiveGaussian()

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

    def iterate_annotations(self, annotations_out, ground_truth_out, spacing, allowed_differences=50, max_iteration=15, z_mult=1):
        '''
        :param annotations:
        :param ground_truth:
        :param spacing:
        :param allowed_differences:
        :param max_iteration:
        :param z_mult: factor by which to ensure slices don't bleed into ones above and below
        :return:
        '''
        self.Remove_Smallest_Structure.spacing = self.spacing
        self.Smooth_Annotation.spacing = self.spacing
        annotations_out[ground_truth_out == 0] = 0
        min_z, max_z, min_r, max_r, min_c, max_c = get_bounding_box_indexes(ground_truth_out)
        annotations = annotations_out[min_z:max_z,min_r:max_r,min_c:max_c,...]
        ground_truth = ground_truth_out[min_z:max_z,min_r:max_r,min_c:max_c,...]
        spacing[-1] *= z_mult
        differences = [np.inf]
        index = 0
        while differences[-1] > allowed_differences and index < max_iteration:
            index += 1
            print('Iterating {}'.format(index))
            previous_iteration = copy.deepcopy(np.argmax(annotations,axis=-1))
            annotations = self.remove_56_78(annotations)
            for i in range(1, annotations.shape[-1]):
                annotation_handle = sitk.GetImageFromArray(annotations[...,i])
                annotation_handle.SetSpacing(self.spacing)
                pruned_handle = self.Remove_Smallest_Structure.remove_smallest_component(annotation_handle)
                annotations[..., i] = sitk.GetArrayFromImage(pruned_handle)
                slices = np.where(annotations[...,i] == 1)
                if slices:
                    slices = np.unique(slices[0])
                    for ii in range(len(slices)):
                        image_handle = sitk.GetImageFromArray(annotations[slices[ii],...,i][None,...])
                        pruned_handle = self.Remove_Smallest_Structure.remove_smallest_component(image_handle)
                        annotations[slices[ii], ..., i] = sitk.GetArrayFromImage(pruned_handle)

            annotations = self.make_distance_map(annotations, ground_truth,spacing=spacing)
            differences.append(np.abs(np.sum(previous_iteration[ground_truth==1]-np.argmax(annotations,axis=-1)[ground_truth==1])))
        annotations_out[min_z:max_z,min_r:max_r,min_c:max_c,...] = annotations
        annotations_out[ground_truth_out == 0] = 0
        return annotations_out

    def run_distance_map(self, array, spacing):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing(spacing)
        output = self.MauererDistanceMap.Execute(image)
        output = sitk.GetArrayFromImage(output)
        return output

    def make_distance_map(self, pred, liver, reduce=True, spacing=(0.975,0.975,2.5)):
        '''
        :param pred: A mask of your predictions with N channels on the end, N=0 is background [# Images, 512, 512, N]
        :param liver: A mask of the desired region [# Images, 512, 512]
        :param reduce: Save time and only work on masked region
        :return:
        '''
        liver = np.squeeze(liver)
        pred = np.squeeze(pred)
        pred = np.round(pred).astype('int')
        min_z, min_r, max_r, min_c, max_c = 0, 0, 512, 0, 512
        max_z = pred.shape[0]
        if reduce:
            min_z, max_z, min_r, max_r, min_c, max_c = get_bounding_box_indexes(liver)
        reduced_pred = pred[min_z:max_z,min_r:max_r,min_c:max_c]
        reduced_liver = liver[min_z:max_z,min_r:max_r,min_c:max_c]
        reduced_output = np.zeros(reduced_pred.shape)
        for i in range(1,pred.shape[-1]):
            temp_reduce = reduced_pred[...,i]
            output = self.run_distance_map(temp_reduce, spacing)
            reduced_output[...,i] = output
        reduced_output[reduced_output>0] = 0
        reduced_output = np.abs(reduced_output)
        reduced_output[...,0] = np.inf
        output = np.zeros(reduced_output.shape,dtype='int')
        mask = reduced_liver == 1
        values = reduced_output[mask]
        output[mask,np.argmin(values,axis=-1)] = 1
        pred[min_z:max_z,min_r:max_r,min_c:max_c] = output
        return pred

    def post_process(self, images, pred, ground_truth=None):
        pred = self.iterate_annotations(pred, ground_truth, spacing=list(self.spacing), z_mult=1, max_iteration=10)
        return images, pred, ground_truth


class Remove_Smallest_Structures(Image_Processor):
    def __init__(self):
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        self.RelabelComponent.SortByObjectSizeOn()

    def remove_smallest_component(self, annotation_handle):
        label_image = self.Connected_Component_Filter.Execute(
            sitk.BinaryThreshold(sitk.Cast(annotation_handle,sitk.sitkFloat32), lowerThreshold=0.01,
                                 upperThreshold=np.inf))
        label_image = self.RelabelComponent.Execute(label_image)
        output = sitk.BinaryThreshold(sitk.Cast(label_image,sitk.sitkFloat32), lowerThreshold=0.1,upperThreshold=1.0)
        return output


class Threshold_and_Expand(Image_Processor):
    def __init__(self, seed_threshold_value=0.8, lower_threshold_value=0.2):
        self.threshold_value = seed_threshold_value
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        self.Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        self.Connected_Threshold.SetLower(lower_threshold_value)
        self.Connected_Threshold.SetUpper(2)
        self.stats = sitk.LabelShapeStatisticsImageFilter()

    def post_process(self, images, pred, ground_truth=None):
        for i in range(1,pred.shape[-1]):
            prediction = sitk.GetImageFromArray(pred[...,i])
            thresholded_image = sitk.BinaryThreshold(prediction,lowerThreshold=self.threshold_value)
            connected_image = self.Connected_Component_Filter.Execute(thresholded_image)
            self.stats.Execute(connected_image)
            seeds = [self.stats.GetCentroid(l) for l in self.stats.GetLabels()]
            seeds = [prediction.TransformPhysicalPointToIndex(i) for i in seeds]
            self.Connected_Threshold.SetSeedList(seeds)
            output = self.Connected_Threshold.Execute(prediction)
            pred[...,i] = sitk.GetArrayFromImage(output)
        return images, pred, ground_truth


class Fill_Binary_Holes(Image_Processor):
    def __init__(self, pred_axis=[1]):
        self.pred_axis = pred_axis
        self.BinaryfillFilter = sitk.BinaryFillholeImageFilter()
        self.BinaryfillFilter.SetFullyConnected(True)
        self.BinaryfillFilter = sitk.BinaryMorphologicalClosingImageFilter()
        self.BinaryfillFilter.SetKernelRadius((5,5,1))
        self.BinaryfillFilter.SetKernelType(sitk.sitkBall)

    def post_process(self, images, pred, ground_truth=None):
        for axis in self.pred_axis:
            temp_pred = pred[...,axis]
            k = sitk.GetImageFromArray(temp_pred.astype('int'))
            k.SetSpacing(self.spacing)
            output = self.BinaryfillFilter.Execute(k)
            # temp_pred_image = sitk.BinaryThreshold(sitk.GetImageFromArray(temp_pred.astype('float32')),lowerThreshold=0.01,upperThreshold=np.inf)
            # output_array = np.zeros(temp_pred.shape)
            # for slice_index in range(temp_pred.shape[0]):
            #     filled = self.BinaryfillFilter.Execute(temp_pred_image[:, :, slice_index])
            #     output_array[slice_index] = sitk.GetArrayFromImage(filled)
            output_array = sitk.GetArrayFromImage(output)
            pred[...,axis] = output_array
        return images, pred, ground_truth


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
        self.min_volume = min_volume * 1000 # cm3 to mm3
        self.min_area = min_area * 100
        self.max_area = max_area * 100
        self.pred_axis = pred_axis
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()

    def post_process(self, images, pred, ground_truth=None):
        for axis in self.pred_axis:
            temp_pred = pred[...,axis]
            if self.min_volume != 0:
                label_image = self.Connected_Component_Filter.Execute(
                    sitk.BinaryThreshold(sitk.GetImageFromArray(temp_pred.astype('float32')), lowerThreshold=0.01, upperThreshold=np.inf))
                self.RelabelComponent.SetMinimumObjectSize(int(self.min_volume/np.prod(self.spacing)))
                label_image = self.RelabelComponent.Execute(label_image)
                temp_pred = sitk.GetArrayFromImage(label_image)
                temp_pred[temp_pred>0] = 1
                temp_pred[temp_pred<1] = 0
            if self.min_area != 0 or self.max_area != np.inf:
                slice_indexes = np.where(np.sum(temp_pred, axis=(1, 2)) > 0)
                if slice_indexes:
                    slice_spacing = np.prod(self.spacing[:-1])
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
                label_image = self.Connected_Component_Filter.Execute(
                    sitk.BinaryThreshold(sitk.GetImageFromArray(temp_pred.astype('float32')), lowerThreshold=0.01, upperThreshold=np.inf))
                self.RelabelComponent.SetMinimumObjectSize(int(self.min_volume/np.prod(self.spacing)))
                label_image = self.RelabelComponent.Execute(label_image)
                temp_pred = sitk.GetArrayFromImage(label_image)
                temp_pred[temp_pred>0] = 1
                temp_pred[temp_pred<1] = 0
            pred[...,axis] = temp_pred
        return images, pred, ground_truth


class SmoothingPredictionRecursiveGaussian(Image_Processor):
    def __init__(self, sigma=(0.1,0.1,0.0001), pred_axis=[1]):
        self.sigma = sigma
        self.pred_axis = pred_axis

    def smooth(self, handle):
        return sitk.BinaryThreshold(sitk.SmoothingRecursiveGaussian(handle), lowerThreshold=.01, upperThreshold=np.inf)

    def post_process(self, images, pred, ground_truth=None):
        for axis in self.pred_axis:
            k = sitk.GetImageFromArray(pred[...,axis])
            k.SetSpacing(self.spacing)
            k = self.smooth(k)
            pred[...,axis] = sitk.GetArrayFromImage(k)
        return images, pred, ground_truth


class To_Categorical(Image_Processor):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pre_process(self, images, annotations=None):
        annotations = to_categorical(annotations,self.num_classes)
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
        data = data[int(len(data)*self.lower_fraction):int(len(data)*self.upper_fraction)]
        mean_val = np.mean(data)
        std_val = np.std(data)
        images = (images - mean_val)/std_val
        return images, annotations


class Normalize_to_Liver(Image_Processor):
    def __init__(self):
        '''
        This is a little tricky... We only want to perform this task once, since it requires potentially large
        computation time, but it also requires that all individual image slices already be loaded
        '''
        # Now performing FWHM


    def pre_process(self, images, annotations=None):
        data = images[annotations == 1].flatten()
        counts, bins = np.histogram(data, bins=1000)
        bins = bins[:-1]
        count_index = np.where(counts == np.max(counts))[0][-1]
        half_counts = counts - np.max(counts) // 2
        half_upper = np.abs(half_counts[count_index:])
        max_50 = np.where(half_upper == np.min(half_upper))[0][0]

        half_lower = np.abs(half_counts[:count_index][-1::-1])
        min_50 = np.where(half_lower == np.min(half_lower))[0][0]

        min_values = bins[count_index - min_50]
        max_values = bins[count_index + max_50]
        data = data[np.where((data >= min_values) & (data <= max_values))]
        mean_val, std_val = np.mean(data), np.std(data)
        images = (images - mean_val)/std_val
        return images, annotations


class Mask_Prediction(Image_Processor):
    def __init__(self, num_repeats):
        self.num_repeats = num_repeats

    def pre_process(self, images, annotations=None):
        mask = annotations[...,None]
        mask = np.repeat(mask, self.num_repeats, axis=-1)
        sum_vals = np.zeros(mask.shape)
        sum_vals[..., 0] = 1 - mask[..., 0]
        return [images, mask, sum_vals], annotations


class remove_potential_ends_threshold(Image_Processor):
    def __init__(self, threshold=-1000):
        self.threshold = threshold

    def post_process(self, images, pred, ground_truth=None):
        indexes = np.where(pred==1)
        values = images[indexes]
        unique_values = np.unique(indexes[0])
        for i in unique_values:
            mean_val = np.mean(values[indexes[0]==i])
            if mean_val < self.threshold:
                pred[i] = 0
        return images, pred, ground_truth


class remove_potential_ends_size(Image_Processor):

    def post_process(self, images, pred, ground_truth=None):
        sum_slice = tuple(range(len(pred.shape))[1:])
        slices = np.where(sum_slice>0)
        if slices and len(slices[0]) > 10:
            reduced_slices = sum_slice[slices[0]]
            local_min = (np.diff(np.sign(np.diff(reduced_slices))) > 0).nonzero()[0] + 1  # local min
            local_max = (np.diff(np.sign(np.diff(reduced_slices))) < 0).nonzero()[0] + 1  # local max
            global_max = np.max(reduced_slices)
            total_slices = len(slices[0])//5 + 1
            for index in range(total_slices):
                if reduced_slices[index] > reduced_slices[index + 1]:
                    pred[reduced_slices[index]]
        indexes = np.where(pred==1)
        values = images[indexes]
        unique_values = np.unique(indexes[0])
        for i in unique_values:
            mean_val = np.mean(values[indexes[0]==i])
            if mean_val < self.threshold:
                pred[i] = 0
        return images, pred, ground_truth


class Make_3D(Image_Processor):
    def pre_process(self, images, annotations=None):
        return images[None,...], annotations

    def post_process(self, images, pred, ground_truth=None):
        return np.squeeze(images), np.squeeze(pred), ground_truth


class Reduce_Prediction(Image_Processor):
    def post_process(self, images, pred, ground_truth=None):
        pred[pred<0.5] = 0
        return images, pred, ground_truth


class Pad_Images(Image_Processor):
    def __init__(self, bounding_box_expansion=(10,10,10), power_val_z=1, power_val_x=1,
                 power_val_y=1):
        self.bounding_box_expansion = bounding_box_expansion
        self.power_val_z, self.power_val_x, self.power_val_y = power_val_z, power_val_x, power_val_y

    def pre_process(self, images, annotations=None):
        images_shape = images.shape
        z_start, r_start, c_start = 0, 0, 0
        z_stop, r_stop, c_stop = images_shape[0], images_shape[1], images_shape[2]
        z_total, r_total, c_total = z_stop - z_start, r_stop - r_start, c_stop - c_start
        remainder_z, remainder_r, remainder_c = self.power_val_z - z_total % self.power_val_z if z_total % self.power_val_z != 0 else 0, \
                                                self.power_val_x - r_total % self.power_val_x if r_total % self.power_val_x != 0 else 0, \
                                                self.power_val_y - c_total % self.power_val_y if c_total % self.power_val_y != 0 else 0
        min_images, min_rows, min_cols = z_total + remainder_z, r_total + remainder_r, c_total + remainder_c
        out_shape = (min_images, min_rows, min_cols)
        if len(images.shape) == 4:
            out_images = np.ones(out_shape + (images.shape[-1],))*np.min(images)
        else:
            out_images = np.ones(out_shape)*np.min(images)
        if len(annotations.shape) == 4:
            out_annotations = np.zeros(out_shape + (annotations.shape[-1],))
            out_annotations[..., 0] = 1
        else:
            out_annotations = np.zeros(out_shape)
        image_cube = images[z_start:z_stop,r_start:r_stop,c_start:c_stop,...]
        annotation_cube = annotations[z_start:z_stop,r_start:r_stop,c_start:c_stop,...]
        img_shape = image_cube.shape
        out_images[:img_shape[0],:img_shape[1],:img_shape[2],...] = image_cube
        out_annotations[:img_shape[0],:img_shape[1],:img_shape[2],...] = annotation_cube
        return out_images, out_annotations


class Image_Clipping_and_Padding(Image_Processor):
    def __init__(self, layers=3, return_mask=False, liver_box=False,  mask_output=False):
        self.mask_output = mask_output
        self.patient_dict = {}
        self.liver_box = liver_box
        power_val_z, power_val_x, power_val_y = (1,1,1)
        pool_base = 2
        for layer in range(layers):
            pooling = [pool_base for _ in range(3)]
            power_val_z *= pooling[0]
            power_val_x *= pooling[1]
            power_val_y *= pooling[2]
        self.return_mask = return_mask
        self.power_val_z, self.power_val_x, self.power_val_y = power_val_z, power_val_x, power_val_y

    def pre_process(self, images,annotations=None):
        x,y = images, annotations
        if self.liver_box and y is not None:
            liver = np.argmax(y,axis=-1)
            z_start, z_stop, r_start, r_stop, c_start, c_stop = get_bounding_box_indexes(liver)
            z_start = max([0,z_start-5])
            z_stop = min([z_stop+5,x.shape[1]])
            r_start = max([0,r_start-10])
            r_stop = min([512,r_stop+10])
            c_start = max([0,c_start-10])
            c_stop = min([512,c_stop+10])
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
        out_images = np.ones([min_images,min_rows,min_cols,x.shape[-1]],dtype=x.dtype)*np.min(x)
        out_images[0:z_stop-z_start,:r_stop-r_start,:c_stop-c_start,:] = x[z_start:z_stop,r_start:r_stop,c_start:c_stop,:]
        if annotations is not None:
            annotations = np.zeros([min_images,min_rows,min_cols], dtype=y.dtype)
            annotations[0:z_stop-z_start,:r_stop-r_start,:c_stop-c_start] = y[z_start:z_stop,r_start:r_stop,c_start:c_stop]
            if self.return_mask:
                return [out_images,np.sum(annotations[...,1:],axis=-1)[...,None]], annotations
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
        images, annotations = np.expand_dims(images,axis=self.axis), np.expand_dims(annotations,axis=self.axis)
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
        self.start_c = self.dif_c //2
        if self.start_r > 0:
            images = images[:,self.start_r:self.start_r+self.image_size,...]
        if self.start_c > 0:
            images = images[:,:,self.start_c:self.start_c + self.image_size,...]
        if self.start_r < 0 or self.start_c < 0:
            output_images = np.ones(images.shape, dtype=images.dtype) * np.min(images)
            output_images[:,abs(self.start_r):abs(self.start_r)+images.shape[1],abs(self.start_c):abs(self.start_c)+images.shape[2],...] = images
        else:
            output_images = images
        return output_images, annotations

    def post_process(self, images, pred, ground_truth=None):
        out_pred = np.zeros([self.og_image_size[0],self.og_image_size[1],self.og_image_size[2],pred.shape[-1]])
        out_pred[:,self.start_r:pred.shape[1] + self.start_r,self.start_c:pred.shape[2] + self.start_c,...] = pred
        return images, out_pred, ground_truth


class VGG_Normalize(Image_Processor):
    def pre_process(self, images, annotations=None):
        images[:, :, :, 0] -= 123.68
        images[:, :, :, 1] -= 116.78
        images[:, :, :, 2] -= 103.94
        return images, annotations


class Repeat_Channel(Image_Processor):
    def __init__(self, num_repeats=3, axis=-1):
        self.num_repeats = num_repeats
        self.axis = axis

    def pre_process(self, images, annotations=None):
        images = np.repeat(images,self.num_repeats,axis=self.axis)
        return images, annotations


class True_Threshold_Prediction(Image_Processor):
    def __init__(self, threshold=0.5, pred_axis = [1]):
        '''
        :param threshold:
        '''
        self.threshold = threshold
        self.pred_axis = pred_axis

    def post_process(self, images, pred, ground_truth=None):
        for axis in self.pred_axis:
            temp_pred = pred[...,axis]
            temp_pred[temp_pred > self.threshold] = 1
            temp_pred[temp_pred<1] = 0
            pred[...,axis] = temp_pred
        return images, pred, ground_truth


class ArgMax_Pred(Image_Processor):
    def post_process(self, images, pred, ground_truth=None):
        out_classes = pred.shape[-1]
        pred = np.argmax(pred,axis=-1)
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
            pred[...,-1] = variable_remove_non_liver(pred[...,-1], threshold=0.2, is_liver = True)
        if self.threshold != 0.0:
            for i in range(1,pred.shape[-1]):
                pred[...,i] = remove_non_liver(pred[...,i], threshold=self.threshold,do_3D=self.single_structure,
                                               min_volume=self.min_volume)
        return images, pred, ground_truth


class Threshold_Images(Image_Processor):
    def __init__(self, lower_bound=-np.inf, upper_bound=np.inf, inverse_image=False, post_load=True, final_scale_value=None):
        '''
        :param lower_bound: Lower bound to threshold images, normally -3.55 if Normalize_Images is used previously
        :param upper_bound: Upper bound to threshold images, normally 3.55 if Normalize_Images is used previously
        :param inverse_image: Should the image be inversed after threshold?
        :param post_load: should this be done each iteration? If False, gets slotted under pre_load_process
        '''
        self.lower = lower_bound
        self.upper = upper_bound
        self.inverse_image = inverse_image
        self.post_load = post_load
        self.final_scale_value = final_scale_value

    def pre_process(self, images, annotations=None):
        if self.post_load:
            images[images < self.lower] = self.lower
            images[images > self.upper] = self.upper
            if self.final_scale_value is not None:
                images = (images - self.lower) / (self.upper - self.lower) * self.final_scale_value
            if self.inverse_image:
                if self.upper != np.inf and self.lower != -np.inf:
                    images = (self.upper + self.lower) - images
                else:
                    images = -1*images
        return images, annotations


class Normalize_Images(Image_Processor):
    def __init__(self, mean_val=0, std_val=1, upper_threshold=None, lower_threshold=None, max_val=1):
        self.mean_val, self.std_val = mean_val, std_val
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.max_val = max_val

    def pre_process(self, images, annotation=None):
        self.raw_images = copy.deepcopy(images)
        if self.upper_threshold is not None:
            images[images > self.upper_threshold] = self.upper_threshold
        if self.lower_threshold is not None:
            images[images < self.lower_threshold] = self.lower_threshold
        if self.mean_val != 0 or self.std_val != 1:
            images = (images - self.mean_val) / self.std_val
            images[images>3.55] = 3.55
            images[images<-3.55] = -3.55
            self.mean_min, self.mean_max = -3.55, 3.55
        else:
            images = (images - self.lower_threshold) /(self.upper_threshold - self.lower_threshold) * self.max_val
            self.mean_min, self.mean_max = self.lower_threshold, self.upper_threshold
        return images, annotation

    def post_process(self, images, pred, ground_truth=None):
        return self.raw_images, pred, ground_truth


class Ensure_Liver_Segmentation(template_dicom_reader):
    def __init__(self, template_dir, channels=1, associations=None, wanted_roi='Liver', liver_folder=None):
        super(Ensure_Liver_Segmentation,self).__init__(template_dir=template_dir, channels=channels,
                                                       get_images_mask=False, associations=associations)
        self.associations = associations
        self.wanted_roi = wanted_roi
        self.liver_folder = liver_folder
        self.reader.set_contour_names([wanted_roi])
        self.reader.set_associations(associations)
        self.Resample = Resample_Class_Object()
        self.desired_output_dim = (None,None,5.)
        self.Fill_Missing_Segments_Class = Fill_Missing_Segments()
        self.rois_in_case = []

    def check_ROIs_In_Checker(self):
        self.roi_name = None
        for roi in self.reader.rois_in_case:
            if roi.lower() == self.wanted_roi.lower():
                self.roi_name = roi.lower()
                return None
        for roi in self.reader.rois_in_case:
            if roi in self.associations:
                if self.associations[roi] == self.wanted_roi:
                    self.roi_name = roi.lower()
                    break

    def process(self, dicom_folder):
        self.reader.make_array(dicom_folder)
        self.check_ROIs_In_Checker()
        go = False
        if self.roi_name is None and go:
            liver_input_path = os.path.join(self.liver_folder, self.reader.ds.PatientID,
                                            self.reader.ds.SeriesInstanceUID)
            liver_out_path = liver_input_path.replace('Input_3', 'Output')
            if os.path.exists(liver_out_path):
                files = [i for i in os.listdir(liver_out_path) if i.find('.dcm') != -1]
                for file in files:
                    self.reader.lstRSFile = os.path.join(liver_out_path,file)
                    self.reader.get_rois_from_RT()
                    self.check_ROIs_In_Checker()
                    if self.roi_name:
                        print('Previous liver contour found at ' + liver_out_path + '\nCopying over')
                        shutil.copy(os.path.join(liver_out_path, file), os.path.join(dicom_folder, file))
                        break
        if self.roi_name is None:
            self.status = False
            print('No liver contour, passing to liver model')
        if self.roi_name:
            self.reader.get_images_mask = True
            self.reader.make_array(dicom_folder)

    def pre_process(self):
        self.reader.get_mask()
        self.og_liver = copy.deepcopy(self.reader.mask)
        image_size = self.reader.ArrayDicom.shape
        self.true_output = np.zeros([image_size[0], image_size[1], image_size[2], 9])
        dicom_handle = self.reader.dicom_handle
        self.input_spacing = dicom_handle.GetSpacing()
        annotation_handle = self.reader.annotation_handle
        self.og_ground_truth = sitk.GetArrayFromImage(annotation_handle)
        self.output_spacing = []
        for i in range(3):
            if self.desired_output_dim[i] is None:
                self.output_spacing.append(self.input_spacing[i])
            else:
                self.output_spacing.append(self.desired_output_dim[i])
        resampled_dicom_handle = self.Resample.resample_image(dicom_handle, input_spacing=self.input_spacing,
                                                              output_spacing=self.output_spacing,is_annotation=False)
        self.resample_annotation_handle = self.Resample.resample_image(annotation_handle, input_spacing=self.input_spacing,
                                                           output_spacing=self.output_spacing, is_annotation=True)
        self.dicom_handle = self.reader.annotation_handle
        x = sitk.GetArrayFromImage(resampled_dicom_handle)
        y = sitk.GetArrayFromImage(self.resample_annotation_handle)
        self.z_start, self.z_stop, self.r_start, self.r_stop, self.c_start, self.c_stop = get_bounding_box_indexes(y)
        images = x[self.z_start:self.z_stop,self.r_start:self.r_stop,self.c_start:self.c_stop]
        y = y[self.z_start:self.z_stop,self.r_start:self.r_stop,self.c_start:self.c_stop]
        return images[...,None], y

    def post_process(self, images, pred, ground_truth=None):
        pred = np.argmax(pred,axis=-1)
        pred = to_categorical(pred, num_classes=9)

        # for i in range(1, pred.shape[-1]):
        #     pred[..., i] = remove_non_liver(pred[..., i], do_2D=True)
        pred = pred[0, ...]
        pred_handle = sitk.GetImageFromArray(pred)
        pred_handle.SetSpacing(self.resample_annotation_handle.GetSpacing())
        pred_handle.SetOrigin(self.resample_annotation_handle.GetOrigin())
        pred_handle.SetDirection(self.resample_annotation_handle.GetDirection())
        pred_handle_resampled = self.Resample.resample_image(pred_handle,input_spacing=self.output_spacing,
                                                             output_spacing=self.input_spacing,is_annotation=True)
        new_pred_og_size = sitk.GetArrayFromImage(pred_handle_resampled)

        ground_truth_handle = sitk.GetImageFromArray(np.squeeze(ground_truth))
        ground_truth_handle.SetSpacing(self.resample_annotation_handle.GetSpacing())
        ground_truth_handle.SetOrigin(self.resample_annotation_handle.GetOrigin())
        ground_truth_handle.SetDirection(self.resample_annotation_handle.GetDirection())

        ground_truth_resampled = self.Resample.resample_image(ground_truth_handle,input_spacing=self.output_spacing,
                                                              output_spacing=self.input_spacing,is_annotation=True)
        new_ground_truth_og_size = sitk.GetArrayFromImage(ground_truth_resampled)

        self.z_start_p, self.z_stop_p, self.r_start_p, self.r_stop_p, self.c_start_p, self.c_stop_p = \
            get_bounding_box_indexes(new_ground_truth_og_size)
        self.z_start, _, self.r_start, _, self.c_start, _ = get_bounding_box_indexes(sitk.GetArrayFromImage(self.reader.annotation_handle))
        z_stop = min([self.z_stop_p-self.z_start_p,self.true_output.shape[0]-self.z_start])
        self.true_output[self.z_start:self.z_start + z_stop,
        self.r_start:self.r_start + self.r_stop_p-self.r_start_p,
        self.c_start:self.c_start + self.c_stop_p - self.c_start_p,
        ...] = new_pred_og_size[self.z_start_p:self.z_start_p+z_stop, self.r_start_p:self.r_stop_p,self.c_start_p:self.c_stop_p, ...]
        # self.true_output = self.Fill_Missing_Segments_Class.iterate_annotations(self.true_output,self.og_ground_truth,
        #                                                                         spacing=spacing, z_mult=1, max_iteration=10)
        return images, self.true_output, self.og_ground_truth


class Ensure_Liver_Disease_Segmentation(template_dicom_reader):
    def __init__(self, template_dir, channels=1, associations=None, wanted_roi='Liver', liver_folder=None):
        super(Ensure_Liver_Disease_Segmentation,self).__init__(template_dir=template_dir, channels=channels,
                                                               get_images_mask=False, associations=associations)
        self.associations = associations
        self.wanted_roi = wanted_roi
        self.liver_folder = liver_folder
        self.reader.set_contour_names([wanted_roi])
        self.reader.set_associations(associations)
        self.Resample = Resample_Class_Object()
        self.desired_output_dim = (None, None, 1.0)
        self.rois_in_case = []

    def check_ROIs_In_Checker(self):
        self.roi_name = None
        for roi in self.reader.rois_in_case:
            if roi.lower() is self.wanted_roi.lower():
                self.roi_name = roi
                return None
        for roi in self.reader.rois_in_case:
            if roi in self.associations:
                if self.associations[roi] == self.wanted_roi:
                    self.roi_name = roi
                    break

    def process(self, dicom_folder):
        self.reader.make_array(dicom_folder)
        self.check_ROIs_In_Checker()
        go = False
        if self.roi_name is None and go:
            liver_input_path = os.path.join(self.liver_folder, self.reader.ds.PatientID,
                                            self.reader.ds.SeriesInstanceUID)
            liver_out_path = liver_input_path.replace('Input_3', 'Output')
            if os.path.exists(liver_out_path):
                files = [i for i in os.listdir(liver_out_path) if i.find('.dcm') != -1]
                for file in files:
                    self.reader.lstRSFile = os.path.join(liver_out_path,file)
                    self.reader.get_rois_from_RT()
                    self.check_ROIs_In_Checker()
                    if self.roi_name:
                        print('Previous liver contour found at ' + liver_out_path + '\nCopying over')
                        shutil.copy(os.path.join(liver_out_path, file), os.path.join(dicom_folder, file))
                        break
        if self.roi_name is None:
            self.status = False
            print('No liver contour, passing to liver model')
        if self.roi_name:
            self.reader.get_images_mask = True
            self.reader.make_array(dicom_folder)

    def pre_process(self):
        self.dicom_handle = self.reader.dicom_handle
        self.reader.get_mask()
        self.og_liver = copy.deepcopy(self.reader.mask)
        image_size = self.reader.ArrayDicom.shape
        self.true_output = np.zeros([image_size[0], image_size[1], image_size[2], 2])
        dicom_handle = self.reader.dicom_handle
        self.input_spacing = dicom_handle.GetSpacing()
        annotation_handle = self.reader.annotation_handle
        self.og_ground_truth = sitk.GetArrayFromImage(annotation_handle)
        self.output_spacing = []
        for i in range(3):
            if self.desired_output_dim[i] is None:
                self.output_spacing.append(self.input_spacing[i])
            else:
                self.output_spacing.append(self.desired_output_dim[i])
        print('Resampling from {} to {}'.format(self.input_spacing,self.output_spacing))
        resampled_dicom_handle = self.Resample.resample_image(dicom_handle, input_spacing=self.input_spacing,
                                                              output_spacing=self.output_spacing,is_annotation=False)
        self.resample_annotation_handle = self.Resample.resample_image(annotation_handle, input_spacing=self.input_spacing,
                                                                       output_spacing=self.output_spacing, is_annotation=True)
        x = sitk.GetArrayFromImage(resampled_dicom_handle)
        y = sitk.GetArrayFromImage(self.resample_annotation_handle)
        self.z_start, self.z_stop, self.r_start, self.r_stop, self.c_start, self.c_stop = get_bounding_box_indexes(y)
        images = x[self.z_start:self.z_stop,self.r_start:self.r_stop,self.c_start:self.c_stop]
        y = y[self.z_start:self.z_stop,self.r_start:self.r_stop,self.c_start:self.c_stop]
        return images[...,None], y

    def post_process(self, images, pred, ground_truth=None):
        pred = pred[0, ...]
        pred_handle = sitk.GetImageFromArray(pred)
        pred_handle.SetSpacing(self.resample_annotation_handle.GetSpacing())
        pred_handle.SetOrigin(self.resample_annotation_handle.GetOrigin())
        pred_handle.SetDirection(self.resample_annotation_handle.GetDirection())
        print('Resampling from {} to {}'.format(self.output_spacing, self.input_spacing))
        pred_handle_resampled = self.Resample.resample_image(pred_handle,input_spacing=self.output_spacing,
                                                             output_spacing=self.input_spacing,is_annotation=True)
        new_pred_og_size = sitk.GetArrayFromImage(pred_handle_resampled)
        ground_truth_handle = sitk.GetImageFromArray(np.squeeze(ground_truth))
        ground_truth_handle.SetSpacing(self.resample_annotation_handle.GetSpacing())
        ground_truth_handle.SetOrigin(self.resample_annotation_handle.GetOrigin())
        ground_truth_handle.SetDirection(self.resample_annotation_handle.GetDirection())

        ground_truth_resampled = self.Resample.resample_image(ground_truth_handle,input_spacing=self.output_spacing,
                                                              output_spacing=self.input_spacing,is_annotation=True)
        new_ground_truth_og_size = sitk.GetArrayFromImage(ground_truth_resampled)

        self.z_start_p, self.z_stop_p, self.r_start_p, self.r_stop_p, self.c_start_p, self.c_stop_p = \
            get_bounding_box_indexes(new_ground_truth_og_size)
        self.z_start, _, self.r_start, _, self.c_start, _ = get_bounding_box_indexes(sitk.GetArrayFromImage(self.reader.annotation_handle))
        z_stop = min([self.z_stop_p-self.z_start_p,self.true_output.shape[0]-self.z_start])
        self.true_output[self.z_start:self.z_start + z_stop,
        self.r_start:self.r_start + self.r_stop_p-self.r_start_p,
        self.c_start:self.c_start + self.c_stop_p - self.c_start_p,
        ...] = new_pred_og_size[self.z_start_p:self.z_start_p+z_stop, self.r_start_p:self.r_stop_p,self.c_start_p:self.c_stop_p, ...]
        self.true_output[self.og_ground_truth==0] = 0
        # Make z direction spacing 10* higher, we don't want bleed through much
        spacing = list(self.input_spacing)
        print(spacing)
        return images, self.true_output, ground_truth


def main():
    pass

if __name__ == '__main__':
    main()
