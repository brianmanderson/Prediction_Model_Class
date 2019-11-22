import copy, shutil, os
from Resample_Class.Resample_Class import Resample_Class, sitk
from Utils import Dicom_to_Imagestack, np, get_bounding_box_indexes, Fill_Missing_Segments, Copy_Folders, \
    remove_non_liver, plot_scroll_Image


class template_dicom_reader(object):
    def __init__(self, template_dir, channels=3, get_images_mask=True, associations={'Liver_BMA_Program_4':'Liver','Liver':'Liver'}):
        self.status = True
        self.reader = Dicom_to_Imagestack(template_dir=template_dir, channels=channels,
                                          get_images_mask=get_images_mask, associations=associations)

    def define_channels(self, channels):
        self.reader.channels = channels

    def define_threshold(self, threshold):
        self.reader.threshold = threshold

    def process(self, dicom_folder, single_structure=True):
        self.reader.make_array(dicom_folder, single_structure=single_structure)

    def return_status(self):
        return self.status

    def pre_process(self):
        return self.reader.ArrayDicom, None

    def post_process(self, images, pred, ground_truth=None):
        return images, pred, ground_truth


class Image_Processor(object):

    def get_path(self, PathDicom):
        self.PathDicom = PathDicom

    def pre_process(self, images, annotations=None):
        return images, annotations

    def post_process(self, images, pred, ground_truth=None):
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
            out_images[annotations == 0] = -3.55
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
        self.reader.Contour_Names = [wanted_roi]
        self.Resample = Resample_Class()
        self.desired_output_dim = (1.,1.,5.)
        self.Fill_Missing_Segments_Class = Fill_Missing_Segments()
        self.rois_in_case = []

    def check_ROIs_In_Checker(self):
        self.roi_name = None
        for roi in self.reader.rois_in_case:
            if roi in self.associations:
                if self.associations[roi] == self.wanted_roi:
                    self.roi_name = roi
                    break

    def process(self, dicom_folder, single_structure=True):
        self.reader.make_array(dicom_folder, single_structure=single_structure)
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
                # for file in os.listdir(dicom_folder):
                #     os.remove(os.path.join(dicom_folder,file))
                # Copy_Folders(dicom_folder, liver_input_path)
        if self.roi_name:
            self.reader.get_images_mask = True
            self.reader.make_array(dicom_folder,single_structure=single_structure)

    def pre_process(self):
        self.reader.get_mask()
        self.og_liver = copy.deepcopy(self.reader.mask)
        self.true_output = np.zeros([self.reader.ArrayDicom.shape[0], 512, 512, 9])
        dicom_handle = self.reader.dicom_handle
        self.input_spacing = dicom_handle.GetSpacing()
        annotation_handle = self.reader.annotation_handle
        self.og_ground_truth = sitk.GetArrayFromImage(annotation_handle)
        resampled_dicom_handle = self.Resample.resample_image(dicom_handle, input_spacing=self.input_spacing,
                                                       output_spacing=self.desired_output_dim,is_annotation=False)
        self.resample_annotation_handle = self.Resample.resample_image(annotation_handle, input_spacing=self.input_spacing,
                                                           output_spacing=self.desired_output_dim, is_annotation=True)
        x = sitk.GetArrayFromImage(resampled_dicom_handle)
        y = sitk.GetArrayFromImage(self.resample_annotation_handle)
        self.z_start, self.z_stop, self.r_start, self.r_stop, self.c_start, self.c_stop = get_bounding_box_indexes(y)
        images = x[self.z_start:self.z_stop,self.r_start:self.r_stop,self.c_start:self.c_stop]
        y = y[self.z_start:self.z_stop,self.r_start:self.r_stop,self.c_start:self.c_stop]
        return images[...,None], y


    def post_process(self, images, pred, ground_truth=None):
        pred[ground_truth == 0] = 0
        for i in range(1, pred.shape[-1]):
            pred[..., i] = remove_non_liver(pred[..., i], do_2D=True)
        pred = pred[0, ...]
        pred_handle = sitk.GetImageFromArray(pred)
        pred_handle.SetSpacing(self.resample_annotation_handle.GetSpacing())
        pred_handle.SetOrigin(self.resample_annotation_handle.GetOrigin())
        pred_handle.SetDirection(self.resample_annotation_handle.GetDirection())
        pred_handle_resampled = self.Resample.resample_image(pred_handle,input_spacing=self.desired_output_dim,
                                                             output_spacing=self.input_spacing,is_annotation=True)
        new_pred_og_size = sitk.GetArrayFromImage(pred_handle_resampled)

        self.z_start_p, self.z_stop_p, self.r_start_p, self.r_stop_p, self.c_start_p, self.c_stop_p = \
            get_bounding_box_indexes(np.sum(new_pred_og_size,axis=-1))
        self.z_start, _, self.r_start, _, self.c_start, _ = get_bounding_box_indexes(sitk.GetArrayFromImage(self.reader.annotation_handle))
        self.true_output[self.z_start:self.z_start + self.z_stop_p-self.z_start_p,
        self.r_start:self.r_start + self.r_stop_p-self.r_start_p,
        self.c_start:self.c_start + self.c_stop_p - self.c_start_p,
        ...] = new_pred_og_size[self.z_start_p:self.z_stop_p, self.r_start_p:self.r_stop_p,self.c_start_p:self.c_stop_p, ...]
        # Make z direction spacing 10* higher, we don't want bleed through much
        amounts = np.sum(self.true_output, axis=(1, 2))
        indexes = np.where((np.max(amounts[:, (5, 6)], axis=-1) > 0) & (np.max(amounts[:, (7, 8)], axis=-1) > 0))
        if indexes:
            indexes = indexes[0]
            for i in indexes:
                if amounts[i, 5] < amounts[i, 8]:
                    self.true_output[i, ..., 5] = 0
                else:
                    self.true_output[i, ..., 8] = 0
                if amounts[i, 6] < amounts[i, 7]:
                    self.true_output[i, ..., 6] = 0
                else:
                    self.true_output[i, ..., 7] = 0
        self.true_output = self.Fill_Missing_Segments_Class.make_distance_map(self.true_output, self.og_ground_truth,
                                                                              spacing=[self.input_spacing[0],
                                                                                       self.input_spacing[1],
                                                                                       100*self.input_spacing[2]])
        return images, self.true_output, ground_truth


def main():
    pass

if __name__ == '__main__':
    main()
