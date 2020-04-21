import os, shutil
import matplotlib.pyplot as plt
import pydicom
from pydicom.tag import Tag
import copy
from skimage import draw, morphology
from skimage.measure import label,regionprops,find_contours
import numpy as np
from scipy.ndimage import gaussian_filter
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
from tensorflow.compat.v1 import Graph, Session, ConfigProto, GPUOptions
from tensorflow.keras.backend import resize_images
from tensorflow.keras.layers import Input
import SimpleITK as sitk
from tensorflow.keras.models import load_model
from Fill_Missing_Segments.Fill_In_Segments_sitk import remove_non_liver


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

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

class Copy_Folders(object):
    def __init__(self, input_path, output_path):
        self.down_copy(input_path,output_path=output_path)

    def down_copy(self, input_path,output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        dirs = []
        files = []
        for _, dirs, files in os.walk(input_path):
            break
        for file in files:
            if file == 'Completed.txt':
                continue
            shutil.copy(os.path.join(input_path,file),os.path.join(output_path,file))
        if 'Completed.txt' in files:
            shutil.copy(os.path.join(input_path, 'Completed.txt'), os.path.join(output_path, 'Completed.txt'))
        for dir_val in dirs:
            new_out = os.path.join(output_path,dir_val)
            if not os.path.exists(new_out):
                os.makedirs(new_out)
            self.down_copy(os.path.join(input_path,dir_val),new_out)

    def down_folder(self, input_path,output=r'\\mymdafiles\di_data1\Morfeus\Andrea\Copy_Logs',base_path=r'G:\Cat'):
        dirs = []
        for _, dirs, _ in os.walk(input_path):
            break
        for dir_val in dirs:
            new_output = os.path.join(input_path.replace(base_path,output),dir_val)
            if not os.path.exists(new_output):
                os.makedirs(new_output)
            print(new_output)
            self.down_copy(os.path.join(input_path,dir_val), new_output)
        return None


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

def normalize_images(images,lower_threshold,upper_threshold,is_CT = True,max_val=255, mean_val=0,std_val=1):
    if is_CT:
        images[images > upper_threshold] = upper_threshold
        images[images < lower_threshold] = lower_threshold
        if mean_val != 0 or std_val != 1:
            images = (images - mean_val) / std_val
            images[images>3.55] = 3.55
            images[images<-3.55] = -3.55
            output = images
            # output = (images + 3.55)/(7.10)*255
        else:
            output = (images - lower_threshold) /(upper_threshold - lower_threshold) * max_val
    else:
        if len(images.shape) > 2:
            output = np.zeros(images.shape)
            iii = 0
            for i in images:
                i = (i - i[i > 100].mean()) / i[i > 100].std()
                i[i > 3] = 3
                i[i < -3] = -3
                min_val_local = i.min()
                max_val_local = i.max()
                i = (i-min_val_local)/(max_val_local - min_val_local) * max_val
                if len(output.shape) == 4:
                    output[iii,:,:,:] = i
                elif len(output.shape) == 3:
                    output[iii, :, :] = i
                else:
                    raise ('Image shape does not look right')
                iii += 1
        else:
            i = images
            i = (i - i[i > 100].mean()) / i[i > 100].std()
            i[i > 3] = 3
            i[i < -3] = -3
            min_val_local = i.min()
            max_val_local = i.max()
            i = (i - min_val_local) / (max_val_local - min_val_local) * max_val
            output = i
    return output

def plot_scroll_Image(x):
    '''
    :param x: input to view of form [rows, columns, # images]
    :return:
    '''
    if x.dtype not in ['float32','float64']:
        x = copy.deepcopy(x).astype('float32')
    if len(x.shape) > 3:
        x = np.squeeze(x)
    if len(x.shape) == 3:
        if x.shape[0] != x.shape[1]:
            x = np.transpose(x,[1,2,0])
        elif x.shape[0] == x.shape[2]:
            x = np.transpose(x, [1, 2, 0])
    fig, ax = plt.subplots(1, 1)
    if len(x.shape) == 2:
        x = np.expand_dims(x,axis=0)
    tracker = IndexTracker(ax, x)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    return fig,tracker
    #Image is input in the form of [#images,512,512,#channels]

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = np.where(self.X != 0)[-1]
        if len(self.ind) > 0:
            self.ind = self.ind[len(self.ind)//2]
        else:
            self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind],cmap='gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def dice_coef_3D(y_true, y_pred, smooth=0.0001):
    intersection = K.sum(y_true[...,1:] * y_pred[...,1:])
    union = K.sum(y_true[...,1:]) + K.sum(y_pred[...,1:])
    return (2. * intersection + smooth) / (union + smooth)

class VGG_Model_Pretrained(object):
    def __init__(self,model_path,gpu=0,graph1=Graph(), session1=Session(config=ConfigProto(gpu_options=GPUOptions(allow_growth=True),
                                                                                           log_device_placement=False)),
                 Bilinear_model=None,loss=None,loss_weights=None,**kwargs):
        print('loaded vgg model ' + model_path)
        self.graph1 = graph1
        self.session1 = session1
        if tf.__version__ == '1.14.0':
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Restrict TensorFlow to only use the first GPU
                try:
                    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
                except:
                    xxx = 1
            with graph1.as_default():
                with session1.as_default():
                    if loss is not None and loss_weights is not None:
                        loss = loss(loss_weights)
                    print('loading VGG Pretrained')
                    self.vgg_model_base = load_model(model_path, custom_objects={'BilinearUpsampling':Bilinear_model,'dice_coef_3D':dice_coef_3D,'loss':loss})
        else:
            with tf.device('/gpu:' + str(gpu)):
                with graph1.as_default():
                    with session1.as_default():
                        if loss is not None and loss_weights is not None:
                            loss = loss(loss_weights)
                        self.vgg_model_base = load_model(model_path, custom_objects={'BilinearUpsampling':Bilinear_model,'dice_coef_3D':dice_coef_3D,'loss':loss})

    def predict(self,images):
        return self.vgg_model_base.predict(images)


class Predict_On_Models():
    images = []

    def __init__(self,vgg_model, verbose=True,**kwargs):
        self.vgg_model = vgg_model
        self.verbose = verbose


    def vgg_pred_model(self):
        self.pred = self.vgg_model.predict(self.images)
        return None

    def make_predictions(self):
        self.vgg_pred_model()

class Resize_Images_Keras():
    def __init__(self,num_channels=1,image_size=256):
        if tf.__version__ == '1.14.0':
            device = tf.compat.v1.device
        else:
            device = tf.device
        with device('/gpu:0'):
            self.graph1 = Graph()
            with self.graph1.as_default():
                gpu_options = GPUOptions(allow_growth=True)
                self.session1 = Session(config=ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with self.session1.as_default():
                    self.input_val = Input((image_size, image_size, num_channels))
                    self.out = resize_images(self.input_val, 2, 2, 'channels_last')
    def resize_images(self,images):
        with self.graph1.as_default():
            with self.session1.as_default():
                x = self.session1.run(self.out,feed_dict={self.input_val:images})
        return x


def get_bounding_box_indexes(annotation, bbox=(0,0,0)):
    '''
    :param annotation: A binary image of shape [# images, # rows, # cols, channels]
    :return: the min and max z, row, and column numbers bounding the image
    '''
    annotation = np.squeeze(annotation)
    if annotation.dtype != 'int':
        annotation[annotation>0.1] = 1
        annotation = annotation.astype('int')
    indexes = np.where(np.any(annotation, axis=(1, 2)) == True)[0]
    min_z_s, max_z_s = indexes[0], indexes[-1]
    min_z_s = max([0, min_z_s - bbox[0]])
    max_z_s = min([annotation.shape[0], max_z_s + bbox[0]])
    '''
    Get the row values of primary and secondary
    '''
    indexes = np.where(np.any(annotation, axis=(0, 2)) == True)[0]
    min_r_s, max_r_s = indexes[0], indexes[-1]
    min_r_s = max([0, min_r_s - bbox[1]])
    max_r_s = min([annotation.shape[1], max_r_s + bbox[1]])
    '''
    Get the col values of primary and secondary
    '''
    indexes = np.where(np.any(annotation, axis=(0, 1)) == True)[0]
    min_c_s, max_c_s = indexes[0], indexes[-1]
    min_c_s = max([0, min_c_s - bbox[2]])
    max_c_s = min([annotation.shape[2], max_c_s + bbox[2]])
    return min_z_s, int(max_z_s + 1), min_r_s, int(max_r_s + 1), min_c_s, int(max_c_s + 1)

def variable_remove_non_liver(annotations, threshold=0.5, is_liver=False):
    image_size_1 = annotations.shape[1]
    image_size_2 = annotations.shape[2]
    compare = copy.deepcopy(annotations)
    if is_liver:
        images_filt = gaussian_filter(copy.deepcopy(annotations), [0, .75, .75])
    else:
        images_filt = gaussian_filter(copy.deepcopy(annotations), [0, 1.5, 1.5])
    compare[compare < .01] = 0
    compare[compare > 0] = 1
    compare = compare.astype('int')
    for i in range(annotations.shape[0]):
        image = annotations[i, :, :]
        out_image = np.zeros([image_size_1,image_size_2])

        labels = morphology.label(compare[i, :, :],connectivity=1)
        for xxx in range(1,labels.max() + 1):
            overlap = image[labels == xxx]
            pred = sum(overlap)/overlap.shape[0]
            cutoff = threshold
            if pred < 0.75:
                cutoff = 0.15
            if cutoff != 0.95 and overlap.shape[0] < 500 and is_liver:
                k = copy.deepcopy(compare[i, :, :])
                k[k > cutoff] = 1
                out_image[labels == xxx] = k[labels == xxx]
            elif not is_liver:
                image_filt = images_filt[i, :, :]
                image_filt[image_filt < threshold] = 0
                image_filt[image_filt > 0] = 1
                image_filt = image_filt.astype('int')
                out_image[labels == xxx] = image_filt[labels == xxx]
            else:
                image_filt = images_filt[i, :, :]
                image_filt[image_filt < cutoff] = 0
                image_filt[image_filt > 0] = 1
                image_filt = image_filt.astype('int')
                out_image[labels == xxx] = image_filt[labels == xxx]
        annotations[i, :, :] = out_image
    return annotations


def cleanout_folder(dicom_dir, empty_folder=True):
    files = []
    for _, _, files in os.walk(dicom_dir):
        break
    for file in files:
        os.remove(os.path.join(dicom_dir,file))
    if len(os.listdir(dicom_dir)) == 0 and empty_folder:
        os.rmdir(dicom_dir)
    return None

class Dicom_to_Imagestack:
    def __init__(self,delete_previous_rois=True, threshold=0.5,Contour_Names=None, template_dir=None, channels=3,
                 get_images_mask=True,arg_max=True,associations={'Liver_BMA_Program_4':'Liver','Liver':'Liver'}, **kwargs):
        self.arg_max = arg_max
        self.template_dir = template_dir
        self.delete_previous_rois = delete_previous_rois
        self.threshold = threshold
        self.Contour_Names = Contour_Names
        self.channels = channels
        keys = list(associations.keys())
        for key in keys:
            associations[key.lower()] = associations[key].lower()
        self.associations, self.hierarchy = associations, {}
        self.get_images_mask = get_images_mask
        self.reader = sitk.ImageSeriesReader()
        self.reader.MetaDataDictionaryArrayUpdateOn()
        self.reader.LoadPrivateTagsOn()
        self.all_RTs = []

    def make_array(self,PathDicom, single_structure=True):
        self.single_structure = single_structure
        self.PathDicom = PathDicom
        self.lstFilesDCM = []
        self.lstRSFile = None
        self.Dicom_info = []
        fileList = []
        for dirName, dirs, fileList in os.walk(PathDicom):
            break
        fileList = [i for i in fileList if i.find('.dcm') != -1]
        if not self.get_images_mask:
            RT_fileList = [i for i in fileList if i.find('RT') == 0 or i.find('RS') == 0]
            print(RT_fileList)
            if RT_fileList:
                fileList = RT_fileList
            for filename in fileList:
                try:
                    ds = pydicom.read_file(os.path.join(dirName,filename))
                    self.ds = ds
                    if ds.Modality == 'CT' or ds.Modality == 'MR' or ds.Modality == 'PT':  # check whether the file's DICOM
                        self.lstFilesDCM.append(os.path.join(dirName, filename))
                        self.Dicom_info.append(ds)
                        self.ds = ds
                    elif ds.Modality == 'RTSTRUCT':
                        self.lstRSFile = os.path.join(dirName, filename)
                        self.all_RTs.append(self.lstRSFile)
                except:
                    # if filename.find('Iteration_') == 0:
                    #     os.remove(PathDicom+filename)
                    continue
            if self.lstFilesDCM:
                self.RefDs = pydicom.read_file(self.lstFilesDCM[0])
        else:
            self.dicom_names = self.reader.GetGDCMSeriesFileNames(self.PathDicom)
            self.reader.SetFileNames(self.dicom_names)
            self.get_images()
            image_files = [i.split(PathDicom)[1][1:] for i in self.dicom_names]
            lstRSFiles = [os.path.join(PathDicom, file) for file in fileList if file not in image_files]
            if lstRSFiles:
                self.lstRSFile = lstRSFiles[0]
            self.RefDs = pydicom.read_file(self.dicom_names[0])
            self.ds = pydicom.read_file(self.dicom_names[0])
        self.mask_exist = False
        self.rois_in_case = []
        if self.lstRSFile is not None:
            self.get_rois_from_RT()
        elif self.get_images_mask:
            self.use_template()

    def get_rois_from_RT(self):
        self.RS_struct = pydicom.read_file(self.lstRSFile)
        if Tag((0x3006, 0x020)) in self.RS_struct.keys():
            self.ROI_Structure = self.RS_struct.StructureSetROISequence
        else:
            self.ROI_Structure = []
        for Structures in self.ROI_Structure:
            if Structures.ROIName not in self.rois_in_case:
                self.rois_in_case.append(Structures.ROIName)

    def get_mask(self):
        self.mask = np.zeros([len(self.dicom_names),self.image_size_1, self.image_size_2, len(self.Contour_Names)+1],
                             dtype='int8')
        self.structure_references = {}
        for contour_number in range(len(self.RS_struct.ROIContourSequence)):
            self.structure_references[self.RS_struct.ROIContourSequence[contour_number].ReferencedROINumber] = contour_number
        found_rois = {}
        for Structures in self.ROI_Structure:
            ROI_Name = Structures.ROIName
            if Structures.ROINumber not in self.structure_references.keys():
                continue
            true_name = None
            if ROI_Name in self.associations:
                true_name = self.associations[ROI_Name]
            elif ROI_Name.lower() in self.associations:
                true_name = self.associations[ROI_Name.lower()]
            if true_name and true_name in self.Contour_Names:
                found_rois[true_name] = {'Hierarchy':999,'Name':ROI_Name,'Roi_Number':Structures.ROINumber}
        for ROI_Name in found_rois.keys():
            if found_rois[ROI_Name]['Roi_Number'] in self.structure_references:
                index = self.structure_references[found_rois[ROI_Name]['Roi_Number']]
                mask = self.get_mask_for_contour(index)
                self.mask[...,self.Contour_Names.index(ROI_Name)+1][mask == 1] = 1
        if self.arg_max:
            self.mask = np.argmax(self.mask,axis=-1)
        self.annotation_handle = sitk.GetImageFromArray(self.mask.astype('int8'))
        self.annotation_handle.SetSpacing(self.dicom_handle.GetSpacing())
        self.annotation_handle.SetOrigin(self.dicom_handle.GetOrigin())
        self.annotation_handle.SetDirection(self.dicom_handle.GetDirection())
        return None

    def get_mask_for_contour(self,i):
        self.Liver_Locations = self.RS_struct.ROIContourSequence[i].ContourSequence
        self.Liver_Slices = []
        for contours in self.Liver_Locations:
            data_point = contours.ContourData[2]
            if data_point not in self.Liver_Slices:
                self.Liver_Slices.append(contours.ContourData[2])
        return self.Contours_to_mask()

    def Contours_to_mask(self):
        mask = np.zeros([len(self.dicom_names), self.image_size_1, self.image_size_2], dtype='int8')
        Contour_data = self.Liver_Locations
        ShiftCols, ShiftRows, _ = [float(i) for i in self.reader.GetMetaData(0,"0020|0032").split('\\')]
        PixelSize = self.dicom_handle.GetSpacing()[0]
        Mag = 1 / PixelSize
        mult1 = mult2 = 1
        if ShiftCols > 0:
            mult1 = -1
        if ShiftRows > 0:
            print('take a look at this one...')
        #    mult2 = -1

        for i in range(len(Contour_data)):
            referenced_sop_instance_uid = Contour_data[i].ContourImageSequence[0].ReferencedSOPInstanceUID
            if referenced_sop_instance_uid not in self.SOPInstanceUIDs:
                print('Error here with instance UID')
                return None
            else:
                slice_index = self.SOPInstanceUIDs.index(referenced_sop_instance_uid)
            cols = Contour_data[i].ContourData[1::3]
            rows = Contour_data[i].ContourData[0::3]
            col_val = [Mag * abs(x - mult1 * ShiftRows) for x in cols]
            row_val = [Mag * abs(x - mult2 * ShiftCols) for x in rows]
            temp_mask = self.poly2mask(col_val, row_val, [self.image_size_1, self.image_size_2])
            mask[slice_index,:,:][temp_mask > 0] = 1
            #scm.imsave('C:\\Users\\bmanderson\\desktop\\images\\mask_'+str(i)+'.png',mask_slice)

        return mask

    def use_template(self):
        self.template = True
        if not self.template_dir:
            self.template_dir = os.path.join('\\\\mymdafiles', 'ro-admin', 'SHARED', 'Radiation physics', 'BMAnderson',
                                             'Auto_Contour_Sites', 'template_RS.dcm')
            if not os.path.exists(self.template_dir):
                self.template_dir = os.path.join('..', '..', 'Shared_Drive', 'Auto_Contour_Sites', 'template_RS.dcm')
        self.key_list = self.template_dir.replace('template_RS.dcm', 'key_list.txt')
        self.RS_struct = pydicom.read_file(self.template_dir)
        print('Running off a template')
        self.changetemplate()

    def get_images(self):
        self.dicom_handle = self.reader.Execute()
        sop_instance_UID_key = "0008|0018"
        self.SOPInstanceUIDs = [self.reader.GetMetaData(i, sop_instance_UID_key) for i in
                                range(self.dicom_handle.GetDepth())]
        slice_location_key = "0020|0032"
        self.slice_info = [self.reader.GetMetaData(i, slice_location_key).split('\\')[-1] for i in
                           range(self.dicom_handle.GetDepth())]
        # Working on the RS structure now
        # The array is sized based on 'ConstPixelDims'
        # ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
        self.ArrayDicom = sitk.GetArrayFromImage(self.dicom_handle)
        self.image_size_1, self.image_size_2, _ = self.dicom_handle.GetSize()

    def poly2mask(self,vertex_row_coords, vertex_col_coords, shape):
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        mask = np.zeros(shape, dtype=np.bool)
        mask[fill_row_coords, fill_col_coords] = True
        return mask
    def with_annotations(self,annotations,output_dir,ROI_Names=None):
        annotations = np.squeeze(annotations)
        self.image_size_0, self.image_size_1 = annotations.shape[1], annotations.shape[2]
        self.ROI_Names = ROI_Names
        self.output_dir = output_dir
        if len(annotations.shape) == 3:
            annotations = np.expand_dims(annotations,axis=-1)
        self.annotations = annotations
        self.Mask_to_Contours()

    def Mask_to_Contours(self):
        self.RefDs = self.ds
        self.ShiftCols, self.ShiftRows, _ = [float(i) for i in self.reader.GetMetaData(0, "0020|0032").split('\\')]
        self.mult1 = self.mult2 = 1
        self.PixelSize = self.dicom_handle.GetSpacing()[0]
        current_names = []
        for names in self.RS_struct.StructureSetROISequence:
            current_names.append(names.ROIName)
        Contour_Key = {}
        xxx = 1
        for name in self.ROI_Names:
            Contour_Key[name] = xxx
            xxx += 1
        self.all_annotations = self.annotations
        base_annotations = copy.deepcopy(self.annotations)
        temp_color_list = []
        color_list = [[128,0,0],[170,110,40],[0,128,128],[0,0,128],[230,25,75],[225,225,25],[0,130,200],[145,30,180],
                      [255,255,255]]
        self.struct_index = -1
        for Name in self.ROI_Names:
            if not temp_color_list:
                temp_color_list = copy.deepcopy(color_list)
            color_int = np.random.randint(len(temp_color_list))
            print('Writing data for ' + Name)
            self.annotations = copy.deepcopy(base_annotations[:,:,:,int(self.ROI_Names.index(Name)+1)])
            if 'Liver_BMA_Program_4_2Dfast' in self.ROI_Names or 'Liver_BMA_Program_4_3D' in self.ROI_Names:
                thresholds = [0.2,0.75,0.2]
                reduced_annotations = remove_non_liver(self.annotations, threshold=thresholds[0])
                self.annotations[reduced_annotations == 0] = 0
                self.annotations = variable_remove_non_liver(self.annotations, threshold=thresholds[1])
                self.annotations = remove_non_liver(self.annotations, threshold=thresholds[2])
            elif self.threshold != 0:
                threshold = self.threshold
                for roi in self.ROI_Names:
                    if roi.find('Pancreas') != -1:
                        threshold = 0.5
                        break
                    if roi.find('right_eye_bma') != -1:
                        threshold = 0.75
                        break
                self.annotations = variable_remove_non_liver(self.annotations, threshold=0.2, structure_name=self.ROI_Names)
                if self.single_structure:
                    self.annotations = remove_non_liver(self.annotations, threshold=threshold)

            self.annotations = self.annotations.astype('int')

            make_new = 1
            allow_slip_in = True
            if Name not in current_names and allow_slip_in:
                self.RS_struct.StructureSetROISequence.append(copy.deepcopy(self.RS_struct.StructureSetROISequence[0]))
                if not self.template:
                    self.struct_index = len(self.RS_struct.StructureSetROISequence)-1
                else:
                    self.struct_index += 1
            else:
                make_new = 0
                self.struct_index = current_names.index(Name) - 1
            new_ROINumber = self.struct_index + 1
            self.RS_struct.StructureSetROISequence[self.struct_index].ROINumber = new_ROINumber
            self.RS_struct.StructureSetROISequence[self.struct_index].ReferencedFrameOfReferenceUID = self.ds.FrameOfReferenceUID
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIName = Name
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIVolume = 0
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIGenerationAlgorithm = 'SEMIAUTOMATIC'
            if make_new == 1:
                self.RS_struct.RTROIObservationsSequence.append(copy.deepcopy(self.RS_struct.RTROIObservationsSequence[0]))
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ObservationNumber = new_ROINumber
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ReferencedROINumber = new_ROINumber
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ROIObservationLabel = Name
            self.RS_struct.RTROIObservationsSequence[self.struct_index].RTROIInterpretedType = 'ORGAN'

            if make_new == 1:
                self.RS_struct.ROIContourSequence.append(copy.deepcopy(self.RS_struct.ROIContourSequence[0]))
            self.RS_struct.ROIContourSequence[self.struct_index].ReferencedROINumber = new_ROINumber
            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[1:] = []
            self.RS_struct.ROIContourSequence[self.struct_index].ROIDisplayColor = temp_color_list[color_int]
            del temp_color_list[color_int]

            contour_num = 0
            if np.max(self.annotations) > 0: # If we have an annotation, write it
                image_locations = np.max(self.annotations,axis=(1,2))
                indexes = np.where(image_locations>0)[0]
                for point, i in enumerate(indexes):
                    print(str(int(point / len(indexes) * 100)) + '% done with ' + Name)
                    annotation = self.annotations[i,:,:]
                    regions = regionprops(label(annotation),coordinates='xy')
                    for ii in range(len(regions)):
                        temp_image = np.zeros([self.image_size_0,self.image_size_1])
                        data = regions[ii].coords
                        rows = []
                        cols = []
                        for iii in range(len(data)):
                            rows.append(data[iii][0])
                            cols.append(data[iii][1])
                        temp_image[rows,cols] = 1
                        points = find_contours(temp_image, 0)[0]
                        output = []
                        for point in points:
                            output.append(((point[1]) * self.PixelSize + self.mult1 * self.ShiftCols))
                            output.append(((point[0]) * self.PixelSize + self.mult2 * self.ShiftRows))
                            output.append(float(self.slice_info[i]))
                        if output:
                            if contour_num > 0:
                                self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence.append(copy.deepcopy(self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[0]))
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[contour_num].ContourNumber = str(contour_num)
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[contour_num].ContourImageSequence[0].ReferencedSOPInstanceUID = self.SOPInstanceUIDs[i]
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[contour_num].ContourData = output
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[contour_num].NumberofContourPoints = round(len(output)/3)
                            contour_num += 1
        self.RS_struct.SOPInstanceUID += '.' + str(np.random.randint(999))
        if self.template and self.delete_previous_rois:
            for i in range(len(self.RS_struct.StructureSetROISequence) - len(self.ROI_Names)):
                del self.RS_struct.StructureSetROISequence[-1]
            for i in range(len(self.RS_struct.RTROIObservationsSequence) - len(self.ROI_Names)):
                # if self.RS_struct.RTROIObservationsSequence[i].ROIObservationLabel not in self.ROI_Names:
                del self.RS_struct.RTROIObservationsSequence[-1]
                # if self.RS_struct.RTROIObservationsSequence[i].ROIObservationLabel not in self.ROI_Names:
            for i in range(len(self.RS_struct.ROIContourSequence) - len(self.ROI_Names)):
                del self.RS_struct.ROIContourSequence[-1]
        for i in range(len(self.RS_struct.StructureSetROISequence)):
            self.RS_struct.StructureSetROISequence[i].ROINumber = i + 1
            self.RS_struct.RTROIObservationsSequence[i].ReferencedROINumber = i + 1
            self.RS_struct.ROIContourSequence[i].ReferencedROINumber = i + 1

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        out_name = os.path.join(self.output_dir,'RS_MRN' + self.RS_struct.PatientID + '_' + self.RS_struct.SeriesInstanceUID + '.dcm')
        if os.path.exists(out_name):
            out_name = os.path.join(self.output_dir,'RS_MRN' + self.RS_struct.PatientID + '_' + self.RS_struct.SeriesInstanceUID + '1.dcm')
        print('Writing out data...')
        pydicom.write_file(out_name, self.RS_struct)
        fid = open(os.path.join(self.output_dir,'Completed.txt'), 'w+')
        fid.close()
        print('Finished!')
        #Raystation_dir = self.output_dir.split('Output_MRN')[0]+'Output_MRN_RayStation\\'+self.RS_struct.PatientID+'\\'
        #if not os.path.exists(Raystation_dir):
        #dicom.write_file(Raystation_dir + 'RS_MRN' + self.RS_struct.PatientID + '_' + self.ds.SeriesInstanceUID + '.dcm', self.RS_struct)
        #fid = open(Raystation_dir+'Completed.txt','w+')
        #fid.close()
        return None

    def changetemplate(self):
        keys = self.RS_struct.keys()
        for key in keys:
            #print(self.RS_struct[key].name)
            if self.RS_struct[key].name == 'Referenced Frame of Reference Sequence':
                break
        self.RS_struct[key]._value[0].FrameOfReferenceUID = self.ds.FrameOfReferenceUID
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].ReferencedSOPInstanceUID = self.ds.StudyInstanceUID
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID = self.ds.SeriesInstanceUID
        for i in range(len(self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence)-1):
            del self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence[-1]
        fill_segment = copy.deepcopy(self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence[0])
        for i in range(len(self.SOPInstanceUIDs)):
            temp_segment = copy.deepcopy(fill_segment)
            temp_segment.ReferencedSOPInstanceUID = self.SOPInstanceUIDs[i]
            self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence.append(temp_segment)
        del self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence[0]

        new_keys = open(self.key_list)
        keys = {}
        i = 0
        for line in new_keys:
            keys[i] = line.strip('\n').split(',')
            i += 1
        new_keys.close()
        for index in keys.keys():
            new_key = keys[index]
            try:
                self.RS_struct[new_key[0], new_key[1]] = self.ds[[new_key[0], new_key[1]]]
            except:
                continue
        return None
            # Get slice locations

def poly2mask(vertex_row_coords, vertex_col_coords):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, [512,512])
    mask = np.zeros([512,512], dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


class Image_Clipping_and_Padding(object):
    def __init__(self, return_mask=False, mean_val=1, std_val=0):
        self.mean_val, self.std_val = mean_val, std_val
        atrous_rate = 2
        filters = 16
        num_atrous_blocks = 3
        layers = 3
        layers_dict = {}
        atrous_block = {'Channels': [filters], 'Atrous_block': [atrous_rate]}
        for layer in range(layers - 1):
            pool = (2, 2, 2)
            layers_dict['Layer_' + str(layer)] = {'Encoding': [atrous_block for _ in range(num_atrous_blocks)],
                                                  'Pooling': pool,
                                                  'Decoding': [atrous_block for _ in range(num_atrous_blocks)]}
            filters = int(filters * 2)
            atrous_block = {'Channels': [filters], 'Atrous_block': [atrous_rate]}
            num_atrous_blocks *= 2
        num_atrous_blocks *= 2
        layers_dict['Base'] = {'Encoding': [atrous_block for _ in range(num_atrous_blocks)]}
        self.patient_dict = {}
        power_val_z, power_val_x, power_val_y = (1,1,1)
        pool_base = 2
        for layer in layers_dict:
            if layer == 'Base':
                continue
            if 'Pooling' in layers_dict[layer]:
                pooling = layers_dict[layer]['Pooling']
            else:
                pooling = [pool_base for _ in range(3)]
            power_val_z *= pooling[0]
            power_val_x *= pooling[1]
            power_val_y *= pooling[2]
        self.return_mask = return_mask
        self.power_val_z, self.power_val_x, self.power_val_y = power_val_z, power_val_x, power_val_y

    def pad_images(self, x, liver):
        x = (x - self.mean_val) / self.std_val
        x[x<-3.55] = -3.55
        x[x>3.55] = 3.55
        x = (x - -3.55) / (3.55 - -3.55)
        z_start, z_stop, r_start, r_stop, c_start, c_stop = get_bounding_box_indexes(liver)
        z_start = max([0,z_start-5])
        z_stop = min([z_stop+5,x.shape[1]])
        r_start = max([0,r_start-10])
        r_stop = min([512,r_stop+10])
        c_start = max([0,c_start-10])
        c_stop = min([512,c_stop+10])
        z_total, r_total, c_total = z_stop - z_start, r_stop - r_start, c_stop - c_start
        remainder_z, remainder_r, remainder_c = self.power_val_z - z_total % self.power_val_z if z_total % self.power_val_z != 0 else 0, \
                                                self.power_val_x - r_total % self.power_val_x if r_total % self.power_val_x != 0 else 0, \
                                                self.power_val_y - c_total % self.power_val_y if c_total % self.power_val_y != 0 else 0
        min_images, min_rows, min_cols = z_total + remainder_z, r_total + remainder_r, c_total + remainder_c
        out_images = np.zeros([1,min_images,min_rows,min_cols,1],dtype=x.dtype)
        out_annotations = np.zeros([1,min_images,min_rows,min_cols,1],dtype=liver.dtype)
        out_images[:,0:z_stop-z_start,:r_stop-r_start,:c_stop-c_start,:] = x[:,z_start:z_stop,r_start:r_stop,c_start:c_stop,:]
        out_annotations[:,0:z_stop-z_start,:r_stop-r_start,:c_stop-c_start,:] = liver[z_start:z_stop,r_start:r_stop,c_start:c_stop,:]
        return out_images, out_annotations


def main():
    pass


if __name__ == "__main__":
    main()
