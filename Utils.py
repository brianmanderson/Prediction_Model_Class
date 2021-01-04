import os, shutil
import matplotlib.pyplot as plt
import pydicom
from pydicom.tag import Tag
import copy
from skimage import draw, morphology
from skimage.measure import label,regionprops,find_contours
import numpy as np
from scipy.ndimage import gaussian_filter
import tensorflow as tf
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

    weights = tf.compat.v1.keras.backend.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.compat.v1.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.compat.v1.keras.backend.clip(y_pred, tf.compat.v1.keras.backend.epsilon(), 1 - tf.compat.v1.keras.backend.epsilon())
        # calc
        loss = y_true * tf.compat.v1.keras.backend.log(y_pred) * weights
        loss = -tf.compat.v1.keras.backend.sum(loss, -1)
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
    intersection = tf.keras.backend.sum(y_true[...,1:] * y_pred[...,1:])
    union = tf.keras.backend.sum(y_true[...,1:]) + tf.keras.backend.sum(y_pred[...,1:])
    return (2. * intersection + smooth) / (union + smooth)


class Resize_Images_Keras():
    def __init__(self,num_channels=1,image_size=256):
        if tf.__version__ == '1.14.0':
            device = tf.compat.v1.device
        else:
            device = tf.device
        with device('/gpu:0'):
            self.graph1 = tf.compat.v1.Graph()
            with self.graph1.as_default():
                gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
                self.session1 = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
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
    return min_z_s, max_z_s, min_r_s, max_r_s, min_c_s, max_c_s


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
        os.remove(os.path.join(dicom_dir, file))
    if len(os.listdir(dicom_dir)) == 0 and empty_folder:
        os.rmdir(dicom_dir)
    return None


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
