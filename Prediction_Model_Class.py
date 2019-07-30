import sys, os
sys.path.append('..')
# from keras.models import *
from tensorflow.python.client import device_lib
# import BMA_Utils as utils_BMA
import Utils as utils_BMA
from Keras_Utils import VGG_Model_Pretrained, Predict_On_Models, Resize_Images_Keras, K, get_bounding_box_indexes
from tensorflow import Graph, Session, ConfigProto, GPUOptions
from skimage.measure import block_reduce
from TensorflowUtils import plot_scroll_Image, normalize_images, down_folder
from functools import partial
import tensorflow as tf
import numpy as np
import copy


class Image_Clipping_and_Padding(object):
    def __init__(self):
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
        self.power_val_z, self.power_val_x, self.power_val_y = power_val_z, power_val_x, power_val_y

    def process(self, x, y):
        liver = np.sum(y[...,1:],axis=-1)
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
        out_annotations = np.zeros([1, min_images, min_rows, min_cols, y.shape[-1]], dtype=y.dtype)
        out_annotations[..., 0] = 1
        out_images[:,0:z_stop-z_start,:r_stop-r_start,:c_stop-c_start,:] = x[:,z_start:z_stop,r_start:r_stop,c_start:c_stop,:]
        out_annotations[:,0:z_stop-z_start,:r_stop-r_start,:c_stop-c_start,:] = y[:,z_start:z_stop,r_start:r_stop,c_start:c_stop,:]
        return out_images, out_annotations


def convert_image_size(images,image_size):
    while images.shape[1] != image_size:
        difference_1 = image_size - images.shape[1]
        if difference_1 > 0:
            images = np.concatenate((images, images[:, :int(difference_1/2), :, :]),
                                    axis=1)
            images = np.concatenate((images[:, -int(difference_1/2):, :, :], images),
                                    axis=1)
        elif difference_1 < 0:
            images = images[:, :int(difference_1 / 2), :, :]
            images = images[:, abs(int(difference_1 / 2)):, :, :]
    while images.shape[2] != image_size:
        difference_2 = image_size - images.shape[2]
        if difference_2 > 0:
            images = np.concatenate((images, images[:, :, :int(difference_2/2), :]),
                                    axis=2)
            images = np.concatenate((images[:, :, -int(difference_2/2):, :], images),
                                    axis=2)
        elif difference_2 < 0:
            images = images[:, :, :int(difference_2 / 2), :]
            images = images[:, :, abs(int(difference_2 / 2)):, :]
    return images


def convert_annotation_out_size(annotations,image_og_size):
    while annotations.shape[1] != image_og_size[1]:
        difference_1 = image_og_size[1] - annotations.shape[1]
        if difference_1 > 0:
            temp_annotations = np.zeros(annotations.shape)
            annotations = np.concatenate((annotations, temp_annotations[:, :int(difference_1 / 2), :, :]),
                                         axis=1)
            annotations = np.concatenate(
                (temp_annotations[:, :int(difference_1 / 2), :, :], annotations),
                axis=1)
        elif difference_1 < 0:
            annotations = annotations[:, :int(difference_1 / 2), :, :]
            annotations = annotations[:, abs(int(difference_1 / 2)):, :, :]
    while annotations.shape[2] != image_og_size[2]:
        difference_2 = image_og_size[2] - annotations.shape[2]
        if difference_2 > 0:
            temp_annotations = np.zeros(annotations.shape)
            annotations = np.concatenate((annotations, temp_annotations[:, :, :int(difference_2 / 2), :]),
                                         axis=2)
            annotations = np.concatenate(
                (temp_annotations[:, :, :int(difference_2 / 2), :], annotations),
                axis=2)
        elif difference_2 < 0:
            annotations = annotations[:, :, :int(difference_2 / 2), :]
            annotations = annotations[:, :, abs(int(difference_2 / 2)):, :]
    return annotations


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def run_model(gpu=0):
    G = get_available_gpus()
    if len(G) == 1:
        gpu = 0
    with K.tf.device('/gpu:' + str(gpu)):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        K.set_session(sess)
        models_info = {}
        try:
            os.listdir('\\\\mymdafiles\\di_data1\\')
            morfeus_path = '\\\\mymdafiles\\di_data1\\'
            shared_drive_path = '\\\\mymdafiles\\ro-ADMIN\\SHARED\\Radiation physics\\BMAnderson\\Auto_Contour_Sites\\'
            raystation_drive_path = '\\\\mymdafiles\\ou-radonc\\Raystation\\Clinical\\Auto_Contour_Sites\\'
        except:
            morfeus_path = os.path.abspath(os.path.join('..','..','..'))
            shared_drive_path = os.path.abspath(os.path.join('..','..','..','Shared_Drive','Auto_Contour_Sites'))
            raystation_drive_path = os.path.abspath(os.path.join('..','..','..','Raystation_LDrive','Clinical','Auto_Contour_Sites'))
        template_dir = os.path.join(shared_drive_path,'template_RS.dcm')
        base_dicom_reader = utils_BMA.Dicom_to_Imagestack(template_dir=template_dir)
        Image_Clipping = Image_Clipping_and_Padding()
        model_info = {'model_path':os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Models','Pancreas','weights-improvement-v3_xception_512-12.hdf5'),
                      'names':['Pancreas_BMA_Program'],'vgg_model':[], 'model_image_size':512,
                      'path':[os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Pancreas_Auto_Contour','Input_3'),
                              os.path.join(shared_drive_path,'Pancreas_Auto_Contour','Input_3')],'is_CT':True,
                      'single_structure': True,'mean_val':0,'std_val':1,'vgg_normalize':True,'file_loader':base_dicom_reader,'post_process':partial(normalize_images,lower_threshold=-100,upper_threshold=300, is_CT=True, mean_val=0,std_val=1)}
        # models_info['pancreas'] = model_info
        # model_info = {'model_path':os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Models','Spleen','weights-improvement-10.hdf5'),
        #               'names':['Spleen_BMA_Program_4'],'vgg_model':[], 'model_image_size':512,
        #               'path':[os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Spleen_Auto_contour','Input_3'),
        #                       os.path.join(shared_drive_path, 'Spleen_Auto_Contour', 'Input_3')],'is_CT':True,
        #               'single_structure': True,'mean_val':40,'std_val':45,'vgg_normalize':False,'threshold':0.5}
        # models_info['spleen'] = model_info
        model_info = {'model_path':os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Models','Liver','weights-improvement-512_v3_model_xception-36.hdf5'),
                      'names':['Liver_BMA_Program_4'],'vgg_model':[], 'model_image_size':512,'post_process':partial(normalize_images,lower_threshold=-100,upper_threshold=300, is_CT=True, mean_val=0,std_val=1),
                      'path':[#os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Liver_Auto_Contour','Input_3'),
                              os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Liver_Auto_Contour','Input_3'),
                              os.path.join(raystation_drive_path,'Liver_Auto_Contour','Input_3')],'is_CT':True,
                      'single_structure': True,'mean_val':0,'std_val':1,'vgg_normalize':True,'threshold':0.5,'file_loader':base_dicom_reader}
        models_info['liver'] = model_info
        model_info = {'model_path':os.path.join(morfeus_path,'Morfeus','BMAnderson','CNN','Data','Data_Liver','Liver_Segments','weights-improvement-200.hdf5'),
                      'names':['Liver_Segment_' + str(i) for i in range(1, 9)],'vgg_model':[], 'model_image_size':512,
                      'path':[#os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Liver_Auto_Contour','Input_3'),
                              os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Liver_Segments_Auto_Contour','Input_3'),
                              os.path.join(raystation_drive_path,'Liver_Segments_Auto_Contour','Input_3')],'is_CT':True,
                      'single_structure': True,'mean_val':80,'std_val':40,'vgg_normalize':True,'threshold':0.5,
                      'file_loader':utils_BMA.Dicom_to_Imagestack(Contour_Names=['Liver'],template_dir=template_dir)}
        # models_info['liver_lobes'] = model_info
        # model_info = {'model_path':os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Models','Cervix','weights-improvement-20.hdf5'),
        #               'names':['UterineCervix_BMA_Program_4'],'vgg_model':[], 'model_image_size':512,
        #               'path':[os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Cervix_Auto_Contour','Input_3'),
        #                       os.path.join(shared_drive_path,'Cervix_Auto_Contour','Input_3')],'is_CT':True,
        #               'single_structure': True,'mean_val':40,'std_val':32,'vgg_normalize':False,'threshold':0.4}
        # models_info['cervix'] = model_info


        vgg_unet = None
        all_sessions = {}
        resize_class_256 = Resize_Images_Keras(num_channels=1)
        resize_class_512 = Resize_Images_Keras(num_channels=1, image_size=512)
        graph1 = Graph()
        with graph1.as_default():
            gpu_options = GPUOptions(allow_growth=True)
            for key in models_info.keys():
                session1 = Session(config=ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with session1.as_default():
                    K.set_session(session1)
                    num_classes = int(1+len(models_info[key]['names']))
                    models_info[key]['vgg_model'] = VGG_Model_Pretrained(models_info[key]['model_path'],num_classes=num_classes,gpu=gpu,
                                                                         image_size=models_info[key]['model_image_size'],graph1=graph1,session1=session1)
                    models_info[key]['predict_model'] = Predict_On_Models(models_info[key]['vgg_model'], vgg_unet,
                                                                          is_CT=models_info[key]['is_CT'],vgg_normalize=models_info[key]['vgg_normalize'],
                                                                          image_size=models_info[key]['model_image_size'],
                                                                          use_unet=False,num_classes=num_classes, step=60)
                    models_info[key]['resize_class_256'] = resize_class_256
                    models_info[key]['resize_class_512'] = resize_class_512
                    all_sessions[key] = session1

        running = True
        print('running')
        attempted = {}
        with graph1.as_default():
            while running and os.path.exists(os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Liver_Auto_Contour','Running.txt')):
                for key in models_info.keys():
                    with all_sessions[key].as_default():
                        K.set_session(all_sessions[key])
                        for path in models_info[key]['path']:
                            dicom_folder_all_out = down_folder(path,[])
                            for dicom_folder in dicom_folder_all_out:
                                print(dicom_folder)
                                if dicom_folder not in attempted.keys():
                                    attempted[dicom_folder] = 0
                                else:
                                    attempted[dicom_folder] += 1
                                print('running')
                                try:
                                    fid = open(os.path.join(dicom_folder,'running.txt'),'w+')
                                    fid.close()
                                    images_class = models_info[key]['file_loader']
                                    images_class.make_array(dicom_folder, single_structure=models_info[key]['single_structure'])
                                    output = os.path.join(path.split('Input_3')[0], 'Output')
                                    true_outpath = os.path.join(output,images_class.ds.PatientID,images_class.SeriesInstanceUID)
                                    images = images_class.ArrayDicom
                                    if 'post_process' in models_info[key]:
                                        images = models_info[key]['post_process'](images)
                                    image_og_size = copy.deepcopy(images.shape)
                                    image_size = models_info[key]['model_image_size']
                                    mult = 0
                                    if images.shape[1] >= image_size*2:
                                        images = block_reduce(images,(1,2,2,1),np.average)
                                        mult = 1
                                    elif images.shape[1] <= int(image_size/2) or images.shape[2] <= int(image_size/2):
                                        images = convert_image_size(images, 256)
                                        images = models_info[key]['resize_class_256'].resize_images(images)
                                        mult = -1
                                    images = convert_image_size(images,image_size)

                                    models_info[key]['predict_model'].images = images
                                    models_info[key]['predict_model'].make_predictions()
                                    pred = models_info[key]['predict_model'].pred
                                    if mult == 1:
                                        annotations = np.zeros(image_og_size[:-1] + tuple([pred.shape[-1]]))
                                        for i in range(pred.shape[-1]):
                                            annotations[...,i] = models_info[key]['resize_class_'+str(pred.shape[1])].resize_images(pred[...,i][...,None])[...,-1]
                                    elif mult == -1:
                                        annotations = block_reduce(pred,(1,2,2,1),np.average)
                                    else:
                                        annotations = pred
                                    annotations = convert_annotation_out_size(annotations,image_og_size)
                                    if 'threshold' in models_info[key].keys():
                                        images_class.theshold = models_info[key]['threshold']
                                    images_class.template = 1

                                    images_class.with_annotations(annotations,true_outpath,
                                                                  ROI_Names=models_info[key]['names'])

                                    print('RT structure ' + images_class.ds.PatientID + ' printed to ' + os.path.join(output,
                                          images_class.ds.PatientID,images_class.SeriesInstanceUID) + ' with name: RS_MRN'
                                          + images_class.ds.PatientID + '.dcm')

                                    utils_BMA.cleanout_folder(dicom_folder)
                                    attempted[dicom_folder] = -1
                                except:
                                    if attempted[dicom_folder] >= 1:
                                        attempted[dicom_folder] += 1
                                        print('Failed once.. trying again')
                                        continue
                                    else:
                                        try:
                                            print('Failed twice')
                                            utils_BMA.cleanout_folder(dicom_folder)
                                            print('had an issue')
                                        except:
                                            xxx = 1
                                        continue

if __name__ == '__main__':
    run_model(gpu=0)