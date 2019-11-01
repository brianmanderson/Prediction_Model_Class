import os, time
from tensorflow.python.client import device_lib
import Utils as utils_BMA
from Utils import Fill_Missing_Segments, Check_ROI_Names
from Utils import VGG_Model_Pretrained, Predict_On_Models, Resize_Images_Keras, K, get_bounding_box_indexes, plot_scroll_Image, normalize_images, down_folder
from tensorflow import Graph, Session, ConfigProto, GPUOptions
from Bilinear_Dsc import BilinearUpsampling
from functools import partial
import tensorflow as tf
import numpy as np


class Image_Processor(object):

    def get_path(self, PathDicom):
        self.PathDicom = PathDicom

    def pre_process(self, images, annotations=None):
        if annotations is None:
            return images
        else:
            return images, annotations

    def post_process(self, images, annotations=None):
        if annotations is None:
            return images
        else:
            return images, annotations


class Make_3D(Image_Processor):
    def pre_process(self, images, annotations=None):
        return images[None,...]

    def post_process(self, images, annotations=None):
        return np.squeeze(images), np.squeeze(annotations)


class Reduce_Prediction(Image_Processor):
    def post_process(self, images, annotations=None):
        annotations[annotations<0.5] = 0
        return images, annotations


class Image_Clipping_and_Padding(Image_Processor):
    def __init__(self, layers=3, return_mask=False, liver_box=False):
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

    def pre_process(self, x,y=None):
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
        if y is not None:
            out_annotations = np.zeros([min_images, min_rows, min_cols, y.shape[-1]], dtype=y.dtype)
            out_annotations[..., 0] = 1
            out_annotations[:,0:z_stop-z_start,:r_stop-r_start,:c_stop-c_start,:] = y[:,z_start:z_stop,r_start:r_stop,c_start:c_stop,:]
            if self.return_mask:
                return [out_images,np.sum(out_annotations[...,1:],axis=-1)[...,None]], out_annotations
            return out_images, out_annotations
        else:
            return out_images


class Turn_Two_Class_Three(Image_Processor):
    def post_process(self, images, annotations=None):
        i_size = annotations.shape[1]
        new_output = np.zeros([annotations.shape[0], annotations.shape[1], annotations.shape[2], 3], dtype=annotations.dtype)
        new_output[..., 0] = annotations[..., 0]
        new_output[:, :, :i_size // 2, 1] = annotations[:, :, :i_size // 2, 1]
        new_output[:, :, i_size // 2:, 2] = annotations[:, :, i_size // 2:, 1]
        return images, new_output


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
        return output_images

    def post_process(self, images, annotations=None):
        out_annotations = np.zeros([self.og_image_size[0],self.og_image_size[1],self.og_image_size[2],annotations.shape[-1]])
        out_annotations[:,self.start_r:annotations.shape[1] + self.start_r,self.start_c:annotations.shape[2] + self.start_c,...] = annotations
        return images, out_annotations


class Normalize_Images(Image_Processor):
    def __init__(self, mean_val=0, std_val=1, upper_threshold=None, lower_threshold=None, max_val=1):
        self.mean_val, self.std_val = mean_val, std_val
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.max_val = max_val

    def pre_process(self, images, annotation=None):
        if self.upper_threshold is not None:
            images[images > self.upper_threshold] = self.upper_threshold
        if self.lower_threshold is not None:
            images[images < self.lower_threshold] = self.lower_threshold
        if self.mean_val != 0 or self.std_val != 1:
            images = (images - self.mean_val) / self.std_val
            images[images>3.55] = 3.55
            images[images<-3.55] = -3.55
        else:
            images = (images - self.lower_threshold) /(self.upper_threshold - self.lower_threshold) * self.max_val
        return images


class Ensure_Liver_Segmentation(Image_Processor):
    def __init__(self, associations=None, wanted_roi='Liver'):
        self.associations = associations
        self.wanted_roi = wanted_roi
        self.Fill_Missing_Segments_Class = Fill_Missing_Segments()
        self.ROI_Checker = Check_ROI_Names()


    def check_ROIs_In_Checker(self):
        for roi in self.ROI_Checker.rois_in_case:
            if roi in self.associations:
                if self.associations[roi] == self.wanted_roi:
                    self.roi_name = roi
                    self.images_class = Dicom_to_Imagestack(Contour_Names=[roi])
                break


    def pre_process(self, images, annotations=None):
        self.roi_name = None
        self.ROI_Checker.get_rois_in_path(path)
        self.check_ROIs_In_Checker()
        if not self.roi_name:
            liver_input_path = os.path.join(liver_folder, self.ROI_Checker.ds.PatientID,
                                            self.ROI_Checker.ds.SeriesInstanceUID)
            liver_out_path = liver_input_path.replace('Input_3', 'Output')
            if os.path.exists(liver_out_path):
                files = [i for i in os.listdir(liver_out_path) if i.find('.dcm') != -1]
                for file in files:
                    self.ROI_Checker.get_rois_in_RS(os.path.join(liver_out_path, file))
                    self.check_ROIs_In_Checker()
                    if self.roi_name:
                        print('Previous liver contour found at ' + liver_out_path + '\nCopying over')
                        shutil.copy(os.path.join(liver_out_path, file), os.path.join(path, file))
                        break
            if not self.roi_name:
                print('No liver contour, passing to liver model')
                Copy_Folders(path, liver_input_path)
                # fid = open(os.path.join(liver_input_path,'Completed.txt'),'w+')
                # fid.close()
class Liver_Lobe_Segments_Processor(object):
    def __init__(self, mean_val, std_val, associations=None, wanted_roi='Liver'):
        self.wanted_roi = wanted_roi
        self.associations = associations

        self.Image_prep = Image_Clipping_and_Padding(mean_val=mean_val, std_val=std_val)
        self.ROI_Checker = Check_ROI_Names()
        self.images_class = None

    def check_ROIs_In_Checker(self):
        for roi in self.ROI_Checker.rois_in_case:
            if roi in self.associations:
                if self.associations[roi] == self.wanted_roi:
                    self.roi_name = roi
                    self.images_class = Dicom_to_Imagestack(Contour_Names=[roi])
                break

    def check_roi_path(self, path):
        self.ROI_Checker.get_rois_in_path(path)

    def pre_process(self, path, liver_folder):
        self.roi_name = None
        self.ROI_Checker.get_rois_in_path(path)
        self.check_ROIs_In_Checker()
        if not self.roi_name:
            liver_input_path = os.path.join(liver_folder, self.ROI_Checker.ds.PatientID,
                                            self.ROI_Checker.ds.SeriesInstanceUID)
            liver_out_path = liver_input_path.replace('Input_3', 'Output')
            if os.path.exists(liver_out_path):
                files = [i for i in os.listdir(liver_out_path) if i.find('.dcm') != -1]
                for file in files:
                    self.ROI_Checker.get_rois_in_RS(os.path.join(liver_out_path, file))
                    self.check_ROIs_In_Checker()
                    if self.roi_name:
                        print('Previous liver contour found at ' + liver_out_path + '\nCopying over')
                        shutil.copy(os.path.join(liver_out_path, file), os.path.join(path, file))
                        break
            if not self.roi_name:
                print('No liver contour, passing to liver model')
                Copy_Folders(path, liver_input_path)
                # fid = open(os.path.join(liver_input_path,'Completed.txt'),'w+')
                # fid.close()

    def process_images(self, image_class):
        image_class.get_mask([self.roi_name])
        x = image_class.ArrayDicom[..., 0]
        liver = image_class.mask
        self.og_liver = copy.deepcopy(liver)
        self.z_start, self.z_stop, self.r_start, self.r_stop, self.c_start, self.c_stop = get_bounding_box_indexes(
            self.og_liver)
        self.true_output = np.zeros([x.shape[0], 512, 512, 9])
        x = x[None, ..., None]
        x, self.liver = self.Image_prep.pad_images(x, liver)
        self.z_start_p, self.z_stop_p, self.r_start_p, self.r_stop_p, self.c_start_p, self.c_stop_p = get_bounding_box_indexes(
            self.liver)
        return x

    def post_process_images(self, pred):
        pred = np.squeeze(pred)
        liver = np.squeeze(self.liver)
        pred[liver == 0] = 0
        for i in range(1, pred.shape[-1]):
            pred[..., i] = remove_non_liver(pred[..., i])
        new_pred = self.Fill_Missing_Segments_Class.make_distance_map(pred, liver)
        self.true_output[self.z_start:self.z_stop, self.r_start:self.r_stop, self.c_start:self.c_stop,
        ...] = new_pred[self.z_start_p:self.z_stop_p, self.r_start_p:self.r_stop_p,
               self.c_start_p:self.c_stop_p, ...]
        return self.true_output
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def run_model(gpu=0):
    G = get_available_gpus()
    if len(G) == 1:
        gpu = 0
    with tf.device('/gpu:' + str(gpu)):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        K.set_session(sess)
        models_info = {}
        try:
            os.listdir('\\\\mymdafiles\\di_data1\\')
            morfeus_path = '\\\\mymdafiles\\di_data1\\'
            shared_drive_path = '\\\\mymdafiles\\ro-ADMIN\\SHARED\\Radiation physics\\BMAnderson\\Auto_Contour_Sites\\'
            raystation_drive_path = '\\\\mymdafiles\\ou-radonc\\Raystation\\Clinical\\Auto_Contour_Sites\\'
            model_load_path = os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Models')
        except:
            desktop_path = os.path.abspath(os.path.join('..','..','..'))
            morfeus_path = os.path.join(desktop_path)
            model_load_path = os.path.join(desktop_path,'Auto_Contour_Models')
            shared_drive_path = os.path.abspath(os.path.join('..','..','..','Shared_Drive','Auto_Contour_Sites'))
            raystation_drive_path = os.path.abspath(os.path.join('..','..','..','Raystation_LDrive','Clinical','Auto_Contour_Sites'))
        template_dir = os.path.join(shared_drive_path,'template_RS.dcm')
        base_dicom_reader = utils_BMA.Dicom_to_Imagestack(template_dir=template_dir, channels=1)
        model_info = {'model_path':os.path.join(model_load_path,'Pancreas','weights-improvement-v3_xception_512-12.hdf5'),
                      'names':['Pancreas_BMA_Program'],'vgg_model':[], 'image_size':512,
                      'path':[os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Pancreas_Auto_Contour','Input_3'),
                              os.path.join(shared_drive_path,'Pancreas_Auto_Contour','Input_3')],'is_CT':True,
                      'single_structure': True,'mean_val':0,'std_val':1,'vgg_normalize':True,'file_loader':base_dicom_reader,'post_process':partial(normalize_images,lower_threshold=-100,upper_threshold=300, is_CT=True, mean_val=0,std_val=1)}
        # models_info['pancreas'] = model_info
        # model_info = {'model_path':os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Models','Spleen','weights-improvement-10.hdf5'),
        #               'names':['Spleen_BMA_Program_4'],'vgg_model':[], 'image_size':512,
        #               'path':[os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Spleen_Auto_contour','Input_3'),
        #                       os.path.join(shared_drive_path, 'Spleen_Auto_Contour', 'Input_3')],'is_CT':True,
        #               'single_structure': True,'mean_val':40,'std_val':45,'vgg_normalize':False,'threshold':0.5}
        # models_info['spleen'] = model_info
        model_info = {'model_path':os.path.join(model_load_path,'Liver','weights-improvement-512_v3_model_xception-36.hdf5'),
                      'names':['Liver_BMA_Program_test'],'vgg_model':[], 'image_size':512,
                      'path':[
                          os.path.join(shared_drive_path,'Liver_Auto_Contour','Input_3'),
                          os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Liver_Auto_Contour','Input_3'),
                          os.path.join(raystation_drive_path,'Liver_Auto_Contour','Input_3')
                          #os.path.join(shared_drive_path, 'Liver_Auto_Contour', 'Input_3')
                              ],'three_channel':True,'is_CT':True,
                      'single_structure': True,'vgg_normalize':True,'threshold':0.5,'file_loader':base_dicom_reader,
                      'image_processor':[Normalize_Images(mean_val=0,std_val=1,lower_threshold=-100,upper_threshold=300, max_val=255)]}
        models_info['liver'] = model_info
        model_info = {'model_path':os.path.join(model_load_path,'Parotid','weights-improvement-best-parotid.hdf5'),
                      'names':['Parotid_R_BMA_Program_4','Parotid_L_BMA_Program_4'],'vgg_model':[], 'image_size':512,
                      'path':[#os.path.join(shared_drive_path,'Liver_Auto_Contour','Input_3')
                              os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Parotid_Auto_Contour','Input_3'),
                              os.path.join(raystation_drive_path,'Parotid_Auto_Contour','Input_3')
                              ],'three_channel':True,'is_CT':False,
                      'single_structure': True,'vgg_normalize':False,'threshold':0.4,'file_loader':base_dicom_reader,
                      'image_processor':[Normalize_Images(mean_val=176,std_val=58),Check_Size(512),Turn_Two_Class_Three()]}
        models_info['parotid'] = model_info
        # model_info = {'model_path':os.path.join(morfeus_path,'Morfeus','BMAnderson','CNN','Data','Data_Liver','Liver_Segments','weights-improvement-200.hdf5'),
        #               'names':['Liver_Segment_' + str(i) for i in range(1, 9)],'vgg_model':[], 'image_size':None,'three_channel':False,
        #               'path':[os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Liver_Segments_Auto_Contour','Input_3'),
        #                       os.path.join(raystation_drive_path,'Liver_Segments_Auto_Contour','Input_3')],'is_CT':True,
        #               'single_structure': True,'mean_val':80,'std_val':40,'vgg_normalize':False,'threshold':0.5,
        #               'pre_process':utils_BMA.Liver_Lobe_Segments_Processor(mean_val=80,std_val=40,associations={'Liver_BMA_Program_4':'Liver','Liver':'Liver'})}
        # models_info['liver_lobes'] = model_info
        model_info = {'model_path':r'C:\users\bmanderson\desktop\weights-improvement-best.hdf5',
                      'names':['Liver_BMA_Program_4_3DAtrous'],'vgg_model':[], 'image_size':512,
                      'path':[os.path.join(shared_drive_path,'Liver_Auto_Contour_3D','Input_3')
                              #os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Liver_Auto_Contour_3D','Input_3'),
                              #os.path.join(raystation_drive_path,'Liver_Auto_Contour_3D','Input_3')
                              ],'three_channel':False,'is_CT':True,'step':128,
                      'single_structure': True,'vgg_normalize':False,'threshold':0.5,'file_loader':base_dicom_reader,
                      'image_processor':[Normalize_Images(mean_val=80,std_val=42),Image_Clipping_and_Padding(),Make_3D(),Reduce_Prediction()]}
        # models_info['liver_3D'] = model_info


        all_sessions = {}
        resize_class_256 = Resize_Images_Keras(num_channels=3)
        resize_class_512 = Resize_Images_Keras(num_channels=3, image_size=512)
        graph1 = Graph()
        with graph1.as_default():
            gpu_options = GPUOptions(allow_growth=True)
            for key in models_info.keys():
                session1 = Session(config=ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with session1.as_default():
                    K.set_session(session1)
                    num_classes = int(1+len(models_info[key]['names']))
                    models_info[key]['vgg_model'] = VGG_Model_Pretrained(**models_info[key],num_classes=num_classes,
                                                                         gpu=gpu,graph1=graph1,session1=session1,
                                                                         Bilinear_model=BilinearUpsampling)
                    models_info[key]['predict_model'] = Predict_On_Models(**models_info[key],use_unet=False,
                                                                          num_classes=num_classes)
                    models_info[key]['resize_class_256'] = resize_class_256
                    models_info[key]['resize_class_512'] = resize_class_512
                    all_sessions[key] = session1

        running = True
        print('running')
        attempted = {}
        with graph1.as_default():
            while running:
                for key in models_info.keys():
                    with all_sessions[key].as_default():
                        K.set_session(all_sessions[key])
                        for path in models_info[key]['path']:
                            dicom_folder_all_out = down_folder(path,[])
                            for dicom_folder in dicom_folder_all_out:
                                true_outpath = None
                                print(dicom_folder)
                                if dicom_folder not in attempted.keys():
                                    attempted[dicom_folder] = 0
                                else:
                                    attempted[dicom_folder] += 1
                                print('running')
                                try:
                                    fid = open(os.path.join(dicom_folder,'running.txt'),'w+')
                                    fid.close()
                                    if 'pre_process' in models_info[key]:
                                        pre_processor = models_info[key]['pre_process']
                                        pre_processor.pre_process(dicom_folder,liver_folder = os.path.join(raystation_drive_path,'Liver_Auto_Contour','Input_3'))
                                        images_class = pre_processor.images_class
                                        if not pre_processor.roi_name:
                                            continue
                                    else: # 'file_loader' in models_info[key]
                                        images_class = models_info[key]['file_loader']
                                    images_class.make_array(dicom_folder, single_structure=models_info[key]['single_structure'])
                                    if 'pre_process' in models_info[key]:
                                        images = pre_processor.process_images(images_class)
                                        # images_class.use_template()
                                    else:
                                        images = images_class.ArrayDicom
                                    if 'image_processor' in models_info[key]:
                                        for processor in models_info[key]['image_processor']:
                                            images = processor.pre_process(images)
                                    output = os.path.join(path.split('Input_3')[0], 'Output')
                                    true_outpath = os.path.join(output,images_class.ds.PatientID,images_class.SeriesInstanceUID)
                                    if 'post_process' in models_info[key]:
                                        images = models_info[key]['post_process'](images)
                                    if 'pad' in models_info[key]:
                                        images = models_info[key]['pad'].process(images)

                                    models_info[key]['predict_model'].images = images
                                    k = time.time()
                                    models_info[key]['predict_model'].make_predictions()
                                    print('Prediction took ' + str(k-time.time()) + ' seconds')
                                    pred = models_info[key]['predict_model'].pred
                                    if 'image_processor' in models_info[key]:
                                        for processor in models_info[key]['image_processor']:
                                            images, pred = processor.post_process(images, pred)
                                    if 'pre_process' in models_info[key]:
                                        pred = pre_processor.post_process_images(pred)
                                    annotations = pred
                                    if 'pad' in models_info[key]:
                                        annotations = annotations[:-models_info[key]['pad'].z,...]
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
                                    if attempted[dicom_folder] <= 1:
                                        attempted[dicom_folder] += 1
                                        print('Failed once.. trying again')
                                        continue
                                    else:
                                        try:
                                            print('Failed twice')
                                            # utils_BMA.cleanout_folder(dicom_folder)
                                            if true_outpath is not None:
                                                if not os.path.exists(true_outpath):
                                                    os.makedirs(true_outpath)
                                                fid = open(os.path.join(true_outpath,'Failed.txt'),'w+')
                                                fid.close()
                                            print('had an issue')
                                        except:
                                            xxx = 1
                                        continue

if __name__ == '__main__':
    run_model(gpu=0)