import os, time
from tensorflow.python.client import device_lib
from Utils import weighted_categorical_crossentropy, cleanout_folder
from Utils import VGG_Model_Pretrained, Predict_On_Models, Resize_Images_Keras, K, plot_scroll_Image, down_folder
from Image_Processing import Normalize_Images, Expand_Dimension, Ensure_Liver_Segmentation, Check_Size, \
    Turn_Two_Class_Three, Image_Clipping_and_Padding, template_dicom_reader
from tensorflow import Graph, Session, ConfigProto, GPUOptions
from Bilinear_Dsc import BilinearUpsampling
from functools import partial
import tensorflow as tf
import numpy as np


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
        base_dicom_reader = template_dicom_reader(template_dir=template_dir,channels=1)
        model_info = {'model_path':os.path.join(model_load_path,'Pancreas','weights-improvement-v3_xception_512-12.hdf5'),
                      'names':['Pancreas_BMA_Program'],'vgg_model':[], 'image_size':512,
                      'path':[os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Pancreas_Auto_Contour','Input_3'),
                              os.path.join(shared_drive_path,'Pancreas_Auto_Contour','Input_3')],'is_CT':True,
                      'single_structure': True,'mean_val':0,'std_val':1,'vgg_normalize':True,'file_loader':base_dicom_reader}
        # models_info['pancreas'] = model_info
        model_info = {'model_path':os.path.join(model_load_path,'Liver','weights-improvement-512_v3_model_xception-36.hdf5'),
                      'names':['Liver_BMA_Program_4'],'vgg_model':[], 'image_size':512,
                      'path':[
                          os.path.join(shared_drive_path,'Liver_Auto_Contour','Input_3'),
                          os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Liver_Auto_Contour','Input_3'),
                          os.path.join(raystation_drive_path,'Liver_Auto_Contour','Input_3')
                          #os.path.join(shared_drive_path, 'Liver_Auto_Contour', 'Input_3')
                              ],'three_channel':True,'is_CT':True,
                      'single_structure': True,'vgg_normalize':True,'threshold':0.5,'file_loader':base_dicom_reader,
                      'image_processor':[Normalize_Images(mean_val=0,std_val=1,lower_threshold=-100,upper_threshold=300, max_val=255)]}
        # models_info['liver'] = model_info
        model_info = {'model_path':os.path.join(model_load_path,'Parotid','weights-improvement-best-parotid.hdf5'),
                      'names':['Parotid_R_BMA_Program_4','Parotid_L_BMA_Program_4'],'vgg_model':[], 'image_size':512,
                      'path':[#os.path.join(shared_drive_path,'Liver_Auto_Contour','Input_3')
                              os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Parotid_Auto_Contour','Input_3'),
                              os.path.join(raystation_drive_path,'Parotid_Auto_Contour','Input_3')
                              ],'three_channel':True,'is_CT':False,
                      'single_structure': True,'vgg_normalize':False,'threshold':0.4,'file_loader':base_dicom_reader,
                      'image_processor':[Normalize_Images(mean_val=176,std_val=58),Check_Size(512),Turn_Two_Class_Three()]}
        # models_info['parotid'] = model_info
        model_info = {'model_path':os.path.join(model_load_path,'Liver_Lobes','weights-improvement-best.hdf5'),
                      'names':['Liver_Segment_{}_BMAProgram0'.format(i) for i in range(1, 9)],'vgg_model':[], 'image_size':None,'three_channel':False,
                      'path':[os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Liver_Segments_Auto_Contour','Input_3'),
                              os.path.join(raystation_drive_path,'Liver_Segments_Auto_Contour','Input_3'),
                              os.path.join(morfeus_path,'Morfeus','bmanderson','test')],
                      'is_CT':True,
                      'single_structure': True,'mean_val':80,'std_val':40,'vgg_normalize':False,'threshold':0,
                      'file_loader':Ensure_Liver_Segmentation(template_dir=template_dir,
                                                              liver_folder=os.path.join(raystation_drive_path,'Liver_Auto_Contour','Input_3'),
                                                              associations={'Liver_BMA_Program_4':'Liver','Liver':'Liver'}),
                      'image_processor':[Normalize_Images(mean_val=97, std_val=53),
                                         Image_Clipping_and_Padding(layers=3, mask_output=True), Expand_Dimension(axis=0)],
                      'loss':partial(weighted_categorical_crossentropy),'loss_weights':[0.14,10,7.6,5.2,4.5,3.8,5.1,4.4,2.7]}
        models_info['liver_lobes'] = model_info
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
                                try:
                                    fid = open(os.path.join(dicom_folder,'running.txt'),'w+')
                                    fid.close()
                                    images_class = models_info[key]['file_loader']
                                    images_class.process(dicom_folder, single_structure=models_info[key]['single_structure'])
                                    if not images_class.return_status():
                                        continue
                                    images, ground_truth = images_class.pre_process()
                                    print('Got images')
                                    if 'image_processor' in models_info[key]:
                                        for processor in models_info[key]['image_processor']:
                                            images, ground_truth = processor.pre_process(images, ground_truth)
                                    output = os.path.join(path.split('Input_3')[0], 'Output')
                                    true_outpath = os.path.join(output,images_class.reader.ds.PatientID,images_class.reader.ds.SeriesInstanceUID)

                                    models_info[key]['predict_model'].images = images
                                    k = time.time()
                                    models_info[key]['predict_model'].make_predictions()
                                    print('Prediction took ' + str(time.time()-k) + ' seconds')
                                    pred = models_info[key]['predict_model'].pred
                                    images, pred, ground_truth = images_class.post_process(images, pred, ground_truth)
                                    if 'image_processor' in models_info[key]:
                                        for processor in models_info[key]['image_processor']:
                                            images, pred, ground_truth = processor.post_process(images, pred, ground_truth)
                                    annotations = pred
                                    if 'pad' in models_info[key]:
                                        annotations = annotations[:-models_info[key]['pad'].z,...]
                                    if 'threshold' in models_info[key].keys():
                                        images_class.define_threshold(models_info[key]['threshold'])
                                    images_class.reader.template = 1

                                    images_class.reader.with_annotations(annotations,true_outpath,
                                                                  ROI_Names=models_info[key]['names'])

                                    print('RT structure ' + images_class.reader.ds.PatientID + ' printed to ' + os.path.join(output,
                                          images_class.reader.ds.PatientID,images_class.reader.RS_struct.SeriesInstanceUID) + ' with name: RS_MRN'
                                          + images_class.reader.ds.PatientID + '.dcm')

                                    cleanout_folder(dicom_folder)
                                    attempted[dicom_folder] = -1
                                except:
                                    if attempted[dicom_folder] <= 1:
                                        attempted[dicom_folder] += 1
                                        print('Failed once.. trying again')
                                        continue
                                    else:
                                        try:
                                            print('Failed twice')
                                            cleanout_folder(dicom_folder)
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
