import sys, shutil
from threading import Thread
from multiprocessing import cpu_count
from queue import *
from Image_Processing import *
from functools import partial
from Utils import cleanout_folder, weighted_categorical_crossentropy
from Utils import VGG_Model_Pretrained, Predict_On_Models, Resize_Images_Keras, K, plot_scroll_Image, down_folder
from tensorflow.compat.v1 import Graph, Session, ConfigProto, GPUOptions
from Bilinear_Dsc import BilinearUpsampling
import tensorflow as tf


class Copy_Files(object):
    def process(self, dicom_folder, local_folder, file):
        input_path = os.path.join(local_folder,file)
        while not os.path.exists(input_path):
            try:
                shutil.copy2(os.path.join(dicom_folder,file),input_path)
            except:
                print('Connection dropped...')
                if os.path.exists(input_path):
                    os.remove(input_path)
        return None


def worker_def(A):
    q = A[0]
    base_class = Copy_Files()
    while True:
        item = q.get()
        if item is None:
            break
        else:
            try:
                base_class.process(**item)
            except:
                print('Failed')
            q.task_done()

def find_base_dir():
    base_path = '.'
    for _ in range(20):
        if 'Morfeus' in os.listdir(base_path):
            break
        else:
            base_path = os.path.join(base_path,'..')
    return base_path


def run_model(gpu=0):
    with tf.device('/gpu:{}'.format(gpu)):
        gpu_options = GPUOptions(allow_growth=True)
        sess = Session(config=ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        # sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':0}, log_device_placement=False))
        K.set_session(sess)
        models_info = {}
        try:
            os.listdir('\\\\mymdafiles\\di_data1\\')
            morfeus_path = '\\\\mymdafiles\\di_data1\\'
            shared_drive_path = '\\\\mymdafiles\\ro-ADMIN\\SHARED\\Radiation physics\\BMAnderson\\Auto_Contour_Sites\\'
            raystation_drive_path = '\\\\mymdafiles\\ou-radonc\\Raystation\\Clinical\\Auto_Contour_Sites\\'
            model_load_path = os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Models')
        except:
            desktop_path = find_base_dir()
            morfeus_path = os.path.join(desktop_path)
            model_load_path = os.path.join(desktop_path,'Auto_Contour_Models')
            shared_drive_path = os.path.abspath(os.path.join(desktop_path,'Shared_Drive','Auto_Contour_Sites'))
            raystation_drive_path = os.path.abspath(os.path.join(desktop_path,'Raystation_LDrive','Clinical','Auto_Contour_Sites'))
        template_dir = os.path.join('.','Dicom_RT_and_Images_to_Mask','template_RS.dcm')
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
                          # os.path.join(morfeus_path, 'Morfeus', 'Test', 'Input_3')
                          os.path.join(shared_drive_path,'Liver_Auto_Contour','Input_3'),
                          os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Liver_Auto_Contour','Input_3'),
                          os.path.join(raystation_drive_path,'Liver_Auto_Contour','Input_3')
                              ],
                      'file_loader':base_dicom_reader,
                      'image_processor':[Normalize_Images(mean_val=0,std_val=1,lower_threshold=-100,upper_threshold=300, max_val=255),
                                         Threshold_Prediction(threshold=0.5, single_structure=True, is_liver=True),
                                         Expand_Dimension(axis=-1), Repeat_Channel(num_repeats=3,axis=-1),
                                         VGG_Normalize()]}
        models_info['liver'] = model_info
        model_info = {'model_path':os.path.join(model_load_path,'Parotid','weights-improvement-best-parotid.hdf5'),
                      'names':['Parotid_R_BMA_Program_4','Parotid_L_BMA_Program_4'],'vgg_model':[], 'image_size':512,
                      'path':[#os.path.join(shared_drive_path,'Liver_Auto_Contour','Input_3')
                              os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Parotid_Auto_Contour','Input_3'),
                              os.path.join(raystation_drive_path,'Parotid_Auto_Contour','Input_3')
                              ],
                      'vgg_normalize':False,'file_loader':base_dicom_reader,
                      'image_processor':[Normalize_Images(mean_val=176,std_val=58),Check_Size(512),
                                         Expand_Dimension(axis=-1), Repeat_Channel(num_repeats=3,axis=-1),Turn_Two_Class_Three(),
                                         Threshold_Prediction(threshold=0.4, single_structure=True)]}
        # models_info['parotid'] = model_info
        model_info = {'model_path':os.path.join(model_load_path,'Lungs'),
                      'initialize':True,
                      'names':['Lung (Left)_BMA_Program_0','Lung (Right)_BMA_Program_0'],'vgg_model':[], 'image_size':512,
                      'path':[
                          os.path.join(shared_drive_path,'Lungs_Auto_Contour','Input_3'),
                          os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Lungs','Input_3'),
                          os.path.join(raystation_drive_path,'Lungs_Auto_Contour','Input_3'),
                          # os.path.join(morfeus_path, 'Morfeus', 'BMAnderson', 'Test', 'Input_3')
                              ],
                      'file_loader':base_dicom_reader,
                      'image_processor':[Ensure_Image_Proportions(image_rows=512, image_cols=512),
                                         Normalize_Images(mean_val=-751,std_val=200),
                                         # Threshold_Images(lower_bound=-5, upper_bound=5),
                                         # ArgMax_Pred(),
                                         Threshold_Prediction(threshold=0.975, single_structure=True),
                                         Expand_Dimension(axis=-1), Repeat_Channel(num_repeats=3,axis=-1)
                                         ]}
        models_info['lungs'] = model_info
        model_info = {'model_path':os.path.join(model_load_path,'Liver_Lobes','weights-improvement-best.hdf5'),
                      'names':['Liver_Segment_{}_BMAProgram1'.format(i) for i in range(1, 9)],'vgg_model':[], 'image_size':None,'three_channel':False,
                      'path':[
                          os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Liver_Segments_Auto_Contour','Input_3'),
                          os.path.join(raystation_drive_path,'Liver_Segments_Auto_Contour','Input_3')
                      ],
                      'is_CT':True,
                      'single_structure': True,'mean_val':80,'std_val':40,'vgg_normalize':False,
                      'file_loader':Ensure_Liver_Segmentation(template_dir=template_dir,wanted_roi='Liver_BMA_Program_4',
                                                              liver_folder=os.path.join(raystation_drive_path,'Liver_Auto_Contour','Input_3'),
                                                              associations={'Liver_BMA_Program_4':'Liver_BMA_Program_4',
                                                                            'Liver':'Liver_BMA_Program_4'}),
                      'image_processor':[Normalize_to_Liver_Old(lower_fraction=0.5, upper_fraction=.9),
                                         Pad_Images(power_val_z=2**6,power_val_y=2**6,power_val_x=2**6), Expand_Dimension(axis=0),
                                         Threshold_Images(lower_bound=-14, upper_bound=14, final_scale_value=1),
                                         Mask_Prediction(9),
                                         Iterate_Lobe_Annotations()
                                         ],
                      'loss':partial(weighted_categorical_crossentropy),'loss_weights':[0.14,10,7.6,5.2,4.5,3.8,5.1,4.4,2.7]}
        models_info['liver_lobes'] = model_info
        model_info = {'model_path':os.path.join(model_load_path,'Liver_Disease_Ablation','weights-improvement-best_FWHM_AddedConv.hdf5'),
                      'names':['Liver_Disease_Ablation_BMA_Program_0'],'vgg_model':[],
                      'path':[
                          os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Liver_Disease_Ablation_Auto_Contour','Input_3'),
                          os.path.join(raystation_drive_path,'Liver_Disease_Ablation_Auto_Contour','Input_3')
                          #os.path.join(morfeus_path, 'Morfeus', 'BMAnderson','Test','Input_3')
                      ],
                      'file_loader':Ensure_Liver_Disease_Segmentation(template_dir=template_dir,wanted_roi='Liver_BMA_Program_4',
                                                              liver_folder=os.path.join(raystation_drive_path,'Liver_Auto_Contour','Input_3'),
                                                              associations={'Liver_BMA_Program_4':'Liver_BMA_Program_4',
                                                                            'Liver':'Liver_BMA_Program_4'}),
                      'image_processor':[Normalize_to_Liver(),
                                         Expand_Dimension(axis=0),
                                         Mask_Prediction(2), Threshold_and_Expand(0.9), Fill_Binary_Holes(),
                                         Minimum_Volume_and_Area_Prediction(min_volume=1, min_area=0.01, pred_axis=[1])]}
        # models_info['liver_disease'] = model_info
        all_sessions = {}
        resize_class_256 = Resize_Images_Keras(num_channels=3)
        resize_class_512 = Resize_Images_Keras(num_channels=3, image_size=512)
        graph1 = Graph()
        model_keys = ['liver_lobes','liver']
        with graph1.as_default():
            gpu_options = GPUOptions(allow_growth=True)
            for key in model_keys:
                session1 = Session(config=ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with session1.as_default():
                    K.set_session(session1)
                    models_info[key]['vgg_model'] = VGG_Model_Pretrained(**models_info[key],
                                                                         gpu=gpu,graph1=graph1,session1=session1,
                                                                         Bilinear_model=BilinearUpsampling)
                    models_info[key]['predict_model'] = Predict_On_Models(**models_info[key])
                    models_info[key]['resize_class_256'] = resize_class_256
                    models_info[key]['resize_class_512'] = resize_class_512
                    all_sessions[key] = session1

        running = True
        print('running')
        attempted = {}
        input_path = os.path.join('.','Input_Data')
        thread_count = int(cpu_count()*0.1+1)
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        q = Queue(maxsize=thread_count)
        A = [q,]
        with graph1.as_default():
            while running:
                for key in model_keys:
                    with all_sessions[key].as_default():
                        K.set_session(all_sessions[key])
                        if 'initialize' in models_info[key]:
                            started_up = False
                            if 'started_up' not in models_info[key]:
                                models_info[key]['started_up'] = False
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
                                    if 'initialize' in models_info[key]:
                                        if not models_info[key]['started_up']:
                                            all_sessions[key].run(tf.compat.v1.global_variables_initializer())
                                            models_info[key]['started_up'] = True
                                    images_class = models_info[key]['file_loader']
                                    cleanout_folder(input_path, empty_folder=False)
                                    threads = []
                                    for worker in range(thread_count):
                                        t = Thread(target=worker_def, args=(A,))
                                        t.start()
                                        threads.append(t)
                                    image_list = os.listdir(dicom_folder)
                                    for file in image_list:
                                        item = {'dicom_folder': dicom_folder, 'local_folder': input_path, 'file': file}
                                        q.put(item)
                                    for i in range(thread_count):
                                        q.put(None)
                                    for t in threads:
                                        t.join()
                                    images_class.process(input_path)
                                    if not images_class.return_status():
                                        continue
                                    images, ground_truth = images_class.pre_process()
                                    if images_class.reader.ds.PatientID.find('Radiopaedia') != -1:
                                        images = np.flip(images, axis=(1))
                                        images = Normalize_JPG_HU(True).normalize_function(images)
                                    images_class.reader.PathDicom = dicom_folder
                                    cleanout_folder(input_path, empty_folder=False)
                                    print('Got images')
                                    if 'image_processor' in models_info[key]:
                                        for processor in models_info[key]['image_processor']:
                                            print('Performing pre process {}'.format(processor))
                                            processor.get_niftii_info(images_class.dicom_handle)
                                            images, ground_truth = processor.pre_process(images, ground_truth)
                                    output = os.path.join(path.split('Input_')[0], 'Output')
                                    true_outpath = os.path.join(output,images_class.reader.ds.PatientID,images_class.reader.ds.SeriesInstanceUID)
                                    models_info[key]['predict_model'].images = images
                                    k = time.time()
                                    models_info[key]['predict_model'].make_predictions()
                                    print('Prediction took ' + str(time.time()-k) + ' seconds')
                                    pred = models_info[key]['predict_model'].pred
                                    images, pred, ground_truth = images_class.post_process(images, pred, ground_truth)
                                    print('Post Processing')
                                    if 'image_processor' in models_info[key]:
                                        for processor in models_info[key]['image_processor']:
                                            print('Performing post process {}'.format(processor))
                                            images, pred, ground_truth = processor.post_process(images, pred, ground_truth)
                                    annotations = pred
                                    if 'pad' in models_info[key]:
                                        annotations = annotations[:-models_info[key]['pad'].z,...]
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
                                            print('had an issue')
                                            fid = open(os.path.join(true_outpath, 'Failed.txt'), 'w+')
                                            fid.close()
                                        except:
                                            xxx = 1
                                        continue


if __name__ == '__main__':
    pass
