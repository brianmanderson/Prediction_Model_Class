import sys, shutil
from threading import Thread
from multiprocessing import cpu_count
from queue import *
from functools import partial
from Utils import cleanout_folder, weighted_categorical_crossentropy
from Utils import plot_scroll_Image, down_folder
from Bilinear_Dsc import BilinearUpsampling
from Image_Processing import *


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


def return_model_info(model_path, roi_names, dicom_paths, file_loader, model_predictor=Base_Predictor,
                      image_processors=[], prediction_processors=[], initialize=False, loss=None, loss_weights=None):
    '''
    :param model_path: path to model file
    :param roi_names: list of names for the predictions
    :param dicom_paths: list of paths that dicom is read and written to
    :param file_loader: the desired file loader
    :param model_predictor: the class for making predictions
    :param image_processors: list of image processors to occur before prediction, and occur in reverse after prediction
    :param prediction_processors: list of processors specifically for prediction
    :param initialize: True/False, only kicks in if model_path is a directory (TF2)
    :return:
    '''
    return {'model_path':model_path, 'names':roi_names, 'path':dicom_paths, 'file_loader':file_loader,
            'model_predictor':model_predictor, 'image_processors':image_processors, 'loss':loss,
            'loss_weights':loss_weights, 'prediction_processors':prediction_processors, 'initialize':initialize}


def run_model():
    with tf.device('/gpu:0'):
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                    gpu_options=gpu_options, log_device_placement=False))
        tf.compat.v1.keras.backend.set_session(sess)
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
        '''
        Liver Model
        '''
        liver_model = {'model_path':os.path.join(model_load_path,'Liver','weights-improvement-512_v3_model_xception-36.hdf5'),
                       'roi_names':['Liver_BMA_Program_4'],
                       'file_loader':base_dicom_reader,
                       'dicom_paths':[ #os.path.join(morfeus_path, 'Morfeus', 'BMAnderson', 'Test', 'Input_3'),
                          os.path.join(shared_drive_path,'Liver_Auto_Contour','Input_3'),
                          os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Liver_Auto_Contour','Input_3'),
                          os.path.join(raystation_drive_path,'Liver_Auto_Contour','Input_3')
                              ],
                       'image_processors':[Normalize_Images(mean_val=0,std_val=1,lower_threshold=-100,upper_threshold=300, max_val=255),
                            Expand_Dimension(axis=-1), Repeat_Channel(num_repeats=3,axis=-1),
                            Ensure_Image_Proportions(image_rows=512, image_cols=512),
                            VGG_Normalize()],
                       'prediction_processors': [Threshold_Prediction(threshold=0.5, single_structure=True,
                                                                      is_liver=True)]
                       }
        models_info['liver'] = return_model_info(**liver_model)
        '''
        Parotid Model
        '''
        partotid_model = {'model_path':os.path.join(model_load_path,'Parotid','whole_model'),
                      'roi_names':['Parotid_L_BMA_Program_4','Parotid_R_BMA_Program_4'],
                      'dicom_paths':[#os.path.join(shared_drive_path,'Liver_Auto_Contour','Input_3')
                              os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Parotid_Auto_Contour','Input_3'),
                              os.path.join(raystation_drive_path,'Parotid_Auto_Contour','Input_3')
                              ],
                      'file_loader':base_dicom_reader,
                      'image_processors':[Normalize_Parotid_MR(),
                                          Expand_Dimension(axis=-1), Repeat_Channel(num_repeats=3, axis=-1),
                                          Ensure_Image_Proportions(image_rows=256, image_cols=256),
                                          ],
                      'prediction_processors': [
                          # Turn_Two_Class_Three(),
                          Threshold_and_Expand(seed_threshold_value=0.9,
                                               lower_threshold_value=.5),
                          Fill_Binary_Holes()]
                      }
        models_info['parotid'] = return_model_info(**partotid_model)
        '''
        Lung Model
        '''

        lung_model = {'model_path':os.path.join(model_load_path,'Lungs', 'v3_model'),
                      'initialize':True,
                      'roi_names':['Lung (Left)_BMA_Program_1','Lung (Right)_BMA_Program_1'],
                      'dicom_paths':[
                          os.path.join(shared_drive_path,'Lungs_Auto_Contour','Input_3'),
                          os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Lungs','Input_3'),
                          os.path.join(raystation_drive_path,'Lungs_Auto_Contour','Input_3'),
                          # os.path.join(morfeus_path, 'Morfeus', 'BMAnderson', 'Test', 'Input_3')
                              ],
                      'file_loader':base_dicom_reader,
                      'image_processors':[
                          Normalize_Images(mean_val=-751,std_val=200),
                          Expand_Dimension(axis=-1), Repeat_Channel(num_repeats=3, axis=-1),
                          Ensure_Image_Proportions(image_rows=512, image_cols=512),
                                         ],
                      'prediction_processors':[ArgMax_Pred(),
                          Rename_Lung_Voxels(on_liver_lobes=False, max_iterations=1),
                          Threshold_Prediction(threshold=0.975, single_structure=True)]
                      }
        models_info['lungs'] = return_model_info(**lung_model)
        '''
        Liver Lobe Model
        '''
        liver_lobe_model = {'model_path':os.path.join(model_load_path,'Liver_Lobes','Model_372'),
                            'roi_names':['Liver_Segment_{}_BMAProgram2'.format(i) for i in range(1, 5)] + ['Liver_Segment_5-8_BMAProgram3'],
                            'dicom_paths': [
                                # os.path.join(morfeus_path, 'Morfeus', 'BMAnderson', 'Test', 'Input_3'),
                                os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Liver_Segments_Auto_Contour','Input_3'),
                                os.path.join(raystation_drive_path,'Liver_Segments_Auto_Contour','Input_3')
                            ],
                            'file_loader':Ensure_Liver_Segmentation(template_dir=template_dir,wanted_roi='Liver_BMA_Program_4',
                                                                    liver_folder=os.path.join(raystation_drive_path,'Liver_Auto_Contour','Input_3'),
                                                                    associations={'Liver_BMA_Program_4':'Liver_BMA_Program_4',
                                                                                  'Liver':'Liver_BMA_Program_4'}),
                            'image_processors': [Normalize_to_Liver_New(),
                                                Resample_Process([None, None, 5.0]),
                                                Box_Images(bbox=(0, 0, 0)),
                                                Pad_Images(power_val_z=64, power_val_x=320, power_val_y=384, min_val=0),
                                                Expand_Dimension(axis=0), Expand_Dimension(axis=-1),
                                                Threshold_Images(lower_bound=-5, upper_bound=5, final_scale_value=None,
                                                                 divide=True),
                                                Mask_Prediction_New()],
                            'prediction_processors': [
                                Threshold_and_Expand_New(seed_threshold_value=[.9, .9, .9, .9, .9],
                                                         lower_threshold_value=[0.5, 0.75, 0.25, 0.25, 0.75])
                            ]}
        lobe_model = return_model_info(**liver_lobe_model)
        lobe_model['loss'] = partial(weighted_categorical_crossentropy)
        lobe_model['loss_weights'] = [0.14,10,7.6,5.2,4.5,3.8,5.1,4.4,2.7]
        models_info['liver_lobes'] =lobe_model

        '''
        Disease Ablation Model
        '''
        model_info = {'model_path':os.path.join(model_load_path,'Liver_Disease_Ablation','model_88'),
                      'initialize':True,
                      'roi_names':['Liver_Disease_Ablation_BMA_Program_0'],
                      'dicom_paths':[
                          os.path.join(morfeus_path,'Morfeus','Auto_Contour_Sites','Liver_Disease_Ablation_Auto_Contour','Input_3'),
                          os.path.join(raystation_drive_path,'Liver_Disease_Ablation_Auto_Contour','Input_3')
                          #os.path.join(morfeus_path, 'Morfeus', 'BMAnderson','Test','Input_3')
                      ],
                      'file_loader':Ensure_Liver_Disease_Segmentation(template_dir=template_dir,
                                                                      wanted_roi='Liver_BMA_Program_4',
                                                                      liver_folder=os.path.join(raystation_drive_path,'Liver_Auto_Contour','Input_3'),
                                                                      associations={'Liver_BMA_Program_4':'Liver_BMA_Program_4',
                                                                                    'Liver':'Liver_BMA_Program_4'}),
                      'model_predictor':Predict_Disease,
                      'image_processors':[
                          Box_Images(),
                          Normalize_to_Liver(mirror_max=True),
                          Threshold_Images(lower_bound=-10, upper_bound=10, divide=True),
                          Resample_Process(desired_output_dim=[None, None, 1.0]),
                          Pad_Images(power_val_z=2 ** 3, power_val_y=2 ** 3, power_val_x=2 ** 3),
                          Expand_Dimension(axis=0), Expand_Dimension(axis=-1),
                          Mask_Prediction_New(),
                          Threshold_and_Expand(seed_threshold_value=0.63, lower_threshold_value=.25)
                                          ],
                      'prediction_processors':
                          [
                              Fill_Binary_Holes(), Mask_within_Liver(),
                              Minimum_Volume_and_Area_Prediction(min_volume=0.5)
                          ]
                      }
        models_info['liver_disease'] = return_model_info(**model_info)
        all_sessions = {}
        graph = tf.compat.v1.Graph()
        model_keys = ['liver_lobes', 'liver', 'lungs', 'parotid', 'liver_disease'] #liver_lobes
        # model_keys = ['liver_lobes']
        with graph.as_default():
            gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
            for key in model_keys:
                session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                    gpu_options=gpu_options, log_device_placement=False))
                with session.as_default():
                    tf.compat.v1.keras.backend.set_session(session)
                    model_info = models_info[key]
                    loss = model_info['loss']
                    loss_weights = model_info['loss_weights']
                    model_info['model_predictor'] = model_info['model_predictor'](model_info['model_path'], graph=graph,
                                                                                  session=session,
                                                                                  Bilinear_model=BilinearUpsampling,
                                                                                  loss=loss, loss_weights=loss_weights)
                    all_sessions[key] = session
        # g.finalize()
        running = True
        print('running')
        attempted = {}
        input_path = os.path.join('.','Input_Data')
        thread_count = int(cpu_count()*0.1+1)
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        q = Queue(maxsize=thread_count)
        A = [q,]
        while running:
            with graph.as_default():
                for key in model_keys:
                    with all_sessions[key].as_default():
                        tf.compat.v1.keras.backend.set_session(all_sessions[key])
                        if os.path.isdir(models_info[key]['model_path']) and 'started_up' not in models_info[key]:
                            models_info[key]['started_up'] = False
                        for path in models_info[key]['path']:
                            dicom_folder_all_out = down_folder(path,[])
                            for dicom_folder in dicom_folder_all_out:
                                if os.path.exists(os.path.join(dicom_folder,'..','Raystation_Export.txt')):
                                    os.remove(os.path.join(dicom_folder,'..','Raystation_Export.txt'))
                                true_outpath = None
                                print(dicom_folder)
                                if dicom_folder not in attempted.keys():
                                    attempted[dicom_folder] = 0
                                else:
                                    attempted[dicom_folder] += 1
                                try:
                                    if os.path.isdir(models_info[key]['model_path']) and not models_info[key]['started_up']:
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
                                    output = os.path.join(path.split('Input_')[0], 'Output')
                                    true_outpath = os.path.join(output,images_class.reader.ds.PatientID,images_class.reader.ds.SeriesInstanceUID)
                                    if not os.path.exists(true_outpath):
                                        os.makedirs(true_outpath)
                                    if not images_class.return_status():
                                        cleanout_folder(input_path, empty_folder=False)
                                        cleanout_folder(dicom_folder)
                                        fid = open(os.path.join(true_outpath, 'Failed.txt'), 'w+')
                                        fid.close()
                                        continue
                                    images, ground_truth = images_class.pre_process()
                                    images_class.reader.PathDicom = dicom_folder
                                    cleanout_folder(input_path, empty_folder=False)
                                    print('Got images')
                                    preprocessing_status = os.path.join(true_outpath, 'Status_Preprocessing.txt')
                                    predicting_status = os.path.join(true_outpath, 'Status_Predicting.txt')
                                    post_processing_status = os.path.join(true_outpath, 'Status_Postprocessing.txt')
                                    writing_status = os.path.join(true_outpath, 'Status_Writing RT Structure.txt')
                                    fid = open(preprocessing_status,'w+')
                                    fid.close()
                                    for processor in models_info[key]['image_processors']:
                                        print('Performing pre process {}'.format(processor))
                                        processor.get_niftii_info(images_class.dicom_handle)
                                        images, ground_truth = processor.pre_process(images, ground_truth)
                                    Model_Prediction = models_info[key]['model_predictor']
                                    k = time.time()
                                    os.remove(preprocessing_status)
                                    fid = open(predicting_status, 'w+')
                                    fid.close()
                                    pred = Model_Prediction.predict(images)
                                    os.remove(predicting_status)
                                    fid = open(post_processing_status, 'w+')
                                    fid.close()
                                    print('Prediction took ' + str(time.time()-k) + ' seconds')
                                    images, pred, ground_truth = images_class.post_process(images, pred, ground_truth)
                                    print('Post Processing')
                                    for processor in models_info[key]['image_processors'][::-1]: # In reverse now
                                        print('Performing post process {}'.format(processor))
                                        images, pred, ground_truth = processor.post_process(images, pred, ground_truth)
                                    for processor in models_info[key]['prediction_processors']:
                                        processor.get_niftii_info(images_class.dicom_handle)
                                        print('Performing prediction process {}'.format(processor))
                                        images, pred, ground_truth = processor.post_process(images, pred, ground_truth)
                                    os.remove(post_processing_status)
                                    fid = open(writing_status, 'w+')
                                    fid.close()
                                    annotations = pred
                                    images_class.reader.template = 1
                                    images_class.reader.with_annotations(annotations,true_outpath,
                                                                         ROI_Names=models_info[key]['names'])

                                    print('RT structure ' + images_class.reader.ds.PatientID + ' printed to ' + os.path.join(output,
                                          images_class.reader.ds.PatientID,images_class.reader.RS_struct.SeriesInstanceUID) + ' with name: RS_MRN'
                                          + images_class.reader.ds.PatientID + '.dcm')
                                    os.remove(writing_status)
                                    cleanout_folder(dicom_folder)
                                    if not os.listdir(os.path.join(dicom_folder,'..')):
                                        os.rmdir(os.path.join(dicom_folder,'..'))
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
                                            if not os.listdir(os.path.join(dicom_folder, '..')):
                                                os.rmdir(os.path.join(dicom_folder, '..'))
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
