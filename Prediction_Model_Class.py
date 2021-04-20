import sys, shutil
from threading import Thread
from multiprocessing import cpu_count
from queue import *
import time
from functools import partial
from Utils import cleanout_folder, weighted_categorical_crossentropy
from Utils import plot_scroll_Image, down_folder
from Bilinear_Dsc import BilinearUpsampling
from Image_Processing import template_dicom_reader, Ensure_Liver_Segmentation, Ensure_Liver_Disease_Segmentation, \
    Predict_Disease, Base_Predictor, Predict_Lobes
from Image_Processors_Module.src.Processors.MakeTFRecordProcessors import *


class Copy_Files(object):
    def process(self, dicom_folder, local_folder, file):
        input_path = os.path.join(local_folder, file)
        while not os.path.exists(input_path):
            try:
                shutil.copy2(os.path.join(dicom_folder, file), input_path)
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
            base_path = os.path.join(base_path, '..')
    return base_path


def return_model_info(model_path, dicom_paths, file_loader, model_predictor=Base_Predictor,
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
    return {'model_path': model_path, 'path': dicom_paths, 'file_loader': file_loader,
            'model_predictor': model_predictor, 'image_processors': image_processors, 'loss': loss,
            'loss_weights': loss_weights, 'prediction_processors': prediction_processors, 'initialize': initialize}


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
            raystation_clinical_path = '\\\\mymdafiles\\ou-radonc\\Raystation\\Clinical\\Auto_Contour_Sites\\'
            model_load_path = os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Models')
            raystation_research_path = '\\\\mymdafiles\\ou-radonc\\Raystation\\Research\\Auto_Contour_Sites\\'
        except:
            desktop_path = find_base_dir()
            morfeus_path = os.path.join(desktop_path)
            model_load_path = os.path.join(desktop_path, 'Auto_Contour_Models')
            shared_drive_path = os.path.abspath(os.path.join(desktop_path, 'Shared_Drive', 'Auto_Contour_Sites'))
            raystation_clinical_path = os.path.abspath(
                os.path.join(desktop_path, 'Raystation_LDrive', 'Clinical', 'Auto_Contour_Sites'))
            raystation_research_path = os.path.abspath(
                os.path.join(desktop_path, 'Raystation_LDrive', 'Research', 'Auto_Contour_Sites'))
        '''
        Liver Model
        '''
        liver_model = {
            'model_path': os.path.join(model_load_path, 'Liver', 'weights-improvement-512_v3_model_xception-36.hdf5'),
            'file_loader': template_dicom_reader(roi_names=['Liver_BMA_Program_4']),
            'dicom_paths': [
                # r'H:\AutoModels\Liver\Input_4',
                os.path.join(morfeus_path, 'Morfeus', 'BMAnderson', 'Test', 'Input_4'),
                os.path.join(shared_drive_path, 'Liver_Auto_Contour', 'Input_3'),
                os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Liver_Auto_Contour', 'Input_3'),
                os.path.join(raystation_clinical_path, 'Liver_Auto_Contour', 'Input_3'),
                os.path.join(raystation_research_path, 'Liver_Auto_Contour', 'Input_3')
            ],
            'image_processors': [
                Threshold_Images(image_keys=('image',), lower_bound=-100, upper_bound=300),
                AddByValues(image_keys=('image',), values=(100,)),
                DivideByValues(image_keys=('image', 'image'), values=(400, 1/255)),
                ExpandDimensions(axis=-1, image_keys=('image',)),
                RepeatChannel(num_repeats=3, axis=-1, image_keys=('image',)),
                Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
                                         post_process_keys=('image', 'prediction')),
                VGGNormalize(image_keys=('image',))],
            'prediction_processors': [Threshold_Prediction(threshold=0.5, single_structure=True,
                                                           is_liver=True, prediction_keys=('prediction',))]
            }
        models_info['liver'] = return_model_info(**liver_model)
        '''
        Parotid Model
        '''
        partotid_model = {'model_path': os.path.join(model_load_path, 'Parotid', 'whole_model'),
                          'roi_names': ['Parotid_L_BMA_Program_4', 'Parotid_R_BMA_Program_4'],
                          'dicom_paths': [  # os.path.join(shared_drive_path,'Liver_Auto_Contour','Input_3')
                              os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Parotid_Auto_Contour',
                                           'Input_3'),
                              os.path.join(raystation_clinical_path, 'Parotid_Auto_Contour', 'Input_3'),
                              os.path.join(raystation_research_path, 'Parotid_Auto_Contour', 'Input_3')
                          ],
                          'file_loader': template_dicom_reader(roi_names=None),
                          'image_processors': [NormalizeParotidMR(image_keys=('image',)),
                                               ExpandDimensions(axis=-1, image_keys=('image',)),
                                               RepeatChannel(num_repeats=3, axis=-1, image_keys=('image',)),
                                               Ensure_Image_Proportions(image_rows=256, image_cols=256,
                                                                        image_keys=('image',),
                                                                        post_process_keys=('image', 'prediction')),
                                               ],
                          'prediction_processors': [
                              # Turn_Two_Class_Three(),
                              Threshold_and_Expand(seed_threshold_value=0.9,
                                                   lower_threshold_value=.5),
                              Fill_Binary_Holes(prediction_key='prediction', dicom_handle_key='primary_handle')]
                          }
       # models_info['parotid'] = return_model_info(**partotid_model)
        '''
        Lung Model
        '''

        lung_model = {'model_path': os.path.join(model_load_path, 'Lungs', 'Covid_Four_Model_50'),
                      'initialize': True,
                      # 'roi_names': ['Ground Glass_BMA_Program_2', 'Lung_BMA_Program_2'],
                      'dicom_paths': [
                          # r'H:\AutoModels\Lung\Input_4',
                          os.path.join(shared_drive_path, 'Lungs_Auto_Contour', 'Input_3'),
                          os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites', 'Lungs', 'Input_3'),
                          os.path.join(raystation_clinical_path, 'Lungs_Auto_Contour', 'Input_3'),
                          os.path.join(raystation_research_path, 'Lungs_Auto_Contour', 'Input_3'),
                          os.path.join(morfeus_path, 'Morfeus', 'BMAnderson', 'Test', 'Input_3')
                      ],
                      'file_loader': template_dicom_reader(roi_names=['Ground Glass_BMA_Program_2',
                                                                      'Lung_BMA_Program_2']),
                      'image_processors': [
                          AddByValues(image_keys=('image',), values=(751,)),
                          DivideByValues(image_keys=('image',), values=(200,)),
                          Threshold_Images(image_keys=('image',), lower_bound=-5, upper_bound=5),
                          DivideByValues(image_keys=('image',), values=(5,)),
                          ExpandDimensions(axis=-1, image_keys=('image',)),
                          RepeatChannel(num_repeats=3, axis=-1, image_keys=('image',)),
                          Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
                                                   post_process_keys=('image', 'prediction')),
                      ],
                      'prediction_processors': [
                          ArgMax(image_keys=('prediction',)),
                          To_Categorical(num_classes=3, annotation_keys=('prediction',)),
                          Rename_Lung_Voxels_Ground_Glass(on_liver_lobes=False, max_iterations=1,
                                                          prediction_key='prediction',
                                                          dicom_handle_key='primary_handle')
                      ]
                      }
        models_info['lungs'] = return_model_info(**lung_model)
        '''
        Liver Lobe Model
        '''
        liver_lobe_model = {'model_path': os.path.join(model_load_path, 'Liver_Lobes', 'Model_397'),
                            #'roi_names': ,
                            'dicom_paths': [
                                # r'H:\AutoModels\Lobes\Input_4',
                                os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites',
                                             'Liver_Segments_Auto_Contour', 'Input_3'),
                                os.path.join(raystation_clinical_path, 'Liver_Segments_Auto_Contour', 'Input_3'),
                                os.path.join(raystation_research_path, 'Liver_Segments_Auto_Contour', 'Input_3'),
                            ],
                            'model_predictor': Predict_Lobes,
                            'file_loader': Ensure_Liver_Segmentation(wanted_roi='Liver_BMA_Program_4',
                                                                     liver_folder=os.path.join(raystation_clinical_path,
                                                                                               'Liver_Auto_Contour',
                                                                                               'Input_3'),
                                                                     associations={
                                                                         'Liver_BMA_Program_4': 'Liver_BMA_Program_4',
                                                                         'Liver': 'Liver_BMA_Program_4'},
                                                                     roi_names=['Liver_Segment_{}_BMAProgram3'.format(i)
                                                                                for i in range(1, 5)] +
                                                                               ['Liver_Segment_5-8_BMAProgram3']),
                            'image_processors': [Normalize_to_annotation(image_key='image', annotation_key='annotation',
                                                                         annotation_value_list=(1,)),
                                                 Ensure_Image_Proportions(image_rows=512, image_cols=512,
                                                                          image_keys=('image', 'annotation')),
                                                 CastData(image_keys=('image', 'annotation'),
                                                          dtypes=('float32', 'int')),
                                                 AddSpacing(spacing_handle_key='primary_handle'),
                                                 Resampler(resample_keys=('image', 'annotation'),
                                                           resample_interpolators=('Linear', 'Nearest'),
                                                           desired_output_spacing=[None, None, 5.0],
                                                           post_process_resample_keys=('image', 'annotation',
                                                                                       'prediction'),
                                                           post_process_original_spacing_keys=('image', 'image',
                                                                                               'image'),
                                                           post_process_interpolators=('Linear', 'Nearest', 'Linear')),
                                                 Box_Images(bounding_box_expansion=(10, 10, 10), image_key='image',
                                                            annotation_key='annotation', wanted_vals_for_bbox=(1,),
                                                            power_val_z=64, power_val_r=320, power_val_c=384,
                                                            post_process_keys=('image', 'annotation', 'prediction'),
                                                            pad_value=0),
                                                 ExpandDimensions(image_keys=('image', 'annotation'), axis=0),
                                                 ExpandDimensions(image_keys=('image', 'annotation'), axis=-1),
                                                 Threshold_Images(image_keys=('image',), lower_bound=-5,
                                                                  upper_bound=5),
                                                 DivideByValues(image_keys=('image',), values=(10,)),
                                                 MaskOneBasedOnOther(guiding_keys=('annotation',),
                                                                     changing_keys=('image',),
                                                                     guiding_values=(0,),
                                                                     mask_values=(0,)),
                                                 CreateTupleFromKeys(image_keys=('image', 'annotation'),
                                                                     output_key='combined'),
                                                 SqueezeDimensions(
                                                     post_prediction_keys=('image', 'annotation', 'prediction'))
                                                 ],
                            'prediction_processors': [
                                Threshold_and_Expand_New(seed_threshold_value=[.9, .9, .9, .9, .9],
                                                         lower_threshold_value=[.75, .9, .25, .2, .75])
                            ]}
        lobe_model = return_model_info(**liver_lobe_model)
        lobe_model['loss'] = partial(weighted_categorical_crossentropy)
        lobe_model['loss_weights'] = [0.14, 10, 7.6, 5.2, 4.5, 3.8, 5.1, 4.4, 2.7]
        models_info['liver_lobes'] = lobe_model

        '''
        Disease Ablation Model
        '''
        model_info = {'model_path':os.path.join(model_load_path, 'Liver_Disease_Ablation', 'Model_42'), # r'H:\Liver_Disease_Ablation\Keras\DenseNetNewMultiBatch\Models\Trial_ID_42\Model_42',
                      'initialize': True,
                      'dicom_paths': [
                          # r'H:\AutoModels\Disease\Input_4',
                          os.path.join(morfeus_path, 'Morfeus', 'Auto_Contour_Sites',
                                       'Liver_Disease_Ablation_Auto_Contour', 'Input_3'),
                          os.path.join(raystation_clinical_path, 'Liver_Disease_Ablation_Auto_Contour', 'Input_3'),
                          os.path.join(raystation_research_path, 'Liver_Disease_Ablation_Auto_Contour', 'Input_3'),
                          os.path.join(morfeus_path, 'Morfeus', 'BMAnderson', 'Test', 'Input_5')
                      ],
                      'file_loader': Ensure_Liver_Disease_Segmentation(wanted_roi='Liver_BMA_Program_4',
                                                                       roi_names=['Liver_Disease_Ablation_BMA_Program_0'],
                                                                       liver_folder=os.path.join(raystation_clinical_path,
                                                                                                 'Liver_Auto_Contour',
                                                                                                 'Input_3'),
                                                                       associations={
                                                                           'Liver_BMA_Program_4': 'Liver_BMA_Program_4',
                                                                           'Liver': 'Liver_BMA_Program_4'}),
                      'model_predictor': Predict_Disease,
                      'image_processors': [
                          Normalize_to_annotation(image_key='image', annotation_key='annotation',
                                                  annotation_value_list=(1,), mirror_max=True),
                          AddSpacing(spacing_handle_key='primary_handle'),
                          Resampler(resample_keys=('image', 'annotation'),
                                    resample_interpolators=('Linear', 'Nearest'),
                                    desired_output_spacing=[None, None, 1.0],
                                    post_process_resample_keys=('image', 'annotation', 'prediction'),
                                    post_process_original_spacing_keys=('image', 'image', 'image'),
                                    post_process_interpolators=('Linear', 'Nearest', 'Linear')),
                          Box_Images(bounding_box_expansion=(5, 20, 20), image_key='image',
                                     annotation_key='annotation', wanted_vals_for_bbox=(1,),
                                     power_val_z=2 ** 4, power_val_r=2 ** 5, power_val_c=2 ** 5),
                          Threshold_Images(lower_bound=-10, upper_bound=10, divide=True, image_keys=('image',)),
                          ExpandDimensions(image_keys=('image', 'annotation'), axis=0),
                          ExpandDimensions(image_keys=('image', 'annotation'), axis=-1),
                          MaskOneBasedOnOther(guiding_keys=('annotation',),
                                              changing_keys=('image',),
                                              guiding_values=(0,),
                                              mask_values=(0,)),
                          CombineKeys(image_keys=('image', 'annotation'), output_key='combined'),
                          SqueezeDimensions(post_prediction_keys=('image', 'annotation', 'prediction'))
                      ],
                      'prediction_processors':
                          [
                              Threshold_and_Expand(seed_threshold_value=0.55, lower_threshold_value=.3,
                                                   prediction_key='prediction'),
                              Fill_Binary_Holes(prediction_key='prediction', dicom_handle_key='primary_handle'),
                              MaskOneBasedOnOther(guiding_keys=('annotation',),
                                                  changing_keys=('prediction',),
                                                  guiding_values=(0,),
                                                  mask_values=(0,)),
                              MinimumVolumeandAreaPrediction(min_volume=0.25, prediction_key='prediction',
                                                             dicom_handle_key='primary_handle')
                          ]
                      }
        models_info['liver_disease'] = return_model_info(**model_info)
        all_sessions = {}
        graph = tf.compat.v1.Graph()
        model_keys = ['liver_lobes', 'liver', 'lungs', 'liver_disease']  # liver_lobes
        # model_keys = ['lungs']
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
        input_path = os.path.join('.', 'Input_Data')
        thread_count = int(cpu_count() * 0.1 + 1)
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        q = Queue(maxsize=thread_count)
        A = [q, ]
        while running:
            with graph.as_default():
                for key in model_keys:
                    with all_sessions[key].as_default():
                        tf.compat.v1.keras.backend.set_session(all_sessions[key])
                        if os.path.isdir(models_info[key]['model_path']) and 'started_up' not in models_info[key]:
                            models_info[key]['started_up'] = False
                        for path in models_info[key]['path']:
                            if not os.path.exists(path):
                                continue
                            dicom_folder_all_out = down_folder(path, [])
                            for dicom_folder in dicom_folder_all_out:
                                if os.path.exists(os.path.join(dicom_folder, '..', 'Raystation_Export.txt')):
                                    os.remove(os.path.join(dicom_folder, '..', 'Raystation_Export.txt'))
                                true_outpath = None
                                print(dicom_folder)
                                if dicom_folder not in attempted.keys():
                                    attempted[dicom_folder] = 0
                                else:
                                    attempted[dicom_folder] += 1
                                try:
                                    if os.path.isdir(models_info[key]['model_path']) and \
                                            not models_info[key]['started_up']:
                                        all_sessions[key].run(tf.compat.v1.global_variables_initializer())
                                        models_info[key]['started_up'] = True
                                    images_class = models_info[key]['file_loader']
                                    cleanout_folder(path_origin=input_path, dicom_dir=input_path, delete_folders=False)
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
                                    input_features = {'input_path': input_path, 'dicom_folder': dicom_folder}
                                    input_features = images_class.process(input_features)
                                    output = os.path.join(path.split('Input_')[0], 'Output')
                                    series_instances_dictionary = images_class.reader.series_instances_dictionary[0]
                                    series_instance_uid = series_instances_dictionary['SeriesInstanceUID']
                                    patientID = series_instances_dictionary['PatientID']
                                    true_outpath = os.path.join(output, patientID, series_instance_uid)
                                    input_features['out_path'] = true_outpath
                                    if not os.path.exists(true_outpath):
                                        os.makedirs(true_outpath)
                                    if not images_class.return_status():
                                        cleanout_folder(path_origin=input_path, dicom_dir=input_path,
                                                        delete_folders=False)
                                        fid = open(os.path.join(true_outpath, 'Failed.txt'), 'w+')
                                        fid.close()
                                        continue
                                    input_features = images_class.pre_process(input_features)
                                    images_class.reader.PathDicom = dicom_folder
                                    cleanout_folder(path_origin=input_path, dicom_dir=input_path, delete_folders=False)
                                    print('Got images')
                                    preprocessing_status = os.path.join(true_outpath, 'Status_Preprocessing.txt')
                                    predicting_status = os.path.join(true_outpath, 'Status_Predicting.txt')
                                    post_processing_status = os.path.join(true_outpath, 'Status_Postprocessing.txt')
                                    writing_status = os.path.join(true_outpath, 'Status_Writing RT Structure.txt')
                                    fid = open(preprocessing_status, 'w+')
                                    fid.close()
                                    for processor in models_info[key]['image_processors']:
                                        print('Performing pre process {}'.format(processor))
                                        # processor.get_niftii_info(images_class.dicom_handle)
                                        input_features = processor.pre_process(input_features)
                                    Model_Prediction = models_info[key]['model_predictor']
                                    k = time.time()
                                    os.remove(preprocessing_status)
                                    fid = open(predicting_status, 'w+')
                                    fid.close()
                                    input_features = Model_Prediction.predict(input_features)
                                    # np.save(os.path.join('.', 'pred.npy'), pred)
                                    # pred = np.load(os.path.join('.', 'pred.npy'))
                                    # return None
                                    os.remove(predicting_status)
                                    fid = open(post_processing_status, 'w+')
                                    fid.close()
                                    print('Prediction took ' + str(time.time() - k) + ' seconds')
                                    input_features = images_class.post_process(input_features)
                                    print('Post Processing')
                                    for processor in models_info[key]['image_processors'][::-1]:  # In reverse now
                                        print('Performing post process {}'.format(processor))
                                        input_features = processor.post_process(input_features)
                                    # np.save(os.path.join(dicom_folder, 'Raw_Pred.npy'), pred[..., 1])
                                    # os.remove(os.path.join(dicom_folder, 'Completed.txt'))
                                    # continue
                                    for processor in models_info[key]['prediction_processors']:
                                        print('Performing prediction process {}'.format(processor))
                                        input_features = processor.post_process(input_features)
                                    os.remove(post_processing_status)
                                    fid = open(writing_status, 'w+')
                                    fid.close()
                                    images_class.write_predicitons(input_features)
                                    print('RT structure ' + patientID + ' printed to ' +
                                          os.path.join(output, patientID, series_instance_uid) +
                                          ' with name: RS_MRN' + patientID + '.dcm')
                                    os.remove(writing_status)
                                    cleanout_folder(path_origin=path, dicom_dir=dicom_folder, delete_folders=True)
                                    attempted[dicom_folder] = -1
                                except:
                                    if attempted[dicom_folder] <= 1:
                                        attempted[dicom_folder] += 1
                                        print('Failed once.. trying again')
                                        continue
                                    else:
                                        try:
                                            print('Failed twice')
                                            cleanout_folder(path_origin=path, dicom_dir=dicom_folder,
                                                            delete_folders=True)
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
