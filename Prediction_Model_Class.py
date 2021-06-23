import shutil
from threading import Thread
from multiprocessing import cpu_count
from queue import *
import time
from Utils import cleanout_folder, down_folder
from Image_Processing import return_liver_model, return_lung_model, return_liver_lobe_model, \
    return_liver_disease_model, plot_scroll_Image, return_lacc_model, return_pancreas_model
from Image_Processors_Module.src.Processors.MakeTFRecordProcessors import *
import tensorflow as tf


def copy_files(A, q, dicom_folder, input_path, thread_count):
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


def copy_file(dicom_folder, local_folder, file):
    input_path = os.path.join(local_folder, file)
    while not os.path.exists(input_path):
        try:
            shutil.copy2(os.path.join(dicom_folder, file), input_path)
        except:
            print('Connection dropped...')
            if os.path.exists(input_path):
                os.remove(input_path)
    return None


class CopyFiles(object):
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
    while True:
        item = q.get()
        if item is None:
            break
        else:
            try:
                copy_file(**item)
            except:
                print('Failed')
            q.task_done()


def run_model():
    with tf.device('/gpu:0'):
        models_info = {
            'liver': return_liver_model(),
            'lungs': return_lung_model(),
            'liver_lobes': return_liver_lobe_model(),
            'liver_disease': return_liver_disease_model(),
            'lacc': return_lacc_model(),
            'pancreas': return_pancreas_model(),
        }

        model_keys = ['liver_lobes', 'liver', 'lungs', 'liver_disease', 'lacc', 'pancreas']

        for key in model_keys:
                model_info = models_info[key]
                model_info.build_model(model_name=key)

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
            for key in model_keys:
                model_runner = models_info[key]
                for path in model_runner.paths:
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
                            cleanout_folder(path_origin=input_path, dicom_dir=input_path, delete_folders=False)
                            copy_files(q=q, A=A, dicom_folder=dicom_folder, input_path=input_path,
                                       thread_count=thread_count)
                            input_features = {'input_path': input_path, 'dicom_folder': dicom_folder}
                            input_features = model_runner.load_images(input_features)
                            print('Got images')
                            output = os.path.join(path.split('Input_')[0], 'Output')
                            series_instances_dictionary = model_runner.return_series_instance_dictionary()
                            series_instance_uid = series_instances_dictionary['SeriesInstanceUID']
                            patientID = series_instances_dictionary['PatientID']
                            true_outpath = os.path.join(output, patientID, series_instance_uid)
                            input_features['out_path'] = true_outpath
                            preprocessing_status = os.path.join(true_outpath, 'Status_Preprocessing.txt')
                            if not os.path.exists(true_outpath):
                                os.makedirs(true_outpath)
                            if not model_runner.return_status():
                                cleanout_folder(path_origin=input_path, dicom_dir=input_path,
                                                delete_folders=False)
                                fid = open(os.path.join(true_outpath, 'Failed.txt'), 'w+')
                                fid.close()
                                continue
                            fid = open(preprocessing_status, 'w+')
                            fid.close()
                            time_flag = time.time()
                            input_features = model_runner.pre_process(input_features)
                            print('Comp. time: pre_process {} seconds'.format(time.time()-time_flag))
                            os.remove(preprocessing_status)
                            cleanout_folder(path_origin=input_path, dicom_dir=input_path, delete_folders=False)
                            predicting_status = os.path.join(true_outpath, 'Status_Predicting.txt')
                            fid = open(predicting_status, 'w+')
                            fid.close()
                            time_flag = time.time()
                            input_features = model_runner.predict(input_features)
                            print('Comp. time: predict {} seconds'.format(time.time()-time_flag))
                            os.remove(predicting_status)
                            post_processing_status = os.path.join(true_outpath, 'Status_Postprocessing.txt')

                            fid = open(post_processing_status, 'w+')
                            fid.close()
                            time_flag = time.time()
                            print('Post Processing')
                            time_flag = time.time()
                            input_features = model_runner.post_process(input_features)
                            print('Comp. time: post_process {} seconds'.format(time.time()-time_flag))
                            time_flag = time.time()
                            input_features = model_runner.prediction_process(input_features)
                            print('Comp. time: prediction_process {} seconds'.format(time.time()-time_flag))
                            os.remove(post_processing_status)

                            writing_status = os.path.join(true_outpath, 'Status_Writing RT Structure.txt')
                            fid = open(writing_status, 'w+')
                            fid.close()
                            time_flag = time.time()
                            model_runner.write_predictions(input_features)
                            print('Comp. time: write_predictions {} seconds'.format(time.time()-time_flag))
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
            time.sleep(1)



def run_model_single(input_path, output_path, model_key):

    with tf.device('/gpu:0'):
        models_info = {
            'liver': return_liver_model(),
            'lungs': return_lung_model(),
            'liver_lobes': return_liver_lobe_model(),
            'liver_disease': return_liver_disease_model(),
            'lacc': return_lacc_model(),
            'pancreas': return_pancreas_model(),
        }

        model_list = ['liver', 'lungs', 'liver_lobes', 'liver_disease', 'lacc', 'pancreas']
        if not model_key in model_list:
            raise ValueError('model_key should be one of {}'.format(model_list))

        # Loading model
        model_info = models_info[model_key]
        model_info.build_model(model_name=model_key)

        model_runner = models_info[model_key]

        print('running {}'.format(model_key))
        input_features = {'input_path': input_path}
        input_features['out_path'] = output_path

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Loading images
        input_features = model_runner.load_images(input_features)
        print('Got images')
        series_instances_dictionary = model_runner.return_series_instance_dictionary()
        patientID = series_instances_dictionary['PatientID']

        # Running preprocessing
        preprocessing_status = os.path.join(output_path, 'Status_Preprocessing.txt')
        fid = open(preprocessing_status, 'w+')
        fid.close()
        time_flag = time.time()
        input_features = model_runner.pre_process(input_features)
        print('Comp. time: pre_process {} seconds'.format(time.time() - time_flag))
        os.remove(preprocessing_status)

        # Running prediction
        predicting_status = os.path.join(output_path, 'Status_Predicting.txt')
        fid = open(predicting_status, 'w+')
        fid.close()
        time_flag = time.time()
        input_features = model_runner.predict(input_features)
        print('Comp. time: predict {} seconds'.format(time.time() - time_flag))
        os.remove(predicting_status)
        post_processing_status = os.path.join(output_path, 'Status_Postprocessing.txt')

        # Running postprocessing
        fid = open(post_processing_status, 'w+')
        fid.close()
        print('Post Processing')
        time_flag = time.time()
        input_features = model_runner.post_process(input_features)
        print('Comp. time: post_process {} seconds'.format(time.time() - time_flag))
        time_flag = time.time()
        input_features = model_runner.prediction_process(input_features)
        print('Comp. time: prediction_process {} seconds'.format(time.time() - time_flag))
        os.remove(post_processing_status)

        # Create resulting RTstruct
        writing_status = os.path.join(output_path, 'Status_Writing RT Structure.txt')
        fid = open(writing_status, 'w+')
        fid.close()
        time_flag = time.time()
        model_runner.write_predictions(input_features)
        print('Comp. time: write_predictions {} seconds'.format(time.time() - time_flag))
        print('RT structure ' + patientID + ' printed to ' + output_path + ' with name: RS_MRN' + patientID + '.dcm')
        os.remove(writing_status)


if __name__ == '__main__':
    pass
