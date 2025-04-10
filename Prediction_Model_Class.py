import shutil
from threading import Thread
from multiprocessing import cpu_count
from queue import *
import time
from Utils import cleanout_folder, down_folder, BaseModelBuilder
from ModelProcessingCode.Parotid_Model import return_parotid_model, plot_scroll_Image, return_paths
from ModelProcessingCode.Prostate_Model import return_prostate_model
from ModelProcessingCode.Prostate_Node_Model import return_prostate_nodes_model
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
            'parotid': return_parotid_model(),
            'prostate': return_prostate_model(),
            'prostatenodes': return_prostate_nodes_model()
        }
        model_keys = ['prostatenodes', 'prostate']
        model_keys = ['prostate']
        for key in model_keys:
            model_info = models_info[key]
            model_info.build_model(model_name=key)

        # g.finalize()
        running = True
        print('running')
        attempted = {}
        local_path = return_paths()
        input_path = os.path.join(local_path, 'Input_Data')
        thread_count = int(cpu_count() * 0.1 + 1)
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        q = Queue(maxsize=thread_count)
        A = [q, ]
        while running:
            for key in model_keys:
                model_runner = models_info[key]
                model_runner: BaseModelBuilder
                for path in model_runner.paths:
                    if not os.path.exists(path):
                        continue
                    dicom_folder_all_out = down_folder(path, [])
                    for dicom_folder in dicom_folder_all_out:
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
                            input_features = {'input_path': input_path, 'dicom_folder': dicom_folder,
                                              'out_path': os.path.join(path.split('Input')[0], 'Output')}
                            model_runner.set_input_features(input_features)
                            model_runner.run_load_images()
                            true_outpath = model_runner.write_folder
                            print('Got images')
                            cleanout_folder(path_origin=input_path, dicom_dir=input_path, delete_folders=False)
                            model_runner.run_prediction()
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
                                    fid = open(os.path.join(true_outpath, 'Status_Failed.txt'), 'w+')
                                    fid.close()
                                except:
                                    xxx = 1
                                continue
            time.sleep(1)


if __name__ == '__main__':
    pass
