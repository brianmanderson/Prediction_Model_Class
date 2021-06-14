import os, sys

gpu = 0  # Default
if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
print('\n\n\nRunning on {}\n\n\n'.format(gpu))

# GPU device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from Prediction_Model_Class import *


def run_model_single_graph(input_path, output_path, model_key):

    with tf.device('/gpu:0'):
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        tf.compat.v1.keras.backend.set_session(sess)

        models_info = {
            'liver': return_liver_model(),
            'lungs': return_lung_model(),
            'liver_lobes': return_liver_lobe_model(),
            'liver_disease': return_liver_disease_model(),
            'lacc': return_lacc_model(),
        }

        model_list = ['liver', 'lungs', 'liver_lobes', 'liver_disease', 'lacc']
        if not model_key in model_list:
            raise ValueError('model_key should be one of {}'.format(model_list))

        graph = tf.compat.v1.Graph()

        # Loading model

        with graph.as_default():
            gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

            session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with session.as_default():
                tf.compat.v1.keras.backend.set_session(session)
                model_info = models_info[model_key]
                model_info.build_model(model_name=model_key, graph=graph, session=session)

        with graph.as_default():
            with session.as_default():
                tf.compat.v1.keras.backend.set_session(session)
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
                input_features = model_runner.pre_process(input_features)
                os.remove(preprocessing_status)

                # Running prediction
                predicting_status = os.path.join(output_path, 'Status_Predicting.txt')
                fid = open(predicting_status, 'w+')
                fid.close()
                k = time.time()
                input_features = model_runner.predict(input_features)
                print('Prediction took ' + str(time.time() - k) + ' seconds')
                os.remove(predicting_status)
                post_processing_status = os.path.join(output_path, 'Status_Postprocessing.txt')

                # Running postprocessing
                fid = open(post_processing_status, 'w+')
                fid.close()
                input_features = model_runner.post_process(input_features)
                print('Post Processing')
                input_features = model_runner.prediction_process(input_features)
                os.remove(post_processing_status)

                # Create resulting RTstruct
                writing_status = os.path.join(output_path, 'Status_Writing RT Structure.txt')
                fid = open(writing_status, 'w+')
                fid.close()
                model_runner.write_predictions(input_features)
                print('RT structure ' + patientID + ' printed to ' + output_path + ' with name: RS_MRN' + patientID + '.dcm')
                os.remove(writing_status)


if __name__ == '__main__':
    run_model_single_graph(input_path=r'Z:\Morfeus\Bastien\Auto_seg\test\input',
                           output_path=r'Z:\Morfeus\Bastien\Auto_seg\test\output', model_key='lungs')