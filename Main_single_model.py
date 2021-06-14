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

from Prediction_Model_Class import run_model_single
run_model_single(input_path=r'Z:\Morfeus\Bastien\Auto_seg\test\input', output_path=r'Z:\Morfeus\Bastien\Auto_seg\test\output', model_key='liver')