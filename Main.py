__author__ = 'Brian M Anderson'
# Created on 3/15/2020

import os, sys

gpu = 0  # Default
if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
print('\n\n\nRunning on {}\n\n\n'.format(gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
from Prediction_Model_Class import run_model
run_model()