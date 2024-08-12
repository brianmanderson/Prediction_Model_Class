import os, sys
sys.path.insert(0, os.path.abspath('..'))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *
import shutil
from functools import partial


class EnsureLiverPresent(TemplateDicomReader):
    def __init__(self, roi_names=None, associations=None, wanted_roi='Liver', liver_folder=None):
        super(EnsureLiverPresent, self).__init__(associations=associations, roi_names=roi_names)
        self.wanted_roi = wanted_roi
        self.liver_folder = liver_folder
        self.reader = DicomReaderWriter(associations=self.associations, Contour_Names=[self.wanted_roi])

    def check_ROIs_In_Checker(self):
        self.roi_name = None
        for roi in self.reader.rois_in_case:
            if roi.lower() == self.wanted_roi.lower():
                self.roi_name = roi
                return None
        for roi in self.reader.rois_in_case:
            if roi in self.associations:
                if self.associations[roi] == self.wanted_roi.lower():
                    self.roi_name = roi
                    break

    def load_images(self, input_features):
        input_path = input_features['input_path']
        self.reader.__reset__()
        self.reader.walk_through_folders(input_path)
        self.check_ROIs_In_Checker()
        go = False
        if self.roi_name is None and go:
            liver_input_path = os.path.join(self.liver_folder, self.reader.ds.PatientID,
                                            self.reader.ds.SeriesInstanceUID)
            liver_out_path = liver_input_path.replace('Input_3', 'Output')
            if os.path.exists(liver_out_path):
                files = [i for i in os.listdir(liver_out_path) if i.find('.dcm') != -1]
                for file in files:
                    self.reader.lstRSFile = os.path.join(liver_out_path, file)
                    self.reader.get_rois_from_RT()
                    self.check_ROIs_In_Checker()
                    if self.roi_name:
                        print('Previous liver contour found at ' + liver_out_path + '\nCopying over')
                        shutil.copy(os.path.join(liver_out_path, file), os.path.join(input_path, file))
                        break
        if self.roi_name is None:
            self.status = False
            print('No liver contour found')
        if self.roi_name:
            self.reader.get_images_and_mask()
            input_features['image'] = self.reader.ArrayDicom
            input_features['primary_handle'] = self.reader.dicom_handle
            input_features['annotation'] = self.reader.mask
        return input_features


def return_psma_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    psma_model = ModelBuilderFromTemplate(image_key='image',
                                          model_path=os.path.join(model_load_path,
                                                                  'PSMA',
                                                                  'DLv3_model_Trial_0.hdf5'),
                                          model_template=deeplabv3plus(input_shape=(512, 512, 1),
                                                                       backbone="xception",
                                                                       classes=5, final_activation='softmax',
                                                                       windowopt_flag=True,
                                                                       normalization='batch', activation='relu',
                                                                       weights=None).Deeplabv3())
    paths = [
        os.path.join(shared_drive_path, 'PSMA_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'PSMA_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'PSMA_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'PSMA_Auto_Contour', 'Input_3')
    ]
    # TODO get spacing and clip by 60*3mm dist
    # see Clip_Images_By_Extension
    psma_model.set_paths(paths)
    psma_model.set_image_processors([
        Processors.DeepCopyKey(from_keys=('annotation',), to_keys=('og_annotation',)),
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.ExpandDimensions(axis=-1, image_keys=('image',)),
        Processors.Ensure_Image_Proportions(image_rows=512, image_cols=512, image_keys=('image',),
                                 post_process_keys=('image', 'prediction')),
    ])
    psma_model.set_prediction_processors([
        ProcessPrediction(prediction_keys=('prediction',),
                          threshold={"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
                          connectivity={"1": False, "2": False, "3": False, "4": False},
                          extract_main_comp={"1": False, "2": False, "3": False, "4": False},
                          dist={}, max_comp={}, min_vol={}, thread_count=4),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v3' for roi in
                     ['Bladder', 'Rectum', 'Iliac Veins', 'Iliac Arteries']]
    else:
        roi_names = ['Bladder', 'Rectum', 'Iliac Veins', 'Iliac Arteries']

    psma_model.set_dicom_reader(EnsureLiverPresent(wanted_roi='Femoral Heads',
                                                   roi_names=roi_names,
                                                   liver_folder=os.path.join(raystation_clinical_path,
                                                                             'FemHeads_Auto_Contour', 'Input_3'),
                                                   associations={'Femoral Heads_MorfeusLab_v0': 'Femoral Heads',
                                                                 'Femoral Heads': 'Femoral Heads'}))
    return psma_model
