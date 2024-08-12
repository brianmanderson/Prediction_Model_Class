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


def return_liver_ablation_3d_model(add_version=True):
    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    required_size = (32, 64, 64)
    ablation_3d_model = PredictWindowSliding(image_key='image', model_path=os.path.join(model_load_path,
                                                                                        'Liver_Ablation_3D',
                                                                                        'BasicUNet3D_Trial_15.hdf5'),
                                             model_template=BasicUnet3D(input_tensor=None,
                                                                        input_shape=required_size + (1,),
                                                                        classes=2, classifier_activation="softmax",
                                                                        activation="leakyrelu",
                                                                        normalization="batch", nb_blocks=3,
                                                                        nb_layers=5, dropout='standard',
                                                                        filters=32, dropout_rate=0.1,
                                                                        skip_type='concat',
                                                                        bottleneck='standard').get_net(),
                                             nb_label=2, required_size=required_size, sw_overlap=0.5, sw_batch_size=8,
                                             )
    paths = [
        os.path.join(shared_drive_path, 'Liver_Ablation_3D_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'Liver_Ablation_3D_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'Liver_Ablation_3D_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'Liver_Ablation_3D_Auto_Contour', 'Input_3')
    ]
    ablation_3d_model.set_paths(paths)
    ablation_3d_model.set_image_processors([
        Processors.DeepCopyKey(from_keys=('annotation',), to_keys=('og_annotation',)),
        Processors.Threshold_Images(image_keys=('image',), lower_bounds=(-1000,), upper_bounds=(1500,), divides=(False,)),
        ZNorm_By_Annotation(image_key='image', annotation_key='annotation'),
        Processors.Threshold_Images(image_keys=('image',), lower_bounds=(-3.55,), upper_bounds=(3.55,), divides=(False,)),
        Processors.Per_Image_MinMax_Normalization(image_keys=('image',), threshold_value=1.0),
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.Resampler(resample_keys=('image', 'annotation'),
                  resample_interpolators=('Linear', 'Nearest'),
                  desired_output_spacing=[None, None, 1],
                  post_process_resample_keys=('prediction',),
                  post_process_original_spacing_keys=('primary_handle',),
                  post_process_interpolators=('Linear',)),
        Box_Images(bounding_box_expansion=(0, 0, 0), image_keys=('image',),
                   annotation_key='annotation', wanted_vals_for_bbox=(1,),
                   power_val_z=required_size[0], power_val_r=required_size[1], power_val_c=required_size[2],
                   post_process_keys=('prediction',)),
        Processors.MaskOneBasedOnOther(guiding_keys=('annotation',),
                            changing_keys=('image',),
                            guiding_values=(0,),
                            mask_values=(0,)),
        Processors.ExpandDimensions(image_keys=('image',), axis=-1),
        Processors.ExpandDimensions(image_keys=('image',), axis=0),
        Processors.SqueezeDimensions(post_prediction_keys=('prediction',))
    ])

    ablation_3d_model.set_prediction_processors([
        Processors.ExpandDimensions(image_keys=('og_annotation',), axis=-1),
        Processors.MaskOneBasedOnOther(guiding_keys=('og_annotation',),
                            changing_keys=('prediction',),
                            guiding_values=(0,),
                            mask_values=(0,)),
        Duplicate_Prediction(prediction_key='prediction'),
        # ProcessPrediction(prediction_keys=('prediction',),
        #                   threshold={"1": 0.5},
        #                   connectivity={"1": False},
        #                   extract_main_comp={"1": False},
        #                   thread_count=1, dist={"1": None}, max_comp={"1": 2}, min_vol={"1": 5000}),
        Processors.Threshold_and_Expand(seed_threshold_value=[0.55, 0.50], lower_threshold_value=[.3, 0.5],
                             prediction_key='prediction'),
        Processors.Fill_Binary_Holes(prediction_key='prediction', dicom_handle_key='primary_handle'),
    ])

    if add_version:
        roi_names = [roi + '_MorfeusLab_v0' for roi in ["Disease", "Ablation"]]
    else:
        roi_names = ["Disease", "Ablation"]

    ablation_3d_model.set_dicom_reader(EnsureLiverPresent(wanted_roi='Liver_BMA_Program_4',
                                                          roi_names=roi_names,
                                                          liver_folder=os.path.join(raystation_clinical_path,
                                                                                    'Liver_Auto_Contour', 'Input_3'),
                                                          associations={'Liver_MorfeusLab_v0': 'Liver_BMA_Program_4',
                                                                        'Liver_BMA_Program_4': 'Liver_BMA_Program_4',
                                                                        'Liver': 'Liver_BMA_Program_4'}))

    return ablation_3d_model


if __name__ == '__main__':
    pass
