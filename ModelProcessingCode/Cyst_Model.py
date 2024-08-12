import os, sys
sys.path.insert(0, os.path.abspath('..'))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
import Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Utils import *


class PredictCyst(ModelBuilderFromTemplate):

    def predict(self, input_features):
        x = input_features['combined']
        required_size = (32, 128, 128)
        step = (32, 128, 128)
        shift = (24, 32, 32)
        start = [0, 0, 0]

        if x[0].shape != (32, 128, 128, 2):
            pred_count = np.zeros(x[..., 0].shape[1:])
            pred = np.zeros(x[..., 0].shape[1:] + (2,))
            while start[0] < x[0].shape[0]:
                start[1] = 0
                while start[1] < x[0].shape[1]:
                    start[2] = 0
                    while start[2] < x[0].shape[2]:
                        image_cube, mask_cube = x[..., 0][:, start[0]:start[0] + step[0], start[1]:start[1] + step[1],
                                                start[2]:start[2] + step[2], ...], x[..., -1][:,
                                                                                   start[0]:start[0] + step[0],
                                                                                   start[1]:start[1] + step[1],
                                                                                   start[2]:start[2] + step[2], ...]
                        image_cube, mask_cube = image_cube[..., None], mask_cube[..., None]

                        remain_z, remain_r, remain_c = required_size[0] - image_cube.shape[1], required_size[1] - \
                                                       image_cube.shape[2], required_size[2] - image_cube.shape[3]
                        image_cube = np.pad(image_cube,
                                            [[0, 0], [floor(remain_z / 2), ceil(remain_z / 2)],
                                             [floor(remain_r / 2), ceil(remain_r / 2)],
                                             [floor(remain_c / 2), ceil(remain_c / 2)], [0, 0]],
                                            mode='reflect')
                        mask_cube = np.pad(mask_cube, [[0, 0], [floor(remain_z / 2), ceil(remain_z / 2)],
                                                       [floor(remain_r / 2), ceil(remain_r / 2)],
                                                       [floor(remain_c / 2), ceil(remain_c / 2)],
                                                       [0, 0]], mode='constant', constant_values=0)

                        pred_cube = self.model.predict([image_cube, mask_cube])
                        pred_cube = pred_cube[:, floor(remain_z / 2):step[0] - ceil(remain_z / 2),
                                    floor(remain_r / 2):step[1] - ceil(remain_r / 2),
                                    floor(remain_c / 2):step[2] - ceil(remain_c / 2), ...]

                        pred[start[0]:start[0] + step[0], start[1]:start[1] + step[1], start[2]:start[2] + step[2],
                        ...] += pred_cube[0, ...]
                        pred_count[start[0]:start[0] + step[0], start[1]:start[1] + step[1],
                        start[2]:start[2] + step[2], ...] += 1
                        start[2] += shift[2]
                    start[1] += shift[1]
                start[0] += shift[0]

            pred /= np.repeat(pred_count[..., None], repeats=2, axis=-1)
        else:
            image_cube, mask_cube = x[..., 0][..., None], x[..., -1][..., None]
            difference = image_cube.shape[1] % 32
            if difference != 0:
                image_cube = np.pad(image_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
                mask_cube = np.pad(mask_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
            pred_cube = self.model.predict([image_cube, mask_cube])
            pred = pred_cube[:, difference:, ...]
        input_features['prediction'] = np.squeeze(pred)
        return input_features


def return_cyst_model():
    # TODO Add connectivity to MaskOneBasedOnOther to keep Cyst that is outside Pancreas

    morfeus_path, model_load_path, shared_drive_path, raystation_clinical_path, raystation_research_path = return_paths()
    pancreas_cyst = PredictCyst(image_key='combined', model_path=os.path.join(model_load_path, 'Cyst',
                                                                              'HybridDLv3_model_Trial_62.hdf5'),
                                model_template=deeplabv3plus(nb_blocks=9, nb_layers=2,
                                                             backbone='mobilenetv2',
                                                             input_shape=(32, 128, 128, 1),
                                                             classes=2, final_activation='softmax',
                                                             activation='swish', normalization='group',
                                                             windowopt_flag=False, nb_output=3,
                                                             add_squeeze=True, add_mask=True,
                                                             dense_decoding=False,
                                                             transition_pool=False, ds_conv=True,
                                                             weights=os.path.join(model_load_path, 'Cyst',
                                                                                  'HybridDLv3_model_Trial_62.hdf5'),
                                                             ).HybridDeeplabv3())
    paths = [
        os.path.join(shared_drive_path, 'Cyst_Auto_Contour', 'Input_3'),
        os.path.join(morfeus_path, 'Auto_Contour_Sites', 'Cyst_Auto_Contour', 'Input_3'),
        os.path.join(raystation_clinical_path, 'Cyst_Auto_Contour', 'Input_3'),
        os.path.join(raystation_research_path, 'Cyst_Auto_Contour', 'Input_3')
    ]
    pancreas_cyst.set_paths(paths)
    pancreas_cyst.set_image_processors([
        Processors.DeepCopyKey(from_keys=('annotation',), to_keys=('og_annotation',)),
        Processors.Normalize_Images(keys=('image',), mean_values=(21.0,), std_values=(24.0,)),
        Processors.Threshold_Images(image_keys=('image',), lower_bounds=(-3.55,), upper_bounds=(3.55,), divides=(False,)),
        Processors.AddByValues(image_keys=('image',), values=(3.55,)),
        Processors.DivideByValues(image_keys=('image', 'image'), values=(7.10, 1 / 255)),
        Processors.AddSpacing(spacing_handle_key='primary_handle'),
        Processors.Resampler(resample_keys=('image', 'annotation'),
                  resample_interpolators=('Linear', 'Nearest'),
                  desired_output_spacing=[1.0, 1.0, 3.0],
                  post_process_resample_keys=('prediction',),
                  post_process_original_spacing_keys=('primary_handle',),
                  post_process_interpolators=('Linear',)),
        DilateBinary(image_keys=('annotation',), radius=(5,)),
        Box_Images(bounding_box_expansion=(5, 20, 20), image_keys=('image',),
                   annotation_key='annotation', wanted_vals_for_bbox=(1,),
                   power_val_z=2 ** 4, power_val_r=2 ** 5, power_val_c=2 ** 5),
        Processors.ExpandDimensions(image_keys=('image', 'annotation'), axis=0),
        Processors.ExpandDimensions(image_keys=('image', 'annotation'), axis=-1),
        Processors.CombineKeys(image_keys=('image', 'annotation'), output_key='combined'),
        Processors.SqueezeDimensions(post_prediction_keys=('image', 'annotation', 'prediction'))
    ])
    pancreas_cyst.set_dicom_reader(EnsureLiverPresent(wanted_roi='Pancreas_DLv3_v0',
                                                      roi_names=['Cyst_HybridDLv3_v0'],
                                                      liver_folder=os.path.join(morfeus_path, 'Bastien', 'RayStation',
                                                                                'Pancreas', 'Input_3'),
                                                      associations={'Pancreas_Ezgi': 'Pancreas_DLv3_v0',
                                                                    'Pancreas': 'Pancreas_DLv3_v0',
                                                                    'Pancreas_DLv3_v0': 'Pancreas_DLv3_v0',
                                                                    'Pancreas_MONAI_v0': 'Pancreas_DLv3_v0',
                                                                    'Pancreas_RSDL_v0': 'Pancreas_DLv3_v0',
                                                                    }))
    pancreas_cyst.set_prediction_processors([
        Processors.Threshold_and_Expand(seed_threshold_value=0.55, lower_threshold_value=.3, prediction_key='prediction'),
        Processors.ExpandDimensions(image_keys=('og_annotation',), axis=-1),
        Processors.MaskOneBasedOnOther(guiding_keys=('og_annotation',), changing_keys=('prediction',),
                            guiding_values=(0,), mask_values=(0,)),
    ])
    return pancreas_cyst


if __name__ == '__main__':
    pass
