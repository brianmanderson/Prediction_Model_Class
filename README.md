# Prediction_Model_Class

A TensorFlow 2 inference server that runs multiple deep-learning auto-segmentation
models on a single workstation or DGX GPU. It watches input folders for incoming DICOM
image series, runs the enabled models, and writes the predicted contours back out as
DICOM RT structure sets, so a treatment-planning environment can drop studies in and pick
finished contours up.

## How it works

- `BaseModelBuilder` (in `Utils.py`) is the shared base class. Each anatomical model is a
  specialization configured by a `return_*_model()` builder in `ModelProcessingCode/`
  (liver, lung, prostate, LACC/cervix, PSMA, pancreas, femoral heads, etc.), which sets
  the model path, image pre-processors, prediction post-processors, and DICOM reader.
- `run_model()` in `Prediction_Model_Class.py` loads the selected models onto `/gpu:0`,
  then loops continuously over each model's watched folders. For each new DICOM series it
  copies the files locally, runs preprocess to predict to postprocess, converts the output
  to an RT structure via `DicomRTTool`, and writes status files (`Status_Predicting`,
  `Status_Failed`, etc.). Failed studies are retried before being marked failed.
- DICOM/RT and image handling come from Brian Anderson's companion repos, included here as
  submodules (`Dicom_RT_and_Images_to_Mask`, `Image_Processors_Module`, `Resample_Class`).

## Requirements

Trained model files and a private "networks" repository are **not** distributed here; the
server expects a folder containing one model per localization.

```
docker pull tensorflow/tensorflow:2.4.1-gpu
pip install -r requirements.txt
```

## Usage

Run the server, passing the GPU index to use (default 0):

```
python Main.py 0
```

`Main.py` sets `CUDA_VISIBLE_DEVICES` and calls `run_model()`. `Main_single_model.py`
runs a single model instead of the multi-model loop.

## Models available (private)

| Model Name | Author | Version | Description |
| --- | --- | --- | --- |
| Create_Liver_BMA | BMAnderson | _BMA_Program_4 | Liver |
| Create_Liver_Disease_Ablation_BMA | BMAnderson | _BMA_Program_0 | Liver_Disease_Ablation |
| Create_Liver_Lobe_Segments_BMA | BMAnderson | _BMAProgram3 | Liver_Segment_1..5-8 |
| Create_Lung_BMA | BMAnderson | _BMA_Program_2 | Ground Glass, Lung |
| Create_LACC_MorfeusLab / AIP_LACC_AI_contours | BRigaud | _MorfeusLab_v4 | Bladder, Rectum, Sigmoid, Vagina, Parametrium, Femur_Head_R/L, Kidney_R/L, SpinalCord, BowelSpace, Femoral Heads, Upper_Vagina_2.0cm, CTVp |
| AIP_LACC_3D_AI_contours | BRigaud | _MorfeusLab_v5 | Bladder, Rectum, Sigmoid, Vagina, Parametrium, Femur_Head_R/L, Kidney_R/L, SpinalCord, BowelSpace, Femoral Heads, Upper_Vagina_2.0cm, CTVp |
| AIP_LACC_AI_CTVN_contours | BRigaud | _MorfeusLab_v2 | CTVn, CTV_PAN, Nodal_CTV |
| AIP_LACC_AI_Duodenum_contours | BRigaud | _MorfeusLab_v2 | Duodenum |
| Create_Pancreas_DLv3_MorfeusLab_v0 | BRigaud | _MorfeusLab_v0 | Pancreas |
| Create_Cyst_HybridDLv3_MorfeusLab_v0 | BRigaud | _HybridDLv3_v0 | (need pancreas) Cyst |
| Create_Disease_Ablation_MorfeusLab_v0 | BRigaud | _MorfeusLab_v0 | Disease, Ablation |
| Create_PSMA_MorfeusLab | BRigaud | _MorfeusLab_v3 | (need femoral heads) Bladder, Rectum, Iliac Veins, Iliac Arteries |
| Create_PSMA_3D_MorfeusLab | BRigaud | _MorfeusLab_v4 | (need femoral heads) Bladder, Rectum, Iliac Veins, Iliac Arteries |
| Create_FemHeads_MorfeusLab | BRigaud | _MorfeusLab_v0 | Femoral Heads |
| Create_Liver_3D_MorfeusLab | BRigaud | _MorfeusLab_v0 | Liver |

## How to reference this work

This repository was used in the following published manuscripts:

- https://doi.org/10.1016/j.prro.2021.02.003

```
@article{anderson2021simple,
  title={Simple Python Module for Conversions Between DICOM Images and Radiation Therapy Structures, Masks, and Prediction Arrays},
  author={Anderson, Brian M and Wahid, Kareem A and Brock, Kristy K},
  journal={Practical radiation oncology}, volume={11}, number={3}, pages={226--229},
  year={2021}, publisher={Elsevier}
}
```

- https://doi.org/10.1016/j.adro.2020.04.023

```
@article{anderson2021automated,
  title={Automated contouring of contrast and noncontrast computed tomography liver images with fully convolutional networks},
  author={Anderson, Brian M and Lin, Ethan Y and Cardenas, Carlos E and Gress, Dustin A and Erwin, William D and Odisio, Bruno C and Koay, Eugene J and Brock, Kristy K},
  journal={Advances in radiation oncology}, volume={6}, number={1}, pages={100464},
  year={2021}, publisher={Elsevier}
}
```

- https://doi.org/10.1016/j.ijrobp.2020.10.038

```
@article{rigaud2021automatic,
  title={Automatic segmentation using deep learning to enable online dose optimization during adaptive radiation therapy of cervical cancer},
  author={Rigaud, Bastien and Anderson, Brian M and Zhiqian, H Yu and Gobeli, Maxime and Cazoulat, Guillaume and S{\"o}derberg, Jonas and Samuelsson, Elin and Lidberg, David and Ward, Christopher and Taku, Nicolette and others},
  journal={International Journal of Radiation Oncology* Biology* Physics}, volume={109}, number={4}, pages={1096--1110},
  year={2021}, publisher={Elsevier}
}
```
