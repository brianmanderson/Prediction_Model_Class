import os
from skimage import draw
import numpy as np


def down_folder(input_path,output):
    files = []
    dirs = []
    for root, dirs, files in os.walk(input_path):
        break
    if 'Completed.txt' in files:
        output.append(input_path)
    for dir_val in dirs:
        output = down_folder(os.path.join(input_path,dir_val),output)
    return output


def cleanout_folder(path_origin, dicom_dir, delete_folders=True):
    files = []
    for _, _, files in os.walk(dicom_dir):
        break
    for file in files:
        os.remove(os.path.join(dicom_dir, file))
    while delete_folders and len(dicom_dir) > len(path_origin):
        if len(os.listdir(dicom_dir)) == 0:
            os.rmdir(dicom_dir)
        dicom_dir = os.path.abspath(os.path.join(dicom_dir, '..'))
    return None


def poly2mask(vertex_row_coords, vertex_col_coords):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, [512,512])
    mask = np.zeros([512,512], dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def main():
    pass


if __name__ == "__main__":
    main()
