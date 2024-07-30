import shutil
import os
import time


def copy_folder(source_folder, destination_folder):
    """
    Copy the source_folder and its subdirectories to the destination_folder.

    Parameters:
    source_folder (str): The path to the source folder.
    destination_folder (str): The path to the destination folder.
    """
    try:
        if not os.path.exists(source_folder):
            print(f"The source folder '{source_folder}' does not exist.")
            return

        # Create destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Copy the folder and its subdirectories
        shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)
        print(f"Copied '{source_folder}' to '{destination_folder}' successfully.")

    except Exception as e:
        print(f"Error copying folder: {e}")


def down_folder_input(input_path, output):
    files = []
    dirs = []
    for root, dirs, files in os.walk(input_path):
        break
    if 'Completed.txt' in files and 'Input' in input_path:
        output.append(input_path)
    for dir_val in dirs:
        output = down_folder_input(os.path.join(input_path,dir_val),output)
    return output


network_path = r'\\vscifs1\PhysicsQAdata\BMA\Predictions'
local_path = r'C:\Users\Markb\Modular_Projects\DICOM'
print("Running file copies")
while True:
    """
    First copy files from network to local for prediction
    """
    for segmentation_site in os.listdir(network_path):
        site_path = os.path.join(network_path, segmentation_site)
        if not os.path.isdir(site_path):
            continue
        input_path = os.path.join(site_path, 'Input')
        if os.path.exists(input_path):
            for series_uid in os.listdir(input_path):
                series_path = os.path.join(input_path, series_uid)
                series_files = os.listdir(series_path)
                if 'Completed.txt' in series_files:
                    out_path = os.path.join(local_path, segmentation_site, 'Input', series_uid)
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    for file in series_files:
                        if not file.endswith('.txt'):
                            shutil.copy2(os.path.join(series_path, file), os.path.join(out_path, file))
                    fid = open(os.path.join(out_path, 'Completed.txt'), 'w+')
                    fid.close()
                    for f in os.listdir(series_path):
                        os.remove(os.path.join(series_path, f))
                    os.rmdir(series_path)
    """
    Next copy over predictions back to network
    """
    for segmentation_site in os.listdir(local_path):
        site_path = os.path.join(local_path, segmentation_site)
        if not os.path.isdir(site_path):
            continue
        input_path = os.path.join(site_path, 'Output')
        if os.path.exists(input_path):
            for series_uid in os.listdir(input_path):
                series_path = os.path.join(input_path, series_uid)
                series_files = os.listdir(series_path)
                if 'Completed.txt' in series_files:
                    out_path = os.path.join(network_path, segmentation_site, 'Output', series_uid)
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    for file in series_files:
                        if not file.endswith('.txt'):
                            shutil.copy2(os.path.join(series_path, file), os.path.join(out_path, file))
                    fid = open(os.path.join(out_path, 'Completed.txt'), 'w+')
                    fid.close()
                    os.remove(os.path.join(series_path, 'Completed.txt'))
                    for f in os.listdir(series_path):
                        os.remove(os.path.join(series_path, f))
                    os.rmdir(series_path)
    time.sleep(5)
