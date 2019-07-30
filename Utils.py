import os
import dicom
from dicom.tag import Tag
import copy
from skimage import draw
from skimage.measure import label,regionprops,find_contours
import numpy as np
import TensorflowUtils as utils
from TensorflowUtils import plot_scroll_Image, plot_Image


def cleanout_folder(dicom_dir):
    files = []
    for _, _, files in os.walk(dicom_dir):
        break
    for file in files:
        os.remove(os.path.join(dicom_dir,file))
    return None

class Dicom_to_Imagestack:
    def __init__(self,delete_previous_rois=True, theshold=0.5,Contour_Names=None, template_dir=None):
        self.template_dir = template_dir
        self.delete_previous_rois = delete_previous_rois
        self.theshold = theshold
        self.Contour_Names = Contour_Names
        self.associations = {}

    def make_array(self,dir_to_dicom, single_structure=True):
        self.single_structure = single_structure
        self.dir_to_dicom = dir_to_dicom
        self.lstFilesDCM = []
        self.Dicom_info = {}
        self.lstRSFile = []
        i = 0
        fileList = []
        for dirName, _, fileList in os.walk(self.dir_to_dicom):
            break
        for filename in fileList:
            print(str(round(i/len(fileList)*100,2))+ '% done loading')
            i += 1
            try:
                ds = dicom.read_file(os.path.join(dirName,filename))
                if ds.Modality != 'RTSTRUCT':  # check whether the file's DICOM
                    self.lstFilesDCM.append(os.path.join(dirName, filename))
                    self.Dicom_info[os.path.join(dirName, filename)] = ds
                    self.SeriesInstanceUID = ds.SeriesInstanceUID
                elif ".dcm" in filename.lower() and ds.Modality == 'RTSTRUCT':
                    self.lstRSFile = os.path.join(dirName, filename)
            except:
                continue
        self.num_images = len(self.lstFilesDCM)
        self.get_images_and_mask()

    def get_mask(self):
        self.hierarchy = {}
        for roi in self.Contour_Names:
            if roi not in self.associations:
                self.associations[roi] = roi
        self.RefDs = dicom.read_file(self.lstFilesDCM[0])
        self.RS_struct = dicom.read_file(self.lstRSFile)
        if Tag((0x3006, 0x020)) in self.RS_struct.keys():
            self.ROI_Structure = self.RS_struct.StructureSetROISequence
        else:
            self.ROI_Structure = []
        self.mask = np.zeros([self.ArrayDicom.shape[0], self.ArrayDicom.shape[1], len(self.lstFilesDCM), len(self.Contour_Names)],
                             dtype='float32')

        self.structure_references = {}
        for contour_number in range(len(self.RS_struct.ROIContourSequence)):
            self.structure_references[self.RS_struct.ROIContourSequence[contour_number].ReferencedROINumber] = contour_number
        found_rois = {}
        for roi in self.Contour_Names:
            found_rois[roi] = {'Hierarchy':999,'Name':[],'Roi_Number':0}
        for Structures in self.ROI_Structure:
            ROI_Name = Structures.ROIName.lower()
            if Structures.ROINumber not in self.structure_references.keys():
                continue
            true_name = None
            if ROI_Name in self.associations:
                true_name = self.associations[ROI_Name]
            elif ROI_Name.lower() in self.associations:
                true_name = self.associations[ROI_Name.lower()]
            if true_name and true_name in self.Contour_Names:
                if true_name in self.hierarchy.keys():
                    for roi in self.hierarchy[true_name]:
                        if roi == ROI_Name:
                            index_val = self.hierarchy[true_name].index(roi)
                            if index_val < found_rois[true_name]['Hierarchy']:
                                found_rois[true_name]['Hierarchy'] = index_val
                                found_rois[true_name]['Name'] = ROI_Name
                                found_rois[true_name]['Roi_Number'] = Structures.ROINumber
                else:
                    found_rois[true_name] = {'Hierarchy':999,'Name':ROI_Name,'Roi_Number':Structures.ROINumber}
        for ROI_Name in found_rois.keys():
            if found_rois[ROI_Name]['Roi_Number'] in self.structure_references:
                index = self.structure_references[found_rois[ROI_Name]['Roi_Number']]
                mask = self.get_mask_for_contour(index)
                self.mask[...,self.Contour_Names.index(ROI_Name)][mask == 1] = 1
        return None

    def get_mask_for_contour(self,i):
        self.Liver_Locations = self.RS_struct.ROIContourSequence[i].ContourSequence
        self.Liver_Slices = []
        for contours in self.Liver_Locations:
            data_point = contours.ContourData[2]
            if data_point not in self.Liver_Slices:
                self.Liver_Slices.append(contours.ContourData[2])
        return self.Contours_to_mask()

    def Contours_to_mask(self):
        mask = np.zeros([self.ArrayDicom.shape[0], self.ArrayDicom.shape[1], len(self.lstFilesDCM)], dtype='float32')
        Contour_data = self.Liver_Locations
        ShiftCols = self.RefDs.ImagePositionPatient[0]
        ShiftRows = self.RefDs.ImagePositionPatient[1]
        PixelSize = self.RefDs.PixelSpacing[0]
        Mag = 1 / PixelSize
        mult1 = mult2 = 1
        if ShiftCols > 0:
            mult1 = -1
        if ShiftRows > 0:
            print('take a look at this one...')
        #    mult2 = -1

        for i in range(len(Contour_data)):
            slice_val = round(Contour_data[i].ContourData[2],2)
            dif = [abs(i - slice_val) for i in self.slice_info]
            slice_index = dif.index(min(dif))  # Now we know which slice to alter in the mask file
            cols = Contour_data[i].ContourData[1::3]
            rows = Contour_data[i].ContourData[0::3]
            col_val = [Mag * abs(x - mult1 * ShiftRows) for x in cols]
            row_val = [Mag * abs(x - mult2 * ShiftCols) for x in rows]
            temp_mask = self.poly2mask(col_val, row_val, [self.ArrayDicom.shape[0], self.ArrayDicom.shape[1]])
            mask[:,:,slice_index][temp_mask > 0] = 1
            #scm.imsave('C:\\Users\\bmanderson\\desktop\\images\\mask_'+str(i)+'.png',mask_slice)

        return mask

    def get_images_and_mask(self):
        self.slice_info = np.zeros([len(self.lstFilesDCM)])
        # Working on the RS structure now
        if self.lstRSFile:
            self.RS_struct = dicom.read_file(self.lstRSFile)
            self.template = False
            try:
                x = self.RS_struct.ROIContourSequence
            except:
                self.template = True
        else:
            self.template = True
        if self.template:
            if not self.template_dir:
                self.template_dir = os.path.join('\\\\mymdafiles','ro-admin','SHARED','Radiation physics','BMAnderson','Auto_Contour_Sites','template_RS.dcm')
                if not os.path.exists(self.template_dir):
                    self.template_dir = os.path.join('..','..','Shared_Drive','Auto_Contour_Sites','template_RS.dcm')
            self.key_list = self.template_dir.replace('template_RS.dcm', 'key_list.txt')
            self.RS_struct = dicom.read_file(self.template_dir)

        # Get ref file
        self.RefDs = dicom.read_file(self.lstFilesDCM[0])

        # The array is sized based on 'ConstPixelDims'
        # ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
        ds = self.Dicom_info[self.lstFilesDCM[0]].pixel_array
        self.ArrayDicom = np.zeros([self.num_images,ds.shape[0], ds.shape[1], 3], dtype='float32')
        self.SOPClassUID_temp =[None] * self.num_images
        self.SOPClassUID = [None] * self.num_images
        # loop through all the DICOM files
        for filenameDCM in self.lstFilesDCM:
            # read the file
            ds = self.Dicom_info[filenameDCM]
            # store the raw image data
            im = ds.pixel_array
            # im[im<200] = 200 #Don't know what the hell these units are, but the min (air) is 0
            im = np.array([im for i in range(3)]).transpose([1,2,0])
            self.ArrayDicom[self.lstFilesDCM.index(filenameDCM),:, :, :] = im
            self.slice_info[self.lstFilesDCM.index(filenameDCM)] = round(ds.ImagePositionPatient[2],3)
            self.SOPClassUID_temp[self.lstFilesDCM.index(filenameDCM)] = ds.SOPInstanceUID
        indexes = [i[0] for i in sorted(enumerate(self.slice_info),key=lambda x:x[1])]
        self.ArrayDicom = self.ArrayDicom[indexes]
        self.slice_info = self.slice_info[indexes]
        try:
            self.ArrayDicom = (self.ArrayDicom + ds.RescaleIntercept) / ds.RescaleSlope
        except:
            xxx = 1
        i = 0
        for index in indexes:
            self.SOPClassUID[i] = self.SOPClassUID_temp[index]
            i += 1
        self.ds = ds
        if self.template:
            print('Running off a template')
            self.changetemplate()
    def poly2mask(self,vertex_row_coords, vertex_col_coords, shape):
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        mask = np.zeros(shape, dtype=np.bool)
        mask[fill_row_coords, fill_col_coords] = True
        return mask
    def with_annotations(self,annotations,output_dir,ROI_Names=None):
        self.image_size_0, self.image_size_1 = annotations.shape[1], annotations.shape[2]
        self.ROI_Names = ROI_Names
        self.output_dir = output_dir
        if len(annotations.shape) == 3:
            annotations = np.expand_dims(annotations,axis=-1)
        self.annotations = annotations
        self.Mask_to_Contours()
    def Mask_to_Contours(self):
        self.RefDs = self.ds
        self.ShiftCols = self.RefDs.ImagePositionPatient[0]
        self.ShiftRows = self.RefDs.ImagePositionPatient[1]
        self.mult1 = self.mult2 = 1
        # if self.ShiftCols > 0:
        #     self.mult1 = -1
        # if self.ShiftRows > 0:
        #     self.mult2 = -1
        self.PixelSize = self.RefDs.PixelSpacing[0]
        current_names = []
        for names in self.RS_struct.StructureSetROISequence:
            current_names.append(names.ROIName)
        Contour_Key = {}
        xxx = 1
        for name in self.ROI_Names:
            Contour_Key[name] = xxx
            xxx += 1
        self.all_annotations = self.annotations
        base_annotations = copy.deepcopy(self.annotations)
        temp_color_list = []
        color_list = [[128,0,0],[170,110,40],[0,128,128],[0,0,128],[230,25,75],[225,225,25],[0,130,200],[145,30,180],
                      [255,255,255]]
        for Name in self.ROI_Names:
            if not temp_color_list:
                temp_color_list = copy.deepcopy(color_list)
            color_int = np.random.randint(len(temp_color_list))
            print('Writing data for ' + Name)
            self.annotations = copy.deepcopy(base_annotations[:,:,:,int(self.ROI_Names.index(Name)+1)])
            if 'Liver_BMA_Program_4_2Dfast' in self.ROI_Names or 'Liver_BMA_Program_4_3D' in self.ROI_Names:
                thresholds = [0.2,0.75,0.2]
                reduced_annotations = utils.remove_non_liver(self.annotations, threshold=thresholds[0])
                self.annotations[reduced_annotations == 0] = 0
                self.annotations = utils.variable_remove_non_liver(self.annotations, threshold=thresholds[1])
                self.annotations = utils.remove_non_liver(self.annotations, threshold=thresholds[2])
            elif self.theshold!=0:
                threshold = self.theshold
                for roi in self.ROI_Names:
                    if roi.find('Pancreas') != -1:
                        threshold = 0.5
                        break
                    if roi.find('right_eye_bma') != -1:
                        threshold = 0.75
                        break
                    if roi.find('parotid') != -1:
                        threshold = 0.85
                self.annotations = utils.variable_remove_non_liver(self.annotations, threshold=0.2, structure_name=self.ROI_Names)
                if self.single_structure:
                    self.annotations = utils.remove_non_liver(self.annotations, threshold=threshold)

            self.annotations = self.annotations.astype('int')

            make_new = 1
            allow_slip_in = True
            if Name not in current_names and allow_slip_in:
                self.RS_struct.StructureSetROISequence.append(copy.deepcopy(self.RS_struct.StructureSetROISequence[0]))
                if not self.template:
                    self.struct_index = len(self.RS_struct.StructureSetROISequence)-1
                else:
                    self.struct_index = 0
            else:
                make_new = 0
                self.struct_index = current_names.index(Name) - 1
            new_ROINumber = self.struct_index + 1
            self.RS_struct.StructureSetROISequence[self.struct_index].ROINumber = new_ROINumber
            self.RS_struct.StructureSetROISequence[self.struct_index].ReferencedFrameofReferenceUID = self.ds.FrameofReferenceUID
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIName = Name
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIVolume = 0
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIGenerationAlgorithm = 'SEMIAUTOMATIC'
            if make_new == 1:
                self.RS_struct.RTROIObservationsSequence.append(copy.deepcopy(self.RS_struct.RTROIObservationsSequence[0]))
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ObservationNumber = new_ROINumber
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ReferencedROINumber = new_ROINumber
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ROIObservationLabel = Name
            self.RS_struct.RTROIObservationsSequence[self.struct_index].RTROIInterpretedType = 'ORGAN'

            if make_new == 1:
                self.RS_struct.ROIContourSequence.append(copy.deepcopy(self.RS_struct.ROIContourSequence[0]))
            self.RS_struct.ROIContourSequence[self.struct_index].ReferencedROINumber = new_ROINumber
            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[1:] = []
            self.RS_struct.ROIContourSequence[self.struct_index].ROIDisplayColor = temp_color_list[color_int]
            del temp_color_list[color_int]

            contour_num = 0
            if np.max(self.annotations) > 0: # If we have an annotation, write it
                image_locations = np.max(self.annotations,axis=(1,2))
                indexes = np.where(image_locations>0)[0]
                for point, i in enumerate(indexes):
                    print(str(int(point / len(indexes) * 100)) + '% done with ' + Name)
                    annotation = self.annotations[i,:,:]
                    regions = regionprops(label(annotation),coordinates='xy')
                    for ii in range(len(regions)):
                        temp_image = np.zeros([self.image_size_0,self.image_size_1])
                        data = regions[ii].coords
                        rows = []
                        cols = []
                        for iii in range(len(data)):
                            rows.append(data[iii][0])
                            cols.append(data[iii][1])
                        temp_image[rows,cols] = 1
                        points = find_contours(temp_image, 0)[0]
                        output = []
                        for point in points:
                            output.append(((point[1]) * self.PixelSize + self.mult1 * self.ShiftCols))
                            output.append(((point[0]) * self.PixelSize + self.mult2 * self.ShiftRows))
                            output.append(self.slice_info[i])
                        if output:
                            if contour_num > 0:
                                self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence.append(copy.deepcopy(self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[0]))
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[contour_num].ContourNumber = str(contour_num)
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[contour_num].ContourImageSequence[0].ReferencedSOPInstanceUID = self.SOPClassUID[i]
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[contour_num].ContourData = output
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[contour_num].NumberofContourPoints = round(len(output)/3)
                            contour_num += 1
        self.RS_struct.SOPInstanceUID += '.' + str(np.random.randint(999))
        if self.template and self.delete_previous_rois:
            for i in range(len(self.RS_struct.StructureSetROISequence) - len(self.ROI_Names), -1 + len(self.ROI_Names), -1):
                del self.RS_struct.StructureSetROISequence[i]
            for i in range(len(self.RS_struct.RTROIObservationsSequence) - len(self.ROI_Names), -1 + len(self.ROI_Names),
                           -1):
                # if self.RS_struct.RTROIObservationsSequence[i].ROIObservationLabel not in self.ROI_Names:
                del self.RS_struct.RTROIObservationsSequence[i]
                # if self.RS_struct.RTROIObservationsSequence[i].ROIObservationLabel not in self.ROI_Names:
            for i in range(len(self.RS_struct.ROIContourSequence) - len(self.ROI_Names), -1 + len(self.ROI_Names),
                           -1):
                del self.RS_struct.ROIContourSequence[i]
        for i in range(len(self.RS_struct.StructureSetROISequence)):
            self.RS_struct.StructureSetROISequence[i].ROINumber = i + 1
            self.RS_struct.RTROIObservationsSequence[i].ReferencedROINumber = i + 1
            self.RS_struct.ROIContourSequence[i].ReferencedROINumber = i + 1

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_name = os.path.join(self.output_dir,'RS_MRN' + self.RS_struct.PatientID + '_' + self.RS_struct.SeriesInstanceUID + '.dcm')
        if os.path.exists(out_name):
            out_name = os.path.join(self.output_dir,'RS_MRN' + self.RS_struct.PatientID + '_' + self.RS_struct.SeriesInstanceUID + '1.dcm')
        print('Writing out data...')
        dicom.write_file(out_name, self.RS_struct)
        fid = open(os.path.join(self.output_dir,'Completed.txt'), 'w+')
        fid.close()
        print('Finished!')
        #Raystation_dir = self.output_dir.split('Output_MRN')[0]+'Output_MRN_RayStation\\'+self.RS_struct.PatientID+'\\'
        #if not os.path.exists(Raystation_dir):
        #dicom.write_file(Raystation_dir + 'RS_MRN' + self.RS_struct.PatientID + '_' + self.ds.SeriesInstanceUID + '.dcm', self.RS_struct)
        #fid = open(Raystation_dir+'Completed.txt','w+')
        #fid.close()
        return None

    def changetemplate(self):
        keys = self.RS_struct.keys()
        for key in keys:
            #print(self.RS_struct[key].name)
            if self.RS_struct[key].name == 'Referenced Frame of Reference Sequence':
                break
        self.RS_struct[key]._value[0].FrameofReferenceUID = self.ds.FrameofReferenceUID
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].ReferencedSOPInstanceUID = self.ds.StudyInstanceUID
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID = self.ds.SeriesInstanceUID
        for i in range(len(self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence)-1):
            del self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence[0]
        for i in range(len(self.SOPClassUID)):
            self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[i].ReferencedSOPInstanceUID = self.SOPClassUID[i]
            self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence.append(copy.deepcopy(self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[0]))
        del self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[-1]

        things_to_change = ['StudyInstanceUID','Specific Character Set','Instance Creation Date','Instance Creation Time','Study Date','Study Time',
                            'Accession Number','Study Description','Patient"s Name','Patient ID','Patients Birth Date','Patients Sex'
                            'Study Instance UID','Study ID','Frame of Reference UID']
        self.RS_struct.PatientsName = self.ds.PatientsName
        self.RS_struct.PatientsSex = self.ds.PatientsSex
        self.RS_struct.PatientsBirthDate = self.ds.PatientsBirthDate
        for key in keys:
            #print(self.RS_struct[key].name)
            if self.RS_struct[key].name in things_to_change:
                try:
                    self.RS_struct[key] = self.ds[key]
                except:
                    continue
        new_keys = open(self.key_list)
        keys = {}
        i = 0
        for line in new_keys:
            keys[i] = line.strip('\n').split(',')
            i += 1
        new_keys.close()
        for index in keys.keys():
            new_key = keys[index]
            try:
                self.RS_struct[new_key[0], new_key[1]] = self.ds[[new_key[0], new_key[1]]]
            except:
                continue
        return None
            # Get slice locations

def poly2mask(vertex_row_coords, vertex_col_coords):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, [512,512])
    mask = np.zeros([512,512], dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

if __name__ == "__main__":
    k = 1