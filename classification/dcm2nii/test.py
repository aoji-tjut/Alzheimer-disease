import SimpleITK as sitk
import numpy as np
import nibabel as nib


series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs("./dcm")
series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames("./dcm", series_id[0])
series_reader = sitk.ImageSeriesReader()
series_reader.SetFileNames(series_file_names)
image3d = series_reader.Execute()
sitk.WriteImage(image3d, "./dcm/res.nii")

X = nib.load("./dcm/res.nii")
init = np.array(X.get_fdata())
X = np.array(X.get_fdata())
print(X.shape)
