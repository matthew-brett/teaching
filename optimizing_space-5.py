import nibabel as nib
img = nib.load('ds107_sub012_t1r2.nii')
data = img.get_data()
data.shape
# (64, 64, 35, 166)
