import nibabel as nib
img = nib.load('an_example_4d.nii')
data = img.get_data()
