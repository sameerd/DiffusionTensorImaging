""" 
Example to use this repository 

This example shows how to load the files from disk and save the output files
back to disk.
"""
# Load the python libraries
import os, sys
import numpy as np

import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import applymask, bounding_box, crop

# Add pymods to our module path
module_path = os.path.join(os.path.abspath(os.path.join('.')), 'pymods')
if module_path not in sys.path:
    sys.path.append(module_path)

from pymods.freewater_runner import FreewaterRunner

# directory names
input_directory = "data/input"
working_directory = "data/working"
output_directory = "data/output"
    
# filenames
fdwi = input_directory + "/dti.nii.gz"
fmask = input_directory + "/nodif_brain.nii.gz"
fbval = input_directory + "/bvals"
fbvec = input_directory + "/bvecs"

# Load the data
img = nib.load(fdwi)
img_data = img.get_data()
# Load the mask
mask = nib.load(fmask)
mask_data = mask.get_data()

#load bvals, bvecs and gradient files
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

# Apply the mask to the volume
mask_boolean = mask_data > 0.01
mins, maxs = bounding_box(mask_boolean)
mask_boolean = crop(mask_boolean, mins, maxs)
cropped_volume = crop(img_data, mins, maxs)
data = applymask(cropped_volume, mask_boolean )

fw_runner = FreewaterRunner(data, gtab) 
fw_runner.LOG = True # turn on logging for this example
fw_runner.run_model(num_iter=100, dt=0.001)

# save the loss function to the working directory
#freewater_runner.plot_loss()

# Save the free water map somewhere
fw_file = output_directory + "/freewater.nii.gz"
nib.save(nib.Nifti1Image(fw_runner.get_fw_map(), img.affine), fw_file)

fw_md_file = output_directory + "/freewater_md.nii.gz"
nib.save(nib.Nifti1Image(fw_runner.get_fw_md(), img.affine), fw_md_file)

fw_fa_file = output_directory + "/freewater_fa.nii.gz"
nib.save(nib.Nifti1Image(fw_runner.get_fw_fa(), img.affine), fw_fa_file)




