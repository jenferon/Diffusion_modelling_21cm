import numpy as np
import os

array = []
for entry in os.scandir('/home/ppxjf3/ST4Diffusion_edited/train_set_LHS_proper'):  
    if entry.is_file():  # check if it's a file
        
        generated_image = np.float32(np.load(entry.path))
        if np.isnan(generated_image).any() == True:
            print("{} is full of nan's".format(entry.path))
            
            array+= [generated_image]
        else:
            continue
print(len(array))
#its.writeto('/home/ppxjf3/ST4Diffusion_edited/includes_nans.fits',np.array(array))