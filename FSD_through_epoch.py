import numpy as np
import matplotlib.pyplot as plt
import ST
import os
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.style.use('/home/ppxjf3/paper_params.mplstyle') 
import torch
import matplotlib.font_manager
import matplotlib as mpl
import calc_FSD 

column_width=240# call "\the\columnwidth" in LaTeX to find
ppi=72#default ppi, can be left the same

scale=2
fig_width=column_width/ppi*scale#inches
fig_height=3*scale#inches

##SET FONT SIZES
font_small_size = 9
font_medium_size = 12
font_bigger_size =14

plt.rc('font', size=font_small_size) # controls default text sizes
plt.rc('axes', titlesize=font_small_size) # fontsize of the axes title
plt.rc('axes', labelsize=font_medium_size) # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_bigger_size) # fontsize of the tick labels
plt.rc('ytick', labelsize=font_bigger_size) # fontsize of the tick labels
plt.rc('legend', fontsize=font_small_size) # legend fontsize
plt.rc('figure', titlesize=font_bigger_size)


#correct font for MNRAS
#can be found at https://www.fontsquirrel.com/fonts/nimbus-roman-no9-l
#can be installed on Unix systems by putting unzipped folder in directory /home/{user}/.fonts
#run "fc-cache -v" in console to inform system of new font
#print(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))
#plt.rc('font', family='Nimbus Roman')
# mpl.rcParams["font.family"] = "Nimbus Roman"
# mpl.rcParams['mathtext.fontset'] = 'custom'
# mpl.rcParams['mathtext.rm'] = 'Nimbus Roman'
# mpl.rcParams['mathtext.it'] = 'Nimbus Roman:italic'
# mpl.rcParams['mathtext.bf'] = 'Nimbus Roman:bold'
"""
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
"""

#DPI of MNRAS is 300
mpl.rcParams['figure.dpi'] = 300/scale
# Set up a figure with four panels, with two rows and columns

max =  43.11382395267486

testing_array = np.zeros([64,64,64])
for ii in range(0,64):
    original_image = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/test_set/Tb_coeval_z12_Tvir_eta_4.4_131.341_{}.npy'.format(ii+1000)))
    original_image = original_image*max
    testing_array[ii,:,:] = original_image

FSD_array=np.zeros(9)
idx = 0 
"""
names = ['train-0xscale_0_test_test01_ema.npy', 'train-10xscale_0_test_test01_ema.npy', 'train-20xscale_0_test_test01_ema.npy', 'train-30xscale_0_test_test01_ema.npy', 'train-40xscale_0_test_test01_ema.npy', 
         'train-50xscale_0_test_test01_ema.npy', 'train-60xscale_0_test_test01_ema.npy', 'train-70xscale_0_test_test01_ema.npy']

"""
for entry in os.scandir('/home/ppxjf3/ST4Diffusion_edited/fsd_epoch/'):  
    if entry.is_file():  # check if it's a file
        print(idx)
        generated_image = np.float32(np.load(entry.path))

        generated_image = np.squeeze(generated_image)
        generated_image = ((generated_image+1)/2)*max
        FSD_class = calc_FSD.FSD(torch.squeeze(torch.from_numpy(generated_image)), torch.squeeze(torch.from_numpy(testing_array)))
        FSD = FSD_class.calc_FSD()
        print("The FSD of this model is {}".format(FSD))
        FSD_array[idx] = FSD
        idx+=1
"""
for name in names:
    generated_image = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/outputs/'+name))
    generated_image = np.squeeze(generated_image)
    generated_image = ((generated_image+1)/2)*max
    FSD_class = calc_FSD.FSD(torch.squeeze(torch.from_numpy(generated_image)), torch.squeeze(torch.from_numpy(testing_array)))
    FSD = FSD_class.calc_FSD()
    print("The FSD of this model is {}".format(FSD))
    FSD_array[idx] = FSD"""

plt.plot(np.arange(0,90,10.0), FSD_array)
plt.yscale('log')
plt.ylabel('FSD')
plt.xlabel('epoch')
plt.savefig('/home/ppxjf3/ST4Diffusion_edited/FSD.pdf')
print(FSD_array)