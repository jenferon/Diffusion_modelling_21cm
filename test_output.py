import numpy as np
import matplotlib.pyplot as plt
import ST
import matplotlib 
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.style.use('/home/ppxjf3/paper_params.mplstyle') 
import torch
import matplotlib.font_manager
import tools21cm as t2c
from scipy import interpolate
import matplotlib.colors as colors

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
matplotlib.rcParams['figure.dpi'] = 300/scale
# Set up a figure with four panels, with two rows and columns

max =  43.11382395267486


#generated_image = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/output_ada_to_test/train-40xscale_0_test_test01_FSD_-654.4299926757812_ema.npy'))
generated_image = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/train-0xscale_0_test_faint_gal_LHS_FSD_-365.5400085449219_ema_500steps.npy')) #train-50xscale_0_test_properLHS_FSD_-609.0_ema_500steps.npy'))#train-10xscale_0_test_properLHS_FSD_-611.0599975585938_ema_500steps.npy')) # train-10xscale_0_test_properLHS_FSD_-609.6199951171875_ema.npy')) #train-80xscale_0_test_test01_FSD_-607.72998046875_ema.npy')) #ST4Diffusion_edited/fsd_epoch/train-110xscale_0_test_test01_FSD_-655.0800170898438_ema.npy')) #used in thesis
generated_image = np.squeeze(generated_image)
generated_image = ((generated_image+1)/2)*max

ii = 0
fig, ax = plt.subplots(1,1)
im = ax.imshow(generated_image[ii,:,:])
plt.colorbar(im, ax=ax)
plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/output_image{}_30epochs.png".format(ii), dpi=330)
plt.close()

original_image_array = np.zeros([64,64,64])
for ii in range(0,64):
    original_image = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/data/Tb_coeval_z12_Tvir_eta_4.4_131.341_{}.npy'.format(ii+1)))
    original_image = original_image*max
    original_image_array[ii,:,:] = original_image
print(np.max(original_image_array))
print(np.min(original_image_array))
fig, ax = plt.subplots(1,1)
im = ax.imshow(original_image)
plt.colorbar(im, ax=ax)
plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/input_image{}_renormalised.png".format(ii), dpi=330)
plt.close()

# 1-to-1 generation means that we generate one field matching the statistics of one given field. 
# if the input is a set of fields, then the 1-to-1 mode will synthesise them independently

# load image
image_target = original_image
hist_bins=50
pdf_array = np.empty((hist_bins,generated_image.shape[0]))
cdf_array = np.empty((hist_bins,generated_image.shape[0]))
# synthesize
for ii in range(0,generated_image.shape[0]):
  image_syn = generated_image[ii,:,:]
  
  fig, ax = plt.subplots(figsize=(fig_width*2, fig_height),
                        nrows=1, ncols=3,)

  pmc = ax[0].imshow(image_target)
  divider = make_axes_locatable(ax[0])
  cax = divider.append_axes('bottom', size='5%', pad=0.5)
  cbar = fig.colorbar(pmc, cax=cax, orientation='horizontal')
  cbar.set_label(r'$\delta T_b$ /mK', size=14)
  ax[0].set_title('simulated field', fontsize=14)
  pmc = ax[1].imshow(image_syn)
  divider = make_axes_locatable(ax[1])
  cax = divider.append_axes('bottom', size='5%', pad=0.5)
  cbar = fig.colorbar(pmc, cax=cax, orientation='horizontal')
  cbar.set_label(r'$\delta T_b$ /mK', size=14)
  ax[1].set_title('generated field', fontsize=14)
  ax[2].hist(image_target.flatten(), hist_bins, histtype='step', label='sim', color='k')
  ax[2].hist(image_syn.flatten(), hist_bins, histtype='step', label='gen', color='blue')
  ax[2].set_yscale('log')
  ax[2].legend(fontsize=14)
  ax[2].set_ylabel(r'$N_{\rm pixels}$')
  ax[2].set_xlabel(r'$\delta T_b$ /mK')
  plt.tight_layout()
  plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/histogram_renormalised"+str(ii)+".pdf", dpi=330)
  plt.close()
    
  """
  See if PDF or CDF represents the brightest pixels better
  """

  # getting data of the histogram
  count, bins_count = np.histogram(image_syn.flatten(), bins=hist_bins)
  count_tar, bins_tar_count = np.histogram(image_target.flatten(), bins=hist_bins)
  # finding the PDF of the histogram using count values
  pdf = count / sum(count)
  pdf_array[:,ii] = pdf
  #pdf_tar = count_tar / sum(count_tar)
  
  """plt.plot(bins_count[1:], pdf, label='synth')
  plt.plot(bins_tar_count[1:], pdf_tar, label='truth')
  plt.tight_layout()
  plt.legend()
  plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/PDF"+str(ii)+".pdf", dpi=330)
  plt.close()"""
  # using numpy np.cumsum to calculate the CDF
  # We can also find using the PDF values by looping and adding
  cdf = np.cumsum(pdf)
  cdf_array[:,ii] = cdf
  #cdf_tar = np.cumsum(pdf_tar)
  
  """plt.plot(bins_count[1:], cdf, label='synth')
  plt.plot(bins_tar_count[1:], cdf_tar, label='truth')
  plt.tight_layout()
  plt.legend()
  plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/CDF"+str(ii)+".pdf", dpi=330)
  plt.close()"""

# make average PDF and CDF
pdf_tar_array = np.empty((hist_bins,generated_image.shape[0]))
cdf_tar_array = np.empty((hist_bins,generated_image.shape[0]))
# synthesize
for ii in range(0,original_image_array.shape[0]):
  count_tar, bins_tar_count = np.histogram(original_image_array[ii,:,:].flatten(), bins=hist_bins)
  pdf_tar_array[:,ii] = count_tar / sum(count_tar)
  cdf_tar_array[:,ii] = np.cumsum(pdf_tar_array[:,ii])


fig, ax = plt.subplots(figsize=(fig_width*2, fig_height),
                        nrows=1, ncols=2,)
ax[0].plot(bins_count[1:], pdf_array.mean(axis=1), label='gen', color='blue')
ax[0].fill_between(bins_count[1:], pdf_array.mean(axis=1)-pdf_array.std(axis=1), pdf_array.mean(axis=1)+pdf_array.std(axis=1), color='blue', alpha=0.5)
ax[0].plot(bins_tar_count[1:], pdf_tar_array.mean(axis=1), label='sim', color='k')
ax[0].fill_between(bins_tar_count[1:], pdf_tar_array.mean(axis=1)-pdf_tar_array.std(axis=1), pdf_tar_array.mean(axis=1)+pdf_tar_array.std(axis=1), color='k', alpha=0.5)
ax[0].set_ylabel('PDF', fontsize=16)
ax[0].set_xlabel(r'$\delta T_b$ \mK', fontsize=16)
ax[0].set_yscale('log')

ax[1].plot(bins_count[1:], cdf_array.mean(axis=1),  label='gen', color='blue')
ax[1].fill_between(bins_count[1:], cdf_array.mean(axis=1)-cdf_array.std(axis=1), cdf_array.mean(axis=1)+cdf_array.std(axis=1), color='blue', alpha=0.5)
ax[1].plot(bins_tar_count[1:], cdf_tar_array.mean(axis=1), label='sim', color='k')
ax[1].fill_between(bins_tar_count[1:], cdf_tar_array.mean(axis=1)-cdf_tar_array.std(axis=1), cdf_tar_array.mean(axis=1)+cdf_tar_array.std(axis=1), color='k', alpha=0.5)
ax[1].legend(fontsize=14)
ax[1].set_ylabel('CDF', fontsize=16)
ax[1].set_xlabel(r'$\delta T_b$ \mK', fontsize=16)

plt.tight_layout()
plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/PDF_CDF_average.pdf", dpi=330)
plt.close()



#comparitive power spectra
"""
Compare power spectra of input vs output
"""
import tools21cm as t2c
from astropy.cosmology import FlatLambdaCDM, LambdaCDM

import astropy.units as u
cosmo = FlatLambdaCDM(H0=71 * u.km / u.s / u.Mpc, Om0=0.27)

def return_power_spectra(data, length, kbins=12, binning='log'):
    box_dims = length
    V = length*length

    p, k = t2c.power_spectrum_1d(data[:,:],  box_dims=box_dims, kbins=kbins, binning = binning, return_n_modes=False)
    return (p*k**3)/(2*np.pi**2), k
  
#find variance of power spectra in samples
power_spec_sample = np.empty([generated_image.shape[1], 12])
k_sample = np.empty([12])
power_spec_train = np.empty([generated_image.shape[1], 12])
k_train = np.empty([12])
for ii in range(0,generated_image.shape[1]):
  power_spec_sample[ii,:], k_sample = return_power_spectra(generated_image[ii,:,:], 128)
  power_spec_train[ii,:], k_train = return_power_spectra(original_image_array[ii,:,:], 128)

fig, ax = plt.subplots(figsize=(fig_width, fig_height),
                       nrows=1, ncols=1,)
ax.plot(k_train, np.mean(power_spec_train, axis=0), label='sim', color='k')
ax.fill_between(k_train, np.mean(power_spec_train, axis=0)-np.std(power_spec_train, axis=0), np.mean(power_spec_train, axis=0)+np.std(power_spec_train, axis=0), color='k', alpha=0.5)
ax.errorbar(k_sample, np.mean(power_spec_sample, axis=0), yerr=np.std(power_spec_sample, axis=0), label='gen', color='blue', capsize=3)
ax.set_ylabel(r'$\Delta_{\rm 2D} ^2(k)/\rm mK^2$', fontsize=16)
ax.set_xlabel(r'$k/(\rm Mpc/h)^{-1}$', fontsize=16)
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(frameon=False, fontsize = 14)
plt.tight_layout()
plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/avg_power_spec_comparison_renormalised.pdf", dpi=330)
plt.close()


#make scattering coefficient graph

J=5
L=4
M=64
N=64

filter_set = ST.FiltersSet(M, N, J, L).generate_morlet(precision='single')
ST_calculator = ST.ST_2D(filter_set, J, L, device='gpu', weight=None)

S_target, S0_target, S1_target, S2_target, _, _, _, _ = ST_calculator.forward(torch.from_numpy(original_image_array), J, L)
S_synth, S0_synth, S1_synth, S2_synth, _, _, _, _ = ST_calculator.forward(torch.from_numpy(generated_image), J, L)
print(np.mean(S0_synth.cpu().numpy(), axis=0))
print(np.mean(S0_target.cpu().numpy(), axis=0))
xx = np.array([0,1,2,3,4])

#make array to plot
xx_2 = [1,2,3,4,2,3,4,3,4,4,1,2,3,4,2,3,4,3,4,4]
jj = np.array([0,0,0,0,1,1,1,2,2,3,0,0,0,0,1,1,1,2,2,3])
yy_syth = np.zeros([20])
yy_target = np.zeros([20])
yerr_syth = np.zeros([20])
yerr_target = np.zeros([20])
for indx, ii in enumerate(xx_2):
  
  yy_syth[indx] = np.mean(S2_synth[:,jj[indx],0,ii,-1].cpu().numpy(), axis=0)
  yy_target[indx] = np.mean(S2_target[:,jj[indx],0,ii,-1].cpu().numpy(), axis=0)

  yerr_syth[indx] = np.std(S2_synth[:,jj[indx],0,ii,-1].cpu().numpy(), axis=0)
  yerr_target[indx] = np.std(S2_target[:,jj[indx],0,ii,-1].cpu().numpy(), axis=0)

#averaged over l1
fig = plt.figure(figsize=(fig_width*2, fig_height))
gs = fig.add_gridspec(1, 4)
fig_ax1 = fig.add_subplot(gs[0])
fig_ax1.errorbar(xx, np.mean(S1_synth[:,:,0].cpu().numpy(), axis=0), yerr=np.std(S1_synth[:,:,0].cpu().numpy(), axis=0), label='gen', color='blue', capsize=3)
fig_ax1.plot(xx, np.mean(S1_target[:,:,0].cpu().numpy(), axis=0), label='sim', color='k')
fig_ax1.fill_between(xx, np.mean(S1_target[:,:,0].cpu().numpy(), axis=0)-np.std(S1_target[:,:,0].cpu().numpy(), axis=0), np.mean(S1_target[:,:,0].cpu().numpy(), axis=0)+np.std(S1_target[:,:,0].cpu().numpy(), axis=0), color='k', alpha=0.5)
fig_ax1.set_ylabel(r'$S_1$', fontsize=16)
fig_ax1.set_xlabel(r'$j_1$', fontsize=16)
fig_ax1.legend(frameon=False, fontsize = 14)
fig_ax2 = fig.add_subplot(gs[1:])
fig_ax2.set_ylabel(r'$S_2$', fontsize=16)
fig_ax2.errorbar(range(len(xx_2)), yy_syth, yerr=yerr_syth, label='gen', color='blue', capsize=3)
fig_ax2.plot(range(len(xx_2)), yy_target, label='synth', color='k')
fig_ax2.fill_between(range(len(xx_2)), yy_target-yerr_target,yy_target +yerr_target, color='k', alpha=0.5)
fig_ax2.set_xlabel(r'$j_2$', fontsize=16)
fig_ax2.set_xlim(0,len(xx_2)-1)
fig_ax2.set_xticks(range(len(xx_2)))
fig_ax2.set_xticklabels(xx_2)
fig_ax2.vlines([3,6,8,13,16,18], 0, 1, transform=fig_ax2.get_xaxis_transform(), linestyle='dashed')
fig_ax2.vlines([9.5], 0, 1, linewidth=2, transform=fig_ax2.get_xaxis_transform(), linestyle='dashed')

for ii in range(0,3):
  fig_ax2.text(0.05+ii*0.15,0.05,r'$j_1={}$'.format(ii),transform=fig_ax2.transAxes, ha='left', va='bottom', fontsize=12)
  fig_ax2.text(0.55+ii*0.15,0.05,r'$j_1={}$'.format(ii),transform=fig_ax2.transAxes, ha='left', va='bottom', fontsize=12)
fig_ax2.text(0.43,0.05,r'$j_1=3$',transform=fig_ax2.transAxes, ha='left', va='bottom', fontsize=12)
fig_ax2.text(0.95,0.05,r'$j_1=3$'.format(ii),transform=fig_ax2.transAxes, ha='left', va='bottom', fontsize=12)
fig_ax2.text(0.35,0.95,r'$(l_2-l_1)\%L=0$',transform=fig_ax2.transAxes, ha='left', va='bottom', fontsize=12)
fig_ax2.text(0.85,0.95,r'$(l_2-l_1)\%L=1$',transform=fig_ax2.transAxes, ha='left', va='bottom', fontsize=12)
"""for ii in range(1,5):
  ax[ii].errorbar(np.array([1,2,3,4,5]), np.mean(S2_synth[:,ii-1,0,:,-1].cpu().numpy(), axis=0), yerr=np.std(S2_synth[:,ii-1,0,:,-1].cpu().numpy(), axis=0), label='synthesised', color='k', capsize=3)
  ax[ii].plot(np.array([1,2,3,4,5]), np.mean(S2_target[:,ii-1,0,:,-1].cpu().numpy(), axis=0), label='target', color='blue')
  ax[ii].fill_between(np.array([1,2,3,4,5]), np.mean(S2_target[:,ii-1,0,:,-1].cpu().numpy(), axis=0)-np.std(S2_target[:,ii-1,0,:,-1].cpu().numpy(), axis=0), np.mean(S2_target[:,ii-1,0,:,-1].cpu().numpy(), axis=0)+np.std(S2_target[:,ii-1,0,:,-1].cpu().numpy(), axis=0), color='blue', alpha=0.5)
  ax[ii].set_xlabel(r'$j_2$', fontsize=16)
  ax[ii].set_xlim(left=ii)
  ax[ii].text(0.05,0.95,r'$j_1={}, l_1=0,l_2=4$'.format(ii-1),transform=ax[ii].transAxes, ha='left', va='top', fontsize=12)"""
plt.tight_layout()
plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/scattering_graph.pdf", dpi=330)
plt.close()


"""fig = plt.figure(figsize=(10, 4))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.3, hspace=0.3)
ax = fig.add_subplot(gs[0])
ax.errorbar(xx, np.mean(S1_synth[:,:,0].cpu().numpy(), axis=0), yerr=np.std(S1_synth[:,:,0].cpu().numpy(), axis=0), label='synthesised', color='k', capsize=3)
ax.plot(xx, np.mean(S1_target[:,:,0].cpu().numpy(), axis=0), label='target', color='blue')
ax.fill_between(xx, np.mean(S1_target[:,:,0].cpu().numpy(), axis=0)-np.std(S1_target[:,:,0].cpu().numpy(), axis=0), np.mean(S1_target[:,:,0].cpu().numpy(), axis=0)+np.std(S1_target[:,:,0].cpu().numpy(), axis=0), color='blue', alpha=0.5)
ax.set_ylabel(r'$S_1$', fontsize=16)
ax.set_xlabel(r'$j_1$', fontsize=16)
ax.legend(frameon=False, fontsize = 14)

for col in range(1, 4):
  ax = fig.add_subplot(gs[col])
  j2 = np.array([1,2,3,4,5])
  j1 = col - 1
  shift = j1 * 0.1
  
  ax.errorbar(j2, np.mean(S2_synth[:,col-1,0,:,-1].cpu().numpy(), axis=0), yerr=np.std(S2_synth[:,col-1,0,:,-1].cpu().numpy(), axis=0), label='synthesised', color='k', capsize=3)
  ax.plot(j2, np.mean(S2_target[:,col-1,0,:,-1].cpu().numpy(), axis=0), label='target', color='blue')
  ax.fill_between(j2, np.mean(S2_target[:,col-1,0,:,-1].cpu().numpy(), axis=0)-np.std(S2_target[:,col-1,0,:,-1].cpu().numpy(), axis=0), np.mean(S2_target[:,col-1,0,:,-1].cpu().numpy(), axis=0)+np.std(S2_target[:,col-1,0,:,-1].cpu().numpy(), axis=0), color='blue', alpha=0.5)
  
  ax.set_xticks(j2)
  ax.set_xlabel("j$_2$")
  if col == 1:
    ax.set_ylabel("S$_2$")
plt.tight_layout()
plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/scattering_graph_test.pdf", dpi=330)
plt.close()
""""""
---------------------------------------------------------------------
parameter checking
---------------------------------------------------------------------
"""
"""
#check the effect of guidence scale
generated_image_scale01 = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/output_ada_to_test/train-30xscale_0.1_test_test01_FSD_-676.3599853515625_ema.npy'))
generated_image_scale01 = np.squeeze(generated_image_scale01)

power_spec_01 = np.empty([generated_image_scale01.shape[1], 12])
k_01 = np.empty([12])
for ii in range(0,generated_image_scale01.shape[1]):
  power_spec_01[ii,:], k_01 = return_power_spectra(generated_image_scale01[ii,:,:], 128)


plt.errorbar(k_train, np.mean(power_spec_train, axis=0), yerr=np.std(power_spec_train, axis=0), label='target')
plt.errorbar(k_sample, np.mean(power_spec_sample, axis=0), yerr=np.std(power_spec_sample, axis=0), label='scale = 0')
plt.errorbar(k_01, np.mean(power_spec_01, axis=0), yerr=np.std(power_spec_01, axis=0), label='scale = 0.1')
plt.ylabel(r'$\Delta_{\rm 2D} ^2(k)/\rm mK^2$')
plt.xlabel(r'$k/(\rm Mpc/h)^{-1}$')
plt.yscale('log')
plt.xscale('log')
plt.legend(frameon=False, fontsize = 14)
plt.tight_layout()
plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/avg_power_spec_guidance_scale.png", dpi=330)
plt.close()


hist_range=(-1, 1)
hist_bins=50
plt.figure(figsize=(9,3), dpi=200)
plt.subplot(131) 
plt.imshow(generated_image_scale01[0,:,:])
plt.xticks([]); plt.yticks([]); plt.title('guidance scale = 0.1')
plt.subplot(132)
plt.imshow(image_syn)
plt.xticks([]); plt.yticks([]); plt.title('guidance scale = 0')
plt.subplot(133); 
plt.hist(generated_image_scale01[0,:,:].flatten(), hist_bins, histtype='step', label='scale = 0.1')
plt.hist(image_syn.flatten(), hist_bins, histtype='step', label='scale = 0')
plt.yscale('log'); plt.legend(loc='lower center'); plt.title('histogram')
plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/histogram_guidance_scale.png", dpi=330)
plt.close()

#test EMA or not - exponetial moving average is supposed to stabalise model convergence - might not make a difference at such a small epoch
generated_image_noEMA = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/output_ada_to_test/train-30xscale_0_test_test01_FSD_-648.4500122070312.npy'))
generated_image_noEMA = np.squeeze(generated_image_noEMA)

power_spec_noEMA = np.empty([generated_image_noEMA.shape[1], 12])
k_noEMA = np.empty([12])
for ii in range(0,generated_image_noEMA.shape[1]):
  power_spec_noEMA[ii,:], k_noEMA = return_power_spectra(generated_image_noEMA[ii,:,:], 128)
  
plt.errorbar(k_train, np.mean(power_spec_train, axis=0), yerr=np.std(power_spec_train, axis=0), label='target')
plt.errorbar(k_sample, np.mean(power_spec_sample, axis=0), yerr=np.std(power_spec_sample, axis=0), label='EMA')
plt.errorbar(k_noEMA, np.mean(power_spec_noEMA, axis=0), yerr=np.std(power_spec_noEMA, axis=0), label='No EMA')
plt.ylabel(r'$\Delta_{\rm 2D} ^2(k)/\rm mK^2$')
plt.xlabel(r'$k/(\rm Mpc/h)^{-1}$')
plt.yscale('log')
plt.xscale('log')
plt.legend(frameon=False, fontsize = 14)
plt.tight_layout()
plt.savefig("/home/ppxjf3/ST4Diffusion_edited/outputs/avg_power_spec_EMA.png", dpi=330)
plt.close()"""
#test FSD with unseen data
import calc_FSD 

"""testing_array = np.zeros([64,64,64])
for ii in range(0,64):
    original_image = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/test_set/Tb_coeval_z12_Tvir_eta_4.4_131.341_{}.npy'.format(ii+1000)))
    original_image = original_image*max
    testing_array[ii,:,:] = original_image"""
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

#FSD_class = calc_FSD.FSD(torch.squeeze(torch.from_numpy(generated_image)), torch.squeeze(torch.from_numpy(original_image_array)))

FSD = calculate_fid(np.squeeze(generated_image).reshape(64,64*64), np.squeeze(original_image_array).reshape(64,64*64)) #FSD_class.calc_FSD()
print("The FSD of this model is {}".format(FSD))
"""
#make power spec cylindrical
plotting_scale={'x': 'log', 'y': 'log', 'z': 'log'}
def cylindircal_ps(data):
  

  pp_BSS2_GPR, kper, kpar= t2c.power_spectrum_2d(data,  binning = 'linear', kbins=8, return_modes=False)
  #normalise to dimensionless power spectrum
  for ii in range(0,len(kper)):
      for jj in range(0,len(kpar)):
          pp_BSS2_GPR[ii,jj] = (pp_BSS2_GPR[ii,jj]*np.sqrt(kper[ii]**2+kpar[jj]**2)**3)/(2*np.pi**2)
   
  return pp_BSS2_GPR, [kper, kpar]

power_spec_sample = np.empty([generated_image.shape[1], 8, 8])
k_sample = np.empty([8,8])
power_spec_train = np.empty([generated_image.shape[1], 8, 8])
k_train = np.empty([8,8])
for ii in range(0,generated_image.shape[1]):
  power_spec_sample[ii,:,:], k_sample = cylindircal_ps(generated_image[ii,:,:])
  power_spec_train[ii,:,:], k_train = cylindircal_ps(original_image_array[ii,:,:])
  

fp = interpolate.interp2d(k_sample[0,:], k_sample[1,:], power_spec_sample.mean(axis=0).T, kind='linear')
CC_gen = fp(k_sample[0,:],k_sample[1,:])
norm_gen = colors.LogNorm(vmin=CC_gen[np.isfinite(CC_gen)].min(), vmax=CC_gen[np.isfinite(CC_gen)].max()) if plotting_scale['z']=='log' else None 

fp = interpolate.interp2d(k_train[0,:], k_train[1,:], power_spec_train.mean(axis=0).T, kind='linear')
CC_sim = fp(k_train[0,:],k_train[1,:])
norm_sim = colors.LogNorm(vmin=CC_sim[np.isfinite(CC_sim)].min(), vmax=CC_sim[np.isfinite(CC_sim)].max()) if plotting_scale['z']=='log' else None 


fig, ax = plt.subplots(figsize=(fig_width*2, fig_height),
                       nrows=1, ncols=2,)
    

pcm_truth = ax[0].pcolormesh(k_sample[0,:], k_sample[1,:],  power_spec_sample.mean(axis=0), norm=norm_gen, cmap='inferno')
cbar= plt.colorbar(pcm_truth, ax=ax[0], pad=0.01)
cbar.set_label(r'$\Delta_{\rm 3D, Truth}^2 (k_{\parallel}, k_{\perp}) /\rm mK^2 $', size=16) 
ax[0].set_xlabel(r'$k_{\perp}/ \rm Mpc^{-1}$', fontsize = 14)
ax[0].set_ylabel(r'$k_{\parallel} /\rm Mpc^{-1}$', fontsize = 14)
ax[0].set_xscale(plotting_scale['x'])
ax[0].set_yscale(plotting_scale['y'])

pcm_res = ax[2].pcolormesh(k_train[0,:], k_train[1,:], power_spec_train.mean(axis=0), norm=norm_sim, cmap='inferno')
cbar= plt.colorbar(pcm_res, ax=ax[2], pad=0.01)
cbar.set_label(r'$\Delta_{\rm 3D, BSS1 ICA}^2 (k_{\parallel}, k_{\perp}) /\rm mK^2 $', size=16) 
ax[2].set_xlabel(r'$k_{\perp}/ \rm Mpc^{-1}$', fontsize = 14)
ax[2].set_ylabel(r'$k_{\parallel} /\rm Mpc^{-1}$', fontsize = 14)
ax[2].set_xscale(plotting_scale['x'])
ax[2].set_yscale(plotting_scale['y'])

plt.tight_layout()
plt.savefig('/home/ppxjf3/ST4Diffusion_edited/outputs/powerspec_comparsion.pdf', dpi=330)
plt.show()"""