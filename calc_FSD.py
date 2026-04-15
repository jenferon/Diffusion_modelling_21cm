import numpy as np
import sys
sys.path.append('/home/ppxjf3/ST4Diffusion_edited/')
#import scattering
import ST
import torch


class FSD(object):
    def __init__(self, true_data, modeled_data, J=5, L=4, M=64, N=64,):
        self.X = torch.squeeze(true_data)
        self.Y = torch.squeeze(modeled_data)
        self.J = J
        self.L = L
        self.M = M
        self.N = N
    
    def calculate_scattering(self):
        filter_set = ST.FiltersSet(self.M, self.N, self.J, self.L).generate_morlet(precision='single')

        ST_calculator = ST.ST_2D(filter_set, self.J, self.L, device='gpu', weight=None)
        S_X, S0, S1, S2, _, _, _, _ = ST_calculator.forward(self.X, self.J, self.L)
        S_Y, S0, S1, S2, _, _, _, _ = ST_calculator.forward(self.Y, self.J, self.L)
        print(S_X.shape)
        print(S_Y.shape)
        mean_x = torch.mean(S_X)
        std_x  = torch.cov(S_X)

        mean_y = torch.mean(S_Y)
        std_y  = torch.cov(S_Y)
        
        
        return torch.log10(mean_x), torch.log10(mean_y), torch.log10(std_x), torch.log10(std_y)
    
    def calc_FSD(self):
        mean_x, mean_y, std_x, std_y = self.calculate_scattering()
        print("mean_x:", mean_x)
        print("mean_y:", mean_y)
        print("std_x min:", std_x.min())
        print("std_y min:", std_y.min())
        temp = 2*torch.sqrt(std_x * std_y)
        FSD = torch.abs(mean_x - mean_y)**2 + torch.trace(std_x + std_y - temp)
        print(FSD)
        
        return FSD
    
    
    
    
"""
load up simulated and created dataset
"""
"""original_image_tensor = torch.zeros([64,64,64])
for ii in range(0,64):
    original_image = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/data/Tb_coeval_z12_Tvir_eta_4.4_131.341_{}.npy'.format(ii+1)))
    original_image_tensor[ii,:,:] = torch.from_numpy(original_image)


#diffusion modeled array into pytorch tensor
diff_array = np.float32(np.load('/home/ppxjf3/ST4Diffusion_edited/outputs/train-60xscale_0.1_test_test01_ema.npy'))
print(diff_array.shape)
generated_image_tensor = torch.zeros([64,64,64])
for ii in range(0,diff_array.shape[1]):
    generated_image_tensor[ii,:,:] =torch.from_numpy(diff_array[0,ii,0,:,:])
"""

