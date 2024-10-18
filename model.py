# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from skimage.feature import match_template




def eulerAnglesToRotationMatrix_torch(theta):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=theta.device)
    one = torch.ones(1, dtype=torch.float32, device=theta.device)
    x0 = torch.cat([ one,    zero,   zero])  # (3, 1)
    x1 = torch.cat([ zero,    torch.cos(theta[0:1]),   -torch.sin(theta[0:1])])  # (3, 1)
    x2 = torch.cat([ zero,    torch.sin(theta[0:1]),   torch.cos(theta[0:1])])  # (3, 1)
    x = torch.stack([x0, x1, x2], dim=0)  # (3, 3)
    
    y0 = torch.cat([ torch.cos(theta[1:2]),    zero,   torch.sin(theta[1:2])])  # (3, 1)
    y1 = torch.cat([ zero,    one,   zero])  # (3, 1)
    y2 = torch.cat([ -torch.sin(theta[1:2]),    zero,   torch.cos(theta[1:2])])  # (3, 1)
    y = torch.stack([y0, y1, y2], dim=0)  # (3, 3)
    
    z0 = torch.cat([ torch.cos(theta[2:3]),   -torch.sin(theta[2:3]),   zero])  # (3, 1)
    z1 = torch.cat([ torch.sin(theta[2:3]),   torch.cos(theta[2:3]),   zero])  # (3, 1)
    z2 = torch.cat([ zero,    zero,   one])  # (3, 1)
    z = torch.stack([z0, z1, z2], dim=0)  # (3, 3)
    
    R = torch.matmul(z, torch.matmul( y, x ))
    
    return R  # (3, 3)



def sample_from_matrix(ref, rot, trans):
    # grid = torch.einsum('imn, ij -> mnj', ref, rot) #HW3
    grid = torch.einsum('jmn, ij -> mni', ref, rot) #HW3
    
    grid_trans = grid+trans.unsqueeze(0).unsqueeze(0)
    
    # grid[40,40,:]
    # grid_trans[40,40,:]
    
    return grid_trans

def encode_position(input, levels, inc_input):
    """
    :param input:   (..., C)            torch.float32
    :param levels:  scalar L            int
    :param inc_input:                   bool
    :return:        (..., C*(2L+1*inc_input))     torch.float32
    C = grid.shape[2] = 3
    L = levels = 10
    inc_input = True
    C*(2L+1*inc_input) = 63
    """

    # this is already doing 'log_sampling' in the official code.
    result_list = [input] if inc_input else []
    for i in range(levels):
        temp = 2.0**i * input  # (..., C)
        result_list.append(torch.sin(temp))  # (..., C)
        result_list.append(torch.cos(temp))  # (..., C)

    result_list = torch.cat(result_list, dim=-1)  # (..., C*(2L+1)) The list has (2L+1) elements, with (..., C) shape each.
    return result_list  # (..., C*(2L+1))
    
def mse2psnr(mse):
    """
    :param mse: scalar
    :return:    scalar np.float32
    """
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)

class LearnPose(nn.Module):
    def __init__(self, store_dict, learn_R=True, learn_t=True):
        super(LearnPose, self).__init__()
        self.learn_R = learn_R
        self.learn_t = learn_t
        self.create_r_t(store_dict)
        self.num_cams = len(store_dict)
        

    def forward(self, pose_id):
        a = self.r[pose_id]  # (3,) axis-angle
        r = eulerAnglesToRotationMatrix_torch(a)    #()
        
        t = self.t[pose_id]  # (3, )
        return r, t
    
    def create_r_t(self, store_dict):
        r = np.zeros((len(store_dict), 3))
        r_pred = np.zeros((len(store_dict), 3))
        r_truth = np.zeros((len(store_dict), 3))
        
        t = np.zeros((len(store_dict), 3))
        t_pred = np.zeros((len(store_dict), 3))
        t_truth = np.zeros((len(store_dict), 3))
        for i in range(len(store_dict)):
            temp = store_dict[i]
            r[i] = temp['rot_pred']
            t[i] = temp['trans_pred']
            r_pred[i] = temp['rot_pred']
            t_pred[i] = temp['trans_pred']
            r_truth[i] = temp['rot_ground']
            t_truth[i] = temp['trans_ground']
        self.r = nn.Parameter(torch.from_numpy(r).float(), requires_grad=self.learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.from_numpy(t).float(), requires_grad=self.learn_t)  # (N, 3)
        self.r_pred = nn.Parameter(torch.from_numpy(r_pred).float(), requires_grad=False)  # (N, 3)
        self.t_pred = nn.Parameter(torch.from_numpy(t_pred).float(), requires_grad=False)  # (N, 3)
        self.r_truth = nn.Parameter(torch.from_numpy(r_truth).float(), requires_grad=False)  # (N, 3)
        self.t_truth = nn.Parameter(torch.from_numpy(t_truth).float(), requires_grad=False)  # (N, 3)
    
    def _all_input(self):
        return self.r, self.t, self.r_pred, self.t_pred, self.r_truth, self.t_truth




    
class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bias=True, w0=1., is_first=False):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.activation = Sine(w0)
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = w0
        self.c = 6
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
            self.layer.weight.uniform_(-w_std, w_std)
            if self.layer.bias is not None:
                self.layer.bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = self.layer(x) 
        out = self.activation(out)
        return out
    

    
class SIREN(nn.Module):
    def __init__(self, pos_in_dims, D):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param D:           scalar, number of hidden dimensions
        """
        super(SIREN, self).__init__()

        self.pos_in_dims = pos_in_dims

        self.layers0 = nn.Sequential(
            SirenLayer(pos_in_dims, D, use_bias=True, w0=30., is_first=True),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
        )
        
        self.layers1 = nn.Sequential(
            SirenLayer(D+pos_in_dims, D, use_bias=True, w0=1., is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
        )

        # self.fc_density = nn.Linear(D, 1)
        self.fc_feature = nn.Linear(D, D)
        self.img_layers = SirenLayer(D, D//2, use_bias=True, w0=1., is_first=False)
        self.fc_img = nn.Linear(D//2, 1)

        # self.fc_density.bias.data = torch.tensor([0.1]).float()
        self.fc_img.bias.data = torch.tensor([0.02]).float()

    def forward(self, pos_enc):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :return: rgb_density (H, W, N_sample, 1)
        """
        x = self.layers0(pos_enc)  # (H, W, N_sample, D)
        x = torch.cat([x, pos_enc], dim=-1)  # (H, W, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (H, W, N_sample, D)

        feat = self.fc_feature(x)  # (H, W, N_sample, D)
        # x = torch.cat([feat, dir_enc], dim=3)  # (H, W, N_sample, D+dir_in_dims)
        x = self.img_layers(feat)  # (H, W, N_sample, D/2)
        img = self.fc_img(x)  # (H, W, N_sample, 1)

        return img


class SIREN_NeRF(nn.Module):
    def __init__(self, pos_in_dims, dir_in_dims, D, isotropic_intensity=False):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(SIREN_NeRF, self).__init__()

        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims
        self.isotropic_intensity = isotropic_intensity

        self.layers0 = nn.Sequential(
            SirenLayer(pos_in_dims, D, use_bias=True, w0=30., is_first=True),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
        )

        self.layers1 = nn.Sequential(
            SirenLayer(D + pos_in_dims, D, use_bias=True, w0=1., is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
        )

        self.fc_density_feature = nn.Linear(D, D)
        if isotropic_intensity:
            self.intensity_param = nn.Parameter(torch.tensor([1.0])).float()
        else:
            intensity_in_dims = D + dir_in_dims
            self.intensity_layers = SirenLayer(intensity_in_dims, D // 2, use_bias=True, w0=1., is_first=False)
            self.fc_intensity = nn.Linear(D // 2, 1)
            self.fc_intensity.bias.data = torch.tensor([0.02]).float()

    def forward(self, pos_enc, dir_enc):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims)
        :param dir_enc: (H, W, N_sample, dir_in_dims)
        :return: densities and intensities (H, W, N_sample, 2)
        """
        x = self.layers0(pos_enc)  # (H, W, N_sample, D)
        x = torch.cat([x, pos_enc], dim=-1)  # (H, W, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (H, W, N_sample, D)

        density_feat = self.fc_density_feature(x)  # (H, W, N_sample, D)
        density = density_feat[..., 0:1]  # (H, W, N_sample, 1)
        if self.isotropic_intensity:
            intensity = self.intensity_param * torch.ones_like(density)  # (H, W, N_sample, 1)
        else:
            x = torch.cat([density_feat, dir_enc], dim=-1)  # (H, W, N_sample, D+dir_in_dims)
            x = self.intensity_layers(x)  # (H, W, N_sample, D/2)
            intensity = self.fc_intensity(x)  # (H, W, N_sample, 1)

        return torch.cat([density, intensity], dim=-1)
