import os
import sys
import numpy as np
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

def ptp(input, dim=None, keepdim=False):
    if dim is None:
        return input.max() - input.min()
    return input.max(dim, keepdim).values - input.min(dim, keepdim).values




def translate_jets(batch, width=1.0):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of eta-phi translated jets, same shape as input
    '''

    device = batch.device
    mask = (batch[:, 0] > 0)  # 1 for constituents with non-zero pT, 0 otherwise
    ptp_eta = ptp(batch[:, 1, :], dim=-1, keepdim=True)  # ptp = 'peak to peak' = max - min
    ptp_phi = ptp(batch[:, 2, :], dim=-1, keepdim=True)  # ptp = 'peak to peak' = max - min
    low_eta = -width * ptp_eta
    high_eta = +width * ptp_eta
    low_phi = torch.max(-width * ptp_phi, -torch.tensor(np.pi, device=device) - torch.amin(batch[:, 2, :], dim=-1, keepdim=True))
    high_phi = torch.min(+width * ptp_phi, +torch.tensor(np.pi, device=device) - torch.amax(batch[:, 2, :], dim=-1, keepdim=True))
    shift_eta = mask * torch.rand((batch.shape[0], 1), device=device) * (high_eta - low_eta) + low_eta
    shift_phi = mask * torch.rand((batch.shape[0], 1), device=device) * (high_phi - low_phi) + low_phi
    shift = torch.stack([torch.zeros((batch.shape[0], batch.shape[2]), device=device), shift_eta, shift_phi], dim=1)
    shifted_batch = batch + shift
    return shifted_batch

def rotate_jets(batch):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets rotated independently in eta-phi, same shape as input
    '''
    device = batch.device
    batch_size = batch.size(0)

    rot_angle = torch.rand(batch_size, device=device) * 2 * np.pi
    c = torch.cos(rot_angle)
    s = torch.sin(rot_angle)
    o = torch.ones_like(rot_angle)
    z = torch.zeros_like(rot_angle)

    #print(o.shape)

    rot_matrix = torch.stack([
    torch.stack([o, z, z], dim=0),
    torch.stack([z, c, -s], dim=0),
    torch.stack([z, s, c], dim=0)], dim=1) # (3, 3, batch_size]

    #print(rot_matrix[:,:,0])

    return torch.einsum('ijk,lji->ilk', batch, rot_matrix)

def normalise_pts(batch):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-normalised jets, pT in each jet sums to 1, same shape as input
    '''
    batch_norm = batch.clone()
    batch_norm[:, 0, :] = torch.nan_to_num(batch_norm[:, 0, :] / torch.sum(batch_norm[:, 0, :], dim=1)[:, None], posinf=0.0, neginf=0.0)
    return batch_norm

def rescale_pts(batch):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-rescaled jets, each constituent pT is rescaled by 600, same shape as input
    '''
    batch_rscl = batch.clone()
    batch_rscl[:,0,:] = torch.nan_to_num(batch_rscl[:,0,:]/600, nan=0.0, posinf=0.0, neginf=0.0)
    return batch_rscl

def crop_jets( batch, nc ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of cropped jets, each jet is cropped to nc constituents, shape (batchsize, 3, nc)
    '''
    batch_crop = batch.clone()
    return batch_crop[:,:,0:nc]

def distort_jets(batch, strength=0.1, pT_clip_min=0.1):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with each constituents position shifted independently, shifts drawn from normal with mean 0, std strength/pT, same shape as input
    '''
    pT = batch[:, 0].to(device)  # (batchsize, n_constit)
    
    #print(pT.device)
    shift_eta = torch.nan_to_num(
        strength * torch.randn(batch.shape[0], batch.shape[2]).to(device) / pT.clip(min=pT_clip_min),
        posinf=0.0,
        neginf=0.0,
    )  # * mask
    shift_phi = torch.nan_to_num(
        strength * torch.randn(batch.shape[0], batch.shape[2]).to(device) / pT.clip(min=pT_clip_min),
        posinf=0.0,
        neginf=0.0,
    )  # * mask
    shift = torch.stack([torch.zeros((batch.shape[0], batch.shape[2])).to(device), shift_eta, shift_phi], 1)
    #print(shift.device)
    shift.to(device)
    return batch + shift

def collinear_fill_jets(batch):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with collinear splittings, the function attempts to fill as many of the zero-padded args.nconstit
    entries with collinear splittings of the constituents by splitting each constituent at most once, same shape as input
    '''
    #device = batch.device
    batch_size = batch.size(0)
    n_constit = batch.size(2)
    
    batchb = batch.clone()
    nzs = torch.sum(batch[:,:,0]>0, dim=1)
    nzs1 = torch.max(torch.cat([nzs.view(batch_size, 1), torch.full((batch_size, 1), int(n_constit/2), dtype=torch.int64, device=device)], dim=1), dim=1).values
    zs1 = n_constit - nzs1
    
    for k in range(batch_size):
        els = torch.randperm(nzs1[k], device=device)[:zs1[k]]
        rs = torch.rand(zs1[k], device=device)
        for j in range(zs1[k]):
            batchb[k,0,els[j]] = rs[j]*batch[k,0,els[j]]
            batchb[k,0,nzs[k]+j] = (1-rs[j])*batch[k,0,els[j]]
            batchb[k,1,nzs[k]+j] = batch[k,1,els[j]]
            batchb[k,2,nzs[k]+j] = batch[k,2,els[j]]
            
    return batchb

def collinear_fill_jets_fast(batch):
    '''
    Fill as many of the zero-padded entries with collinear splittings
    of the constituents by splitting each constituent at most once.
    Parameters
    ----------
    batch : torch.Tensor
        batch of jets with zero-padding
    Returns
    -------
    batch_filled : torch.Tensor
        batch of jets with collinear splittings
    '''
    batch_filled = batch.clone()
    n_constit = batch_filled.shape[2]
    n_nonzero = torch.sum(batch_filled[:,0,:]>0, dim=1)
    
    n_split = torch.min(torch.stack([n_nonzero, n_constit-n_nonzero], dim=1), dim=1).values
    idx_flip = torch.where(n_nonzero != n_split)[0]
    mask_split = (batch_filled[:,0,:] != 0)
    
    mask_split[idx_flip] = torch.flip(mask_split[idx_flip].float(), dims=[1]).bool()

    print(mask_split)
    mask_split[idx_flip] = ~mask_split[idx_flip]
    r_split = torch.rand(size=mask_split.shape, device=batch.device)
    
    a = r_split*mask_split*batch_filled[:,0,:]
    b = (1-r_split)*mask_split*batch_filled[:,0,:]
    c = ~mask_split*batch_filled[:,0,:]
    batch_filled[:,0,:] = a+c+torch.flip(b, dims=[1])
    batch_filled[:,1,:] += torch.flip(mask_split*batch_filled[:,1,:], dims=[1])
    batch_filled[:,2,:] += torch.flip(mask_split*batch_filled[:,2,:], dims=[1])
    return batch_filled


#-------------------------------------_ExperimentCorner_---------------------------------------#


def phi_rotate_jets(batch):

    rot_batch = batch # is this right when it should stay on gpu?
    rot_batch = rot_batch.to(device)
    batch_size = batch.size(0)
    constit = batch.size(2)

    rotate_tensor = torch.rand([batch_size,constit]) * 2 * np.pi #creating the array of random rotations
    rotate_tensor = rotate_tensor.to(device)

    rot_batch[:,2,:] =+ np.pi # shifting the phi tensor to make use of the % function
    rot_batch[:,2,:] += rotate_tensor
    rot_batch[:,2,:] %= 2 * np.pi # getting it in the same output range
    rot_batch[:,2,:] =- np.pi # shifting back

    return rot_batch

def collinear_fill_jets_new(batch):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with collinear splittings, the function attempts to fill as many of the zero-padded args.nconstit
    entries with collinear splittings of the constituents by splitting each constituent at most once, same shape as input
    '''
    batch_size = batch.size(0)
    n_constit = batch.size(2)
    device = batch.device

    # Compute number of nonzero pT constituents for each jet
    nzs = torch.sum(batch[:,:,0]>0, dim=1)

    # Compute number of constituents to split for each jet
    zs1 = n_constit - torch.max(
        torch.stack([nzs, torch.full((batch_size,), int(n_constit/2), dtype=torch.int64, device=device)], dim=1),
        dim=1
    ).values

    # Compute indices of constituents to split for each jet
    mask = zs1 > 0
    els = torch.rand(zs1.sum(), batch_size, device=device).argsort(dim=0)[:zs1.sum()].split(zs1.tolist())
    els = [e.transpose(0, 1) for e in els if len(e) > 0]
    els = torch.cat(els)[:, mask]



    # Compute random split fractions for each jet and each split constituent
    rs = torch.rand(zs1.sum(), device=device)

    # Construct tensors of random split fractions and corresponding constituent pT values
    rs_tiled = rs.repeat_interleave(2)
    pts_tiled = torch.cat([batch[:,0,:nzs.max()], torch.zeros((batch_size, n_constit-nzs.max()), device=device)], dim=1)[:, els.view(-1)].flatten()
    pts_split = torch.stack([rs_tiled*pts_tiled, (1-rs_tiled)*pts_tiled], dim=1)

    # Create tensor of indices where split pT values should be inserted
    indices = (nzs.view(batch_size, 1) + torch.arange(zs1.max(), device=device).view(1, -1)).unsqueeze(1).repeat_interleave(2, dim=1)

    # Create tensor of values to insert
    values = pts_split.flatten()

    # Use scatter_add to update batch of jets with split constituents
    batchb = batch.clone()
    batchb.scatter_add_(2, indices, values.view(batch_size, 2, zs1.max()*2))

    return batchb
