#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
June 2024 by Paulien Voorter
paulien.voorter@gmail.com 
https://www.github.com/paulienvoorter

Code is uploaded as part of our paper: 'Diffusion-derived intravoxel-incoherent motion anisotropy relates to cerebrospinal fluid and blood flow'

"""

# this loads all patient data and evaluates it all.
import os
import time
import nibabel as nib
import numpy as np
import IVIMDTINET.deep as deep
import torch
from hyperparams import hyperparams as hp
import download_data

### download data from zenodo if it is nonexisting
download_data.download_data()


torch.manual_seed(0)
np.random.seed(0)

# nr of networks you want to train
repeats=3 # you can train multiple networks and observe their loss function in the 'plots' folder
# define folder patient 
folder = 'data'
# mask data available? 1=yes, 0=no
maskfile = 1

arg = hp()

testdata = False

subjectlist=['subject01']

for subjnr in range(len(subjectlist)):
### load data 
    subj=subjectlist[subjnr]
    print('Load subject data {subj}... '.format(subj=subj))
    # load and init b-values
    bvec = np.genfromtxt('{folder}/{subj}/tensorIVIM.bvec'.format(folder=folder,subj=subj))
    bval = np.genfromtxt('{folder}/{subj}/tensorIVIM.bval'.format(folder=folder,subj=subj))
    selsb = np.array(bval) == 0
    #load nifti
    data = nib.load('{folder}/{subj}/IVIM_EPI_SM_EC_corr.nii'.format(folder=folder,subj=subj))
    datas = data.get_fdata() 
    sx, sy, sz, n_bval = datas.shape 
    if maskfile==1:
        #load mask if it exists--> select only relevant values, delete background and noise
        mask = nib.load('{folder}/{subj}/brain_mask_dilated.nii'.format(folder=folder,subj=subj))
        mask = mask.get_fdata() 
        mask4D = np.zeros([sx, sy, sz, n_bval])
        for ii in range(n_bval):
            mask4D[:,:,:,ii] = mask
        datas=datas*mask4D 
    X_dw = np.reshape(datas, (sx * sy * sz, n_bval)) # reshape image for fitting
    S0 = np.nanmean(X_dw[:, selsb], axis=1)
    S0[S0 != S0] = 0
    S0 = np.squeeze(S0)
    valid_id = (S0 > (0.5 * np.median(S0[S0 > 0])))  # also selects only relevant values, delete background and noise
    data = X_dw[valid_id, :]
    # normalise data
    S0 = np.nanmean(data[:, selsb], axis=1).astype('<f')
    data = data / S0[:, None]
    # stack all data when you train on data from multiple subjects
    if subjnr==0:
        datatot=data
    else:
        datatot=np.concatenate((datatot, data),axis=0)
    print('Done! \n')
        
   
print('All subject data loaded\n')
start_time = time.time()
res = [i for i, val in enumerate(datatot != datatot) if not val.any()] # Remove NaN data

#create folder to save trained networks if it does not exist yet
pathnn = 'trained_networks'
# Check whether the specified path exists or not
isExist = os.path.exists(pathnn)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(pathnn)  

if repeats>1:
    for i in range(repeats):
            net = deep.learn_IVIM(datatot[res], bval, bvec, arg)
            torch.save(net.state_dict(), 'trained_networks/{name}-{i}.pt'.format(name=arg.save_name,i=i))
        
else:
    net = deep.learn_IVIM(datatot[res], bval, bvec, arg)
    torch.save(net.state_dict(), 'trained_networks/{name}.pt'.format(name=arg.save_name))


