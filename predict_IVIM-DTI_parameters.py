#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:13:21 2022

@author: g10041175
"""
# this loads all patient data and evaluates it all.
import os
import time
import nibabel as nib
import numpy as np
import IVIMDTINET.deep as deep
import torch
from hyperparams import hyperparams as hp
import sys
import tqdm
from joblib import Parallel, delayed


torch.manual_seed(0)
np.random.seed(0)
arg = hp()

### folder patient data
networkfolder='trained_networks'
networkname='{networkfolder}/{name}-1.pt'.format(name=arg.save_name,networkfolder=networkfolder)
folder = 'data'
# mask data available? 1=yes, 0=no
maskfile = 1

subjectlist=['subject01']

for subjnr in range(len(subjectlist)):
### load patient data
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
        mask1 = nib.load('{folder}/{subj}/brain_mask_dilated.nii'.format(folder=folder,subj=subj))
        mask = mask1.get_fdata() 
        mask4D = np.zeros([sx, sy, sz, n_bval])
        for ii in range(n_bval):
            mask4D[:,:,:,ii] = mask
        datas=datas*mask4D 
    # reshape image for fitting
    X_dw = np.reshape(datas, (sx * sy * sz, n_bval))
    ### select only relevant values, delete background and noise, and normalise data
    S0 = np.nanmean(X_dw[:, selsb], axis=1)
    S0[S0 != S0] = 0
    S0 = np.squeeze(S0)
    valid_id = (S0 > (0.00 * np.median(S0[S0 > 0]))) #we want all id's to be valid now, otherwise no fit structures that have a low signal at b=0
    data2 = X_dw[valid_id, :]
    # normalise data
    S0 = np.nanmean(data2[:, selsb], axis=1).astype('<f')
    data2 = data2 / S0[:, None]
    print('Done! \n Predict IVIM-DTI parameters for all voxels...')
    
    #ensemble part
    bval = torch.FloatTensor(bval[:])
    bvec = torch.FloatTensor(bvec[:])
    #load the trained network
    net = deep.Net(bval, bvec, arg.net_pars)
    net.load_state_dict(torch.load(networkname))
    net.eval()
    #predict the ivim-dti parameters
    paramsNN = deep.predict_IVIM(data2, bval, bvec, net, arg)
        
    print('Done! \n Saving images...')

    
    names = ['Dxx',
             'Dxy',
             'Dxz',
             'Dyy',
             'Dyz',
             'Dzz',
             'Dpxx',
             'Dpxy',
             'Dpxz',
             'Dpyy',
             'Dpyz',
             'Dpzz',
             'f',
             'S0']
    
    #create folder to save parameter maps if it does not exist yet
    pathsubj = '{folder}/{subj}/parammaps_IVIM-DTI-NET'.format(folder=folder,subj=subj)
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathsubj)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathsubj)  
    
    ### D tensor
    D_xx=paramsNN[0]
    D_xy=paramsNN[1]
    D_xz=paramsNN[2]
    D_yy=paramsNN[3]
    D_yz=paramsNN[4]
    D_zz=paramsNN[5]
    
    princ_vector=np.zeros([len(D_xx),3])
    MD=np.zeros(len(D_xx))
    lambda1=np.zeros(len(D_xx))
    lambda2=np.zeros(len(D_xx))
    lambda3=np.zeros(len(D_xx))
    for vox in range(len(D_xx)):
        lambd, eig_vector = np.linalg.eig([[D_xx[vox],D_xy[vox],D_xz[vox]],[D_xy[vox],D_yy[vox],D_yz[vox]],[D_xz[vox],D_yz[vox],D_zz[vox]]]);
        lambd = lambd;
        ind = np.argmax(lambd);            
        princ_vector[vox,:] = eig_vector[:,ind];
        MD[vox] = np.mean(lambd);
        lambda1[vox] = np.max(lambd); #this is also AD
        lambda2[vox] = np.median(lambd);
        lambda3[vox] = np.min(lambd);  

    FA = np.sqrt(3/2) * ( np.sqrt((lambda1 - MD)**2 + (lambda2 - MD)**2 + (lambda3 - MD)**2)) / np.sqrt((lambda1**2 + lambda2**2 + lambda3**2) );
    RD = (lambda2 + lambda3) / 2.0;      

    # fill image array and make nifti
    tot = 0
    new_header = header=data.header.copy()
    new_header.set_slope_inter(1, 0)
    for k in range(len(names)):
        img = np.zeros([sx * sy * sz])
        img[valid_id] = paramsNN[k][tot:(tot + sum(valid_id))]
        img = np.reshape(img, [sx, sy, sz])
        nib.save(nib.Nifti1Image(img, data.affine, mask1.header),'{pathsubj}/{name}.nii'.format(pathsubj=pathsubj,name=names[k]))
    
    #save FA D
    img = np.zeros([sx * sy * sz])
    img[valid_id] = FA[tot:(tot + sum(valid_id))]
    img = np.reshape(img, [sx, sy, sz])
    nib.save(nib.Nifti1Image(img, data.affine, data.header),'{pathsubj}/D_FA.nii'.format(pathsubj=pathsubj))

    #save MD D 
    img = np.zeros([sx * sy * sz])
    img[valid_id] = MD[tot:(tot + sum(valid_id))]
    img = np.reshape(img, [sx, sy, sz])
    nib.save(nib.Nifti1Image(img, data.affine, data.header),'{pathsubj}/D_MD.nii'.format(pathsubj=pathsubj))
    
    #save RD D 
    img = np.zeros([sx * sy * sz])
    img[valid_id] = RD[tot:(tot + sum(valid_id))]
    img = np.reshape(img, [sx, sy, sz])
    nib.save(nib.Nifti1Image(img, data.affine, data.header),'{pathsubj}/D_RD.nii'.format(pathsubj=pathsubj))
    
    #save AD D 
    img = np.zeros([sx * sy * sz])
    img[valid_id] = lambda1[tot:(tot + sum(valid_id))]
    img = np.reshape(img, [sx, sy, sz])
    nib.save(nib.Nifti1Image(img, data.affine, data.header),'{pathsubj}/D_AD.nii'.format(pathsubj=pathsubj))
    
    ### Dp tensor
    Dp_xx=paramsNN[6]
    Dp_xy=paramsNN[7]
    Dp_xz=paramsNN[8]
    Dp_yy=paramsNN[9]
    Dp_yz=paramsNN[10]
    Dp_zz=paramsNN[11]
    
    princ_vectorp=np.zeros([len(Dp_xx),3])
    MDp=np.zeros(len(Dp_xx))
    lambda1p=np.zeros(len(Dp_xx))
    lambda2p=np.zeros(len(Dp_xx))
    lambda3p=np.zeros(len(Dp_xx))
    for vox in range(len(Dp_xx)):
        lambdp, eig_vectorp = np.linalg.eig([[Dp_xx[vox],Dp_xy[vox],Dp_xz[vox]],[Dp_xy[vox],Dp_yy[vox],Dp_yz[vox]],[Dp_xz[vox],Dp_yz[vox],Dp_zz[vox]]]);
        lambdp = lambdp;
        indp = np.argmax(lambdp);            
        princ_vectorp[vox,:] = eig_vectorp[:,indp];
        MDp[vox] = np.mean(lambdp);
        lambda1p[vox] = np.max(lambdp); #this is also AD
        lambda2p[vox] = np.median(lambdp);
        lambda3p[vox] = np.min(lambdp);  
   
    
    FAp = np.sqrt(3/2) * ( np.sqrt((lambda1p - MDp)**2 + (lambda2p - MDp)**2 + (lambda3p - MDp)**2)) / np.sqrt((lambda1p**2 + lambda2p**2 + lambda3p**2) );
    RDp = (lambda2p+lambda3p)/2.0

    #save FA Dp
    img = np.zeros([sx * sy * sz])
    img[valid_id] = FAp[tot:(tot + sum(valid_id))]
    img = np.reshape(img, [sx, sy, sz])
    nib.save(nib.Nifti1Image(img, data.affine, data.header),'{pathsubj}/Dp_FA.nii'.format(pathsubj=pathsubj))

    #save MD Dp
    img = np.zeros([sx * sy * sz])
    img[valid_id] = MDp[tot:(tot + sum(valid_id))]
    img = np.reshape(img, [sx, sy, sz])
    nib.save(nib.Nifti1Image(img, data.affine, data.header),'{pathsubj}/Dp_MD.nii'.format(pathsubj=pathsubj))
    
    #save AD Dp
    img = np.zeros([sx * sy * sz])
    img[valid_id] = lambda1p[tot:(tot + sum(valid_id))]
    img = np.reshape(img, [sx, sy, sz])
    nib.save(nib.Nifti1Image(img, data.affine, data.header),'{pathsubj}/Dp_AD.nii'.format(pathsubj=pathsubj))
    
    #save RD Dp
    img = np.zeros([sx * sy * sz])
    img[valid_id] = RDp[tot:(tot + sum(valid_id))]
    img = np.reshape(img, [sx, sy, sz])
    nib.save(nib.Nifti1Image(img, data.affine, data.header),'{pathsubj}/Dp_RD.nii'.format(pathsubj=pathsubj))

    print('Done!')















