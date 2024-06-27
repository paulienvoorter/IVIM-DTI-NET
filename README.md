# IVIM-DTI-NET

This repository contains the code regarding our paper: Diffusion-derived intravoxel-incoherent motion anisotropy relates to cerebrospinal fluid and blood flow 

## Description
The aim of this repository is to enable you to use our IVIM-DTI-NET relatively easily on your own multi-b-value multi-directional diffusion-weighted data. 

train_network.py --> trains a selfsupervised physics-informed neural network using multi-b-value multi-directional data, which you can provide yourself or can be downloaded from https://zenodo.org/records/12545278 (note that our code downloads this data automatically in the folder 'data').

After your network is trained, it is being saved in the folder 'trained_networks', and you can observe the corresponding loss curve in the folder 'plots'

Now, you can run predict_IVIM-DTI_parameters.py, which loads the trained network en predict all IVIM-DTI model parameters. 

## Create conda environment
To directly run the code, we added a '.yml' file which can be run in anaconda. To create a conda environment with the '.yml' file enter the command below in the terminal: conda env create -f environment.yml 

This now creates an environment called 'ivimdti' that can be activated by: conda activate ivim

## Authors
Paulien Voorter paulien.voorter@gmail.com | p.voorter@maastrichtuniversity.nl | https://github.com/paulienvoorter

## Acknowledgement

Note that this code is build upon previous repositories, and I would like to the authors for sharing their code:

June 2021        Oliver Gurney-Champion and Misha Kaandorp https://github.com/oliverchampion/IVIMNET

August 2019      Sebastiano Barbieri: https://github.com/sebbarb/deep_ivim

