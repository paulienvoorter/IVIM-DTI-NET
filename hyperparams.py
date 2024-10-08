"""

June 2024 by Paulien Voorter
paulien.voorter@gmail.com 
https://www.github.com/paulienvoorter

Code is uploaded as part of our paper in MRM: 'Diffusion-derived intravoxel-incoherent motion anisotropy relates to CSF and blood flow'

"""
import torch
import numpy as np


#most of these are options from the article and explained in the M&M.
class train_pars:
    def __init__(self):
        self.optim='adam' #these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
        self.lr = 0.00003 # this is the learning rate.
        self.patience= 10 # this is the number of epochs without improvement that the network waits untill determining it found its optimum
        self.batch_size= 128 # number of datasets taken along per iteration
        self.maxit = 500 # max iterations per epoch
        self.split = 0.9 # split of test and validation data
        self.loss_fun = 'rms' # what is the loss used for the model. rms is root mean square (linear regression-like); L1 is L1 normalisation (less focus on outliers)
        self.scheduler = False # LR is important. This approach allows to reduce the LR itteratively when there is no improvement throughout an 5 consecutive epochs
        # use GPU if available
        self.use_cuda = torch.cuda.is_available()
        self.device = self.device = torch.device("cpu") #torch.device("cuda:0" if self.use_cuda else "cpu") --> using gpu should work as well, but I was not able to test this.
        self.select_best = True


class net_pars:
    def __init__(self):
        # select a network setting
        self.dropout = 0.1 #0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
        self.batch_norm = True # False/True turns on batch normalistion
        self.cons_min = [.011, -0.15, 0.0, -.129, -0.213, 0.9]  # V1-3, V4-6, Fp, U1-3, U4-6, S0
        self.cons_max = [0.041, 0.15, 1.0, .297, 0.213, 1.1]  # V1-3, V4-6, Fp, U1-3, U4-6, S0
        self.fitS0 = False #indicates whether to fit S0 (True) or fix it to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
        self.depth = 2 # number of layers
        self.width = 0 # determines network width. Putting to 0 makes it as wide as the number of b-values


class hyperparams:
    def __init__(self):
        self.fig = True # plot and save training and validation loss of network
        self.save_name = 'ivim-dti' 
        self.net_pars = net_pars()
        self.train_pars = train_pars()
