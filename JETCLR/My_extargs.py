#!/bin/env python3
logfile = "/remote/gpu05/rueschkamp/outputs_from_queue/CLR/my_logfile.txt"
tr_dat_path = "/remote/gpu05/rueschkamp/data/Jandata/Zendoo/train.h5"
val_dat_path = "/remote/gpu05/rueschkamp/data/Jandata/Zendoo/val.h5"
test_dat_path = "/remote/gpu05/rueschkamp/data/Jandata/Zendoo/test.h5"
ratio = 0.2
n_jets = 1e5
n_constit = 50
n_epochs = 100 *5 
batch_size = 128

#stuff for Transformer
model_dim = 1000
output_dim = 1000
n_heads = 4
dim_feedforward = 1000
n_layers = 4
n_head_layers = 2
opt = "adam"
learning_rate = 0.00005

mask = False
cmask = True


#stuff for augmentations
rot = True #rotations

cf = True #collinear_fill

ptd = True#need Torchversion > 1.8
ptcm = 0.1 # all the three for distort_jets!
ptst = 0.1 #

trs = True  #
trsw = 1.0 # Both for translate_jets


#stuff for loop
temperature = 0.10