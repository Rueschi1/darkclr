#!/bin/env python3.7

# load custom modules required for CLR training
from modules.TransformerEncoder import Transformer
from modules.ContrastiveLosses import clr_loss,anomclr_loss,anomclr_plus_loss,anomclr_loss_bonus

from modules.my_jet_augs import rotate_jets, distort_jets, rescale_pts, crop_jets, translate_jets, collinear_fill_jets, collinear_fill_jets_fast , shift_pT ,pt_reweight_jet,drop_constits_jet



# import args from extargs.py file
import My_Anom_extargs as args


# load standard python modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc



# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

#starting counter
t0 = time.time()

# initialise logfile
logfile = open("/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/normal/my_logfile.txt", "a" )
print( "logfile initialised" , file=logfile, flush=True  )

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
print( "device: " + str( device )  , file=logfile, flush=True  )




#loading in data ------------------------------------------------------------

sys.path.insert(1, '/remote/gpu05/rueschkamp/projects/torch_datasets/')
from semi_dataset import SemiV_Dataset
from torch.utils.data import DataLoader


#starting training loader --------------------------------------


training_set = SemiV_Dataset(
                                    data_path =args.data_path,
                                    signal_origin= "qcd",
                                    usage= "training",
                                    number_constit= args.n_constit,
                                    number_of_jets= args.n_jets,
                                    ratio = args.ratio
                                    )

dl_training = DataLoader(training_set,batch_size=args.batch_size, shuffle=True)

t1 = time.time()
print( "time taken to load and preprocess data: "+str( np.round( t1-t0, 2 ) ) + " seconds" , file=logfile, flush=True   )


# set up results directory------------------------------------------------------------------------------------------------------------------

base_dir = "/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/normal/" 
expt_tag = "RunDrop0.3allpos_rescaleCheck" #args.expt
expt_dir = base_dir + "experiments/" + expt_tag + "/"
#print(expt_dir)
# check if experiment already exists
if os.path.isdir(expt_dir):
    sys.exit("ERROR: experiment already exists, don't want to overwrite it by mistake")
else:
    os.makedirs(expt_dir)
print("experiment: "+str(expt_tag), file=logfile, flush=True)




#initializing the network 
input_dim = 3 

net = Transformer( input_dim, args.model_dim, args.output_dim, args.n_heads, args.dim_feedforward, args.n_layers, args.learning_rate, args.n_head_layers, dropout=0.1, opt=args.opt )
# send network to device
net.to( device );


#####################To be schafft away#########################To be schafft away#########################To be schafft away#########################To be schafft away####
# Plot the training loss
def plotingstuff(epoch,stars,dashes,denoms):
    x = np.linspace(0, epoch - 1, epoch)

    fig, ax1 = plt.subplots()
    ax1.plot(x, losses, label="loss")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f"AnomCLR Loss with {args.n_jets:.0e} jets")
    ax1.legend()
    
    plt.savefig((expt_dir+f"CLR-Loss_{epoch}epochs_{args.n_jets:.0e}Jets.pdf"), format="pdf")

    # Create a new figure and axes for the second plot
    fig, ax2 = plt.subplots()
    ax2.plot(x, stars, label="Anomaly Similarity")
    ax2.plot(x, dashes, label="Physical Similarity")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Similarity')
    ax2.set_title('Similarity of the Transformer Network')
    ax2.legend()
    plt.savefig((expt_dir+f"Similarities_{epoch}epochs_{args.n_jets:.0e}Jets.pdf"), format="pdf")


    fig, ax3 = plt.subplots()
    ax3.plot(x, denoms, label="Denominator")
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Similarity')
    ax3.set_title('Similarity of the Transformer Network')
    ax3.legend()
    plt.savefig((expt_dir+f"Denominator_{epoch}epochs_{args.n_jets:.0e}Jets.pdf"), format="pdf")
#####################To be schafft away#########################To be schafft away#########################To be schafft away#########################To be schafft away####









print( "starting training loop, running for " + str( args.n_epochs ) + " epochs" +  str( args.n_jets )+"Jets" , file=logfile, flush=True  )
print( "---"  , file=logfile, flush=True  )

losses = []
stars = []
dashes = []
denoms = []

breaker = 1
# the loop
#for epoch in range( args.n_epochs ):
for epoch in range( args.n_epochs+1):
    # initialise timing stats
    losses_e = []
    stars_e = [] #anomaly
    dashes_e = [] # physical
    denoms_e = []

    if (epoch%30) ==0 and epoch> 0:

        tms0 = time.time()

        filename = expt_dir+ f"Model_{epoch}epochs_{args.n_jets:.0e}Jets.pt"
        torch.save(net.state_dict(), filename)
        tms1 = time.time()
        print( f"time taken to save model: {round( tms1-tms0, 1 )}s", file=logfile, flush=True  )

        plotingstuff(epoch,stars,dashes,denoms)



    print("epoch: ",epoch, file=logfile, flush=True )
    for i, (data, labels) in enumerate(dl_training):
        net.optimizer.zero_grad()
        x_i = data
        
        # print(x_i.shape) # checking what Tensor is fed into the augmentations
        x_i = rotate_jets( x_i ) # to undo the previos centring
        x_j = x_i.clone()
        x_k = x_i.clone()


        # POSITIVE AUGMENTATIONS
        x_j = rotate_jets( x_j ) 
        x_j = collinear_fill_jets_fast( x_j )
        x_j = collinear_fill_jets_fast( x_j ) #Why two times?
        x_j = distort_jets( x_j, strength=args.ptst, pT_clip_min=args.ptcm )

        x_i = translate_jets( x_i, width=args.trsw )
        x_j = translate_jets( x_j, width=args.trsw )
        x_k = translate_jets( x_k, width=args.trsw ) # what if we would skip this?

        #x_i = rescale_pts( x_i )
        #x_j = rescale_pts( x_j ) moved to preprocessing


        # NEGATIVE AUGMENTATIONS
        
        #x_k = pt_reweight_jet( x_k)
        x_k = drop_constits_jet(x_k,0.3)

        # Getting representations

        x_i = rescale_pts( x_i )
        x_j = rescale_pts( x_j )
        x_k = rescale_pts( x_k )
        
        x_i = x_i.transpose(1,2)
        x_j = x_j.transpose(1,2)
        x_k = x_k.transpose(1,2)
        
        z_i  = net(x_i, use_mask=args.mask, use_continuous_mask=args.cmask) #dim: x_i = torch.Size([104, 50, 3]) and z_i = torch.Size([104, 1000])
        z_j = net(x_j, use_mask=args.mask, use_continuous_mask=args.cmask)
        z_k = net(x_k,use_mask = args.mask, use_continuous_mask = args.cmask)

        if torch.isnan(z_k).any():
            
            print("Representation, Size:",z_k.size(),'Full Tensor:',z_k , file=logfile, flush=True )
            print("x_i, Size:",x_k.size(),'Full Tensor:',x_k , file=logfile, flush=True  )
            breaker = 0
        # compute the loss, back-propagate, and update scheduler if required
        loss , star , dash , denom = anomclr_loss_bonus( z_i, z_j, z_k,args.temperature )
        loss = loss.to( device )
        loss.backward()
        net.optimizer.step()
        losses_e.append( loss.detach().cpu().numpy() )
        stars_e.append(np.mean(star.detach().cpu().numpy()))
        dashes_e.append(np.mean(dash.detach().cpu().numpy()))
        denoms_e.append(np.mean(denom.detach().cpu().numpy()))
        
        if breaker==0:
            break
        #print(loss)
    if breaker==0:
            break    
        
    
    loss_e = np.mean( np.array( losses_e ) )
    print(loss_e, file=logfile, flush=True )
    losses.append( loss_e )

    star_e = np.mean(np.array(stars_e))
    print(star_e, file=logfile, flush=True )
    stars.append(star_e)

    dash_e = np.mean(np.array(dashes_e))
    print(dash_e, file=logfile, flush=True )
    dashes.append(dash_e)

    denom_e = np.mean(np.array(denoms_e))
    print(dash_e, file=logfile, flush=True )
    denoms.append(denom_e)


t2 = time.time()

print( f"Training done. Time taken : {round( t2-t1, 1 )}s"  , file=logfile, flush=True )

tms0 = time.time()