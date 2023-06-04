#!/bin/env python3.7

# load custom modules required for CLR training
from modules.TransformerEncoder import Transformer
from modules.ContrastiveLosses import clr_loss,anomclr_loss,anomclr_plus_loss , anomclr_plus_loss_bonus
from modules.fcn_linear import fully_connected_linear_network
from modules.my_jet_augs import rotate_jets, distort_jets, rescale_pts, crop_jets, translate_jets, collinear_fill_jets, collinear_fill_jets_fast , shift_pT ,pt_reweight_jet, drop_constits_jet



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

# initialise logfile
logfile = open("/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/LCT/my_LCT_logfile.txt", "a" )
print( "logfile initialised"   )

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
print( "device: " + str( device )    )

#loading in data ------------------------------------------------------------

sys.path.insert(1, '/remote/gpu05/rueschkamp/projects/torch_datasets/')
from top_dataset import My_Dataset
from semi_dataset import SemiV_Dataset
from torch.utils.data import DataLoader

t0 = time.time()
#starting training loader --------------------------------------

ratio =0.2

#initializing the network 
input_dim = 3 

net = Transformer( input_dim, args.model_dim, args.output_dim, args.n_heads, args.dim_feedforward, args.n_layers, args.learning_rate, args.n_head_layers, dropout=0.1, opt=args.opt )
# send network to device
net.to( device );

loading_model = True
if loading_model :
    # Create an instance of your model
    net = Transformer( input_dim, args.model_dim, args.output_dim, args.n_heads, args.dim_feedforward, args.n_layers, args.learning_rate, args.n_head_layers, dropout=0.1, opt=args.opt )

    # Load the saved state dictionary
    #state_dict = torch.load("/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/Model_21epochs_1e+04Jets.pt")
    epochs = 60
    n_jets =1e5
    Transformer_filename = f"/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/plus/Model_{epochs}epochs_{n_jets:.0e}Jets.pt"
    state_dict = torch.load(Transformer_filename)
    # Load the state dictionary into the model
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

net.eval();

#####################################Training###########################
n_jets = 50000
LCT_training_set_aachen = SemiV_Dataset(
                                    data_path =args.data_path,
                                    signal_origin= "aachen",
                                    usage= "training",
                                    number_constit= args.n_constit,
                                    number_of_jets= n_jets,
                                    ratio = ratio
                                    )

dl_LCT_training_aachen = DataLoader(LCT_training_set_aachen,batch_size=args.batch_size, shuffle=True)

LCT_training_set_heidelberg = SemiV_Dataset(
                                    data_path =args.data_path,
                                    signal_origin= "heidelberg",
                                    usage= "training",
                                    number_constit= args.n_constit,
                                    number_of_jets= n_jets,
                                    ratio = ratio
                                    )

dl_LCT_training_heidelberg = DataLoader(LCT_training_set_heidelberg,batch_size=args.batch_size, shuffle=True)

LCT_training_top = My_Dataset("/remote/gpu05/rueschkamp/data/Jandata/Zendoo/train.h5",
                                  "/remote/gpu05/rueschkamp/data/Jandata/Zendoo/val.h5",
                                  "/remote/gpu05/rueschkamp/data/Jandata/Zendoo/test.h5",
                                  "training",
                                  50,
                                  number_of_jets= n_jets)

dl_LCT_training_top = DataLoader(LCT_training_top,batch_size=args.batch_size, shuffle=True)

#####################################Testing###################################

LCT_testing_set_aachen = SemiV_Dataset(
                                    data_path =args.data_path,
                                    signal_origin= "aachen",
                                    usage= "testing",
                                    number_constit= args.n_constit,
                                    number_of_jets= n_jets,
                                    ratio = ratio
                                    )

dl_LCT_testing_aachen = DataLoader(LCT_testing_set_aachen,batch_size=args.batch_size, shuffle=False)

LCT_testing_set_heidelberg = SemiV_Dataset(
                                    data_path =args.data_path,
                                    signal_origin= "heidelberg",
                                    usage= "testing",
                                    number_constit= args.n_constit,
                                    number_of_jets= n_jets,
                                    ratio = ratio
                                    )

dl_LCT_testing_heidelberg = DataLoader(LCT_testing_set_heidelberg,batch_size=args.batch_size, shuffle=False)

LCT_testing_top = My_Dataset("/remote/gpu05/rueschkamp/data/Jandata/Zendoo/train.h5",
                                  "/remote/gpu05/rueschkamp/data/Jandata/Zendoo/val.h5",
                                  "/remote/gpu05/rueschkamp/data/Jandata/Zendoo/test.h5",
                                  "testing",
                                  50,
                                  number_of_jets= n_jets)

dl_LCT_testing_top = DataLoader(LCT_testing_top,batch_size=args.batch_size, shuffle=False)


t1 = time.time()
print( "time taken to load and preprocess data: "+str( np.round( t1-t0, 2 ) ) + " seconds" , file=logfile, flush=True   )

linear_input_size = args.output_dim
linear_n_epochs = 750
linear_learning_rate = 0.001
linear_batch_size = 128
linear_opt = "adam"


linear_model_aachen = fully_connected_linear_network( linear_input_size, 1, linear_opt, linear_learning_rate )
linear_model_aachen = linear_model_aachen.to(device)

linear_model_heidelberg = fully_connected_linear_network( linear_input_size, 1, linear_opt, linear_learning_rate )
linear_model_heidelberg = linear_model_heidelberg.to(device)


criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCELoss()
#sigmoid = nn.Sigmoid()
linear_learning_rate = 0.0001

# Utilize a named tuple to keep track of scores at each epoch
import collections # whats this???


def train(epochs, model, criterion, train_dataloader):

    model_hist = collections.namedtuple('Model','epoch loss')
    model_loss = model_hist(epoch = [], loss = [])

    try:
        c = model_loss.epoch[-1]
    except:
        c = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for i, (inputs, labels) in enumerate(train_dataloader):
            # Zero the parameter gradients
            model.optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            # inputs = inputs.to(device)
            labels = labels.to(device)

            #     Stuff from earlier loop
            x_i = inputs
            time1 = time.time()

            x_i = x_i.transpose(1,2) #this should not be a thing lol WTF ______-----____
            
            #print(x_i.shape)
            point_on_sphere = net(x_i, use_mask=args.mask, use_continuous_mask=args.cmask )


            outputs =model(point_on_sphere)
            #outputs = sigmoid(model(point_on_sphere))

            loss = criterion(outputs, labels.unsqueeze(-1).float())
            loss.backward()
            model.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        model_loss.epoch.append(c+epoch)
        model_loss.loss.append(loss.item())
        print(f"Epoch {epoch}: loss={epoch_loss:.4f}", file=logfile, flush=True)


epochs_LCT = 100


print("Aachen Training:", file=logfile, flush=True)
train(epochs_LCT,linear_model_aachen,criterion,dl_LCT_training_aachen) 

x = np.linspace(0,epochs_LCT-1 , epochs_LCT)
plt.plot(x, model_loss.loss, label="loss aachen")
plt.legend()
plt.savefig("/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/LCT/Loss_Aachen.pdf",format="pdf")


def calculate_errors(CLR_model,LCT_model, dataloader):
    errors =[]
    output =[]
    #print(CLR_model)
    #print(LCT_model)
    CLR_model.eval() # set model to evaluation mode
    LCT_model.eval()
    with torch.no_grad(): # turn off gradients since we're only evaluating
        for inputs, labels in dataloader:
            
            x_i = inputs

            x_i =  x_i.transpose(1,2)
            
            point_on_sphere = CLR_model(x_i, use_mask=args.mask, use_continuous_mask=args.cmask)
            outputs = LCT_model(point_on_sphere).to(torch.device("cpu"))
            
            output.extend(outputs)

    return np.array(output)

def get_true_labels(data_loader):
    labels = []
    for batch in data_loader:
        _, batch_labels = batch
        labels.extend(batch_labels.to(torch.device("cpu")).numpy().tolist())
    return np.array(labels)

output_net = calculate_errors(net,linear_model_aachen, dl_LCT_testing_aachen)
labels_net = get_true_labels(dl_LCT_testing_aachen)
#print(labels_net)
print(output_net.shape, file=logfile, flush=True)

fpr, tpr, thresholds = roc_curve(labels_net, output_net) #getting the data needed to plot the ROC curve
roc_auc = auc(fpr, tpr) #getting the AUC

# Plot the ROC curve
fig, ax = plt.subplots(figsize=(8, 6))
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([1e-4, 1.0])  # set the lower limit to 0.0001 for logarithmic y-axis
plt.xscale('linear')     # set x-axis to linear scale
#plt.yscale('log')        # set y-axis to logarithmic scale
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve Aachen')
plt.legend(loc="lower right")
plt.savefig("/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/LCT/AUC_LCT_Aachen.pdf",format="pdf")


print("Heidelberg Training:", file=logfile, flush=True)
train(epochs_LCT,linear_model_heidelberg,criterion,dl_LCT_training_heidelberg) 

x = np.linspace(0,epochs_LCT-1 , epochs_LCT)
plt.plot(x, model_loss.loss, label="loss heidelberg")
plt.legend()
plt.savefig("/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/LCT/Loss_Heidelberg.pdf",format="pdf")

output_net = calculate_errors(net,linear_model_heidelberg, dl_LCT_testing_heidelberg)
labels_net = get_true_labels(dl_LCT_testing_heidelberg)
#print(labels_net)
print(output_net.shape, file=logfile, flush=True)

fpr, tpr, thresholds = roc_curve(labels_net, output_net) #getting the data needed to plot the ROC curve
roc_auc = auc(fpr, tpr) #getting the AUC

# Plot the ROC curve
fig, ax = plt.subplots(figsize=(8, 6))
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([1e-4, 1.0])  # set the lower limit to 0.0001 for logarithmic y-axis
plt.xscale('linear')     # set x-axis to linear scale
#plt.yscale('log')        # set y-axis to logarithmic scale
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve Heidelberg')
plt.legend(loc="lower right")
plt.savefig("/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/LCT/AUC_LCT_Heidelberg",format="pdf")