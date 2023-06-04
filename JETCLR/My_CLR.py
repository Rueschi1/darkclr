#!/bin/env python3.7

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
from sklearn.metrics import roc_curve, auc

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# load custom modules required for jetCLR training
from modules.jet_augs import rotate_jets, distort_jets, rescale_pts, crop_jets, translate_jets, collinear_fill_jets
from modules.transformer import Transformer
from modules.losses import contrastive_loss, align_loss, uniform_loss
from modules.perf_eval import get_perf_stats, linear_classifier_test 


from modules.fcn_linear import fully_connected_linear_network


# importing the torch modules
from modules.my_jet_augs import rotate_jets, distort_jets, rescale_pts, crop_jets, translate_jets, collinear_fill_jets_fast

# import args from extargs.py file
import My_extargs as args

#starting counter
t0 = time.time()

# initialise logfile
logfile = open( args.logfile, "a" )
print( "logfile initialised", file=logfile, flush=True )

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
print( "device: " + str( device ), flush=True, file=logfile )

#loading in data ------------------------------------------------------------


#TEST
def img_mom (x, y, weights, x_power, y_power):
    return ((x**x_power)*(y**y_power)*weights).sum()

def centre_jet(x, y, weights):
    x_centroid = img_mom(x, y, weights, 1, 0) / weights.sum()
    y_centroid = img_mom(x, y, weights, 0, 1)/ weights.sum()
    x = x - x_centroid
    y = y - y_centroid
    return x, y


def converter_plus(data_path,num_jets):

    data_frame = pd.read_hdf(data_path, key='table', start=0, stop=num_jets)
    #give names
    max_const = args.n_constit
    feat_list =  ["E","PX","PY","PZ"] 
    cols = ["{0}_{1}".format(feature,constit) for feature in feat_list for constit in range(max_const)]
    #reshape
    vec4 = np.expand_dims(data_frame[cols],axis=-1).reshape(-1, len(feat_list), max_const)
    #getting p_vec and E
    E  = vec4[:,0,:]
    pxs   = vec4[:,1,:]
    pys   = vec4[:,2,:]
    pzs   = vec4[:,3,:]
    # get pT,eta,phi
    pTs = pT(pxs,pys)
    etas = eta(pTs,pzs)
    phis = phi(pxs,pys)

    for i in range(etas.shape[0]):
        etas[i,:], phis[i,:] = centre_jet( etas[i,:], phis[i,:], pTs[i,:] )

    #get them together
    jet_data = np.stack([pTs,etas,phis],axis = 1)
    labels = data_frame["is_signal_new"].to_numpy()
    return torch.Tensor(jet_data).to(device) , torch.Tensor(labels).to(device)

def converter_lorenz(data_path,num_jets):

    data_frame = pd.read_hdf(data_path, key='table', start=0, stop=num_jets)
    #give names
    max_const = args.n_constit
    feat_list =  ["E","PX","PY","PZ"] 
    cols = ["{0}_{1}".format(feature,constit) for feature in feat_list for constit in range(max_const)]
    #reshape
    vec4 = np.expand_dims(data_frame[cols],axis=-1).reshape(-1, len(feat_list), max_const)
    #getting p_vec and E
    E  = vec4[:,0,:]
    pxs   = vec4[:,1,:]
    pys   = vec4[:,2,:]
    pzs   = vec4[:,3,:]
    # get pT,eta,phi
    pTs = pT(pxs,pys)
    etas = eta(pTs,pzs)
    phis = phi(pxs,pys)

    for i in range(etas.shape[0]):
        etas[i,:], phis[i,:] = centre_jet( etas[i,:], phis[i,:], pTs[i,:] )

    phis = (phis.T - phis[:,0]).T
    phis[phis < -np.pi] += 2*np.pi
    phis[phis > np.pi] -= 2*np.pi

    #get them together
    jet_data = np.stack([pTs,etas,phis],axis = 1)
    labels = data_frame["is_signal_new"].to_numpy()
    return torch.Tensor(jet_data).to(device) , torch.Tensor(labels).to(device)
#TEST



#coverter functions -----------------------------

def pT(px,py):
    pT = np.sqrt( px**2 + py**2 )
    return pT

# Calculate pseudorapidity of pixel entries
def eta(pT, pz):
    small = 1e-10
    small_pT = (np.abs(pT) < small)
    small_pz = (np.abs(pz) < small)
    not_small = ~(small_pT | small_pz)
    theta = np.arctan(pT[not_small]/pz[not_small])
    theta[theta < 0] += np.pi
    etas = np.zeros_like(pT)
    etas[small_pz] = 0
    etas[small_pT] = 1e-10
    etas[not_small] = np.log(np.tan(theta/2))
    return etas

# Calculate phi of the pixel entries (in range [-pi,pi])
# phis are returned in radians, np.arctan(0,0)=0 -> zero constituents set to -np.pi
def phi (px, py):
    phis = np.arctan2(py,px)
    phis[phis < 0] += 2*np.pi
    phis[phis > 2*np.pi] -= 2*np.pi
    phis = phis - np.pi 
    return phis

#doing the convertions
def converter(data_path,num_jets):

    data_frame = pd.read_hdf(data_path, key='table', start=0, stop=num_jets)
    #give names
    max_const = args.n_constit
    feat_list =  ["E","PX","PY","PZ"] 
    cols = ["{0}_{1}".format(feature,constit) for feature in feat_list for constit in range(max_const)]
    #reshape
    vec4 = np.expand_dims(data_frame[cols],axis=-1).reshape(-1, len(feat_list), max_const)
    #getting p_vec and E
    E  = vec4[:,0,:]
    pxs   = vec4[:,1,:]
    pys   = vec4[:,2,:]
    pzs   = vec4[:,3,:]
    # get pT,eta,phi
    pTs = pT(pxs,pys)
    etas = eta(pTs,pzs)
    phis = phi(pxs,pys)
    #get them together
    jet_data = np.stack([pTs,etas,phis],axis = 1)
    labels = data_frame["is_signal_new"].to_numpy()
    return torch.Tensor(jet_data).to(device) , torch.Tensor(labels).to(device)

#defining classes ------------------------------------------
    
class My_Dataset(Dataset):
    def __init__(self, training_path , validation_path, test_path, usage, transform=None, target_transform=None):
        #getting data
        num_jets = args.n_jets
        train_data , train_labels = converter_lorenz(training_path,int( num_jets) )
        #val_data , val_labels = converter(validation_path, int(num_jets*args.ratio) ) # for larger datasets below two time *args.ratio in the LCT part!
        #test_data, test_labels = converter(test_path,int(num_jets*args.ratio))
        val_data , val_labels = converter_lorenz(validation_path, int(num_jets) )
        test_data, test_labels = converter_lorenz(test_path,int(num_jets))
        
        # re-scale test data, for the training data this will be done on the fly due to the augmentations
        test_data = rescale_pts( test_data )
        val_data = rescale_pts(val_data)

        if usage== "training" :
            self.labels = train_labels
            self.data = train_data
        elif usage=="validation":
            self.labels = val_labels
            self.data = val_data
        elif usage=="testing":
            self.labels = test_labels
            self.data = test_data
        elif usage=="LCT_train":
            self.labels = test_labels[:int(num_jets*(1-args.ratio))]
            self.data = test_data[:int(num_jets*(1-args.ratio))]
        elif usage=="LCT_test":
            self.labels = test_labels[int(num_jets*(args.ratio)):]
            self.data = test_data[int(num_jets*(args.ratio)):]
        else:
            print("check usage!")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        return data, label
    
    
#starting training loader --------------------------------------

training_set = My_Dataset(args.tr_dat_path,args.val_dat_path,args.test_dat_path,"training")

dl_training = DataLoader(training_set,batch_size=args.batch_size, shuffle=True)

t1 = time.time()
print( "time taken to load and preprocess data: "+str( np.round( t1-t0, 2 ) ) + " seconds", flush=True, file=logfile )


#initializing the network 
input_dim = 3 

net = Transformer( input_dim, args.model_dim, args.output_dim, args.n_heads, args.dim_feedforward, args.n_layers, args.learning_rate, args.n_head_layers, dropout=0.1, opt=args.opt )
# send network to device
net.to( device )

# THE TRAINING LOOP -----------------------------------------------------------

print( "starting training loop, running for " + str( args.n_epochs ) + " epochs", flush=True, file=logfile )
print( "---", flush=True, file=logfile )

losses = []

# initialise timing stats
td1 = 0
td2 = 0
td3 = 0
td4 = 0
td5 = 0
td6 = 0
td7 = 0
td8 = 0

#initialize 
te0 = time.time()
# the loop
for epoch in range( args.n_epochs ):
    
    
    # initialise lists to store batch stats
    loss_align_e = []
    loss_uniform_e = []
    losses_e = []

    # initialise timing stats
    print("epoch: ",epoch , file=logfile, flush=True)

    # the inner loop goes through the dataset batch by batch
    # augmentations of the jets are done on the fly
    for i, (data, labels) in enumerate(dl_training):
        net.optimizer.zero_grad()
        x_i = data
        # x_i = x_i.to(device) #is already on device
        
        time1 = time.time()
        # print(x_i.shape) # checking what Tensor is fed into the augmentations
        #x_i = rotate_jets( x_i )
        x_j = x_i.clone()
        if args.rot:
            x_j = rotate_jets( x_j )
        time2 = time.time()

        if args.cf:
            x_j = collinear_fill_jets_fast( x_j )
            x_j = collinear_fill_jets_fast( x_j ) #Why two times?
        time3 = time.time()

        if args.ptd:
            x_j = distort_jets( x_j, strength=args.ptst, pT_clip_min=args.ptcm )
        time4 = time.time()

        if args.trs:
            x_j = translate_jets( x_j, width=args.trsw )
            x_i = translate_jets( x_i, width=args.trsw ) # Why are both translated?
        time5 = time.time()

        x_i = rescale_pts( x_i )
        x_j = rescale_pts( x_j )
        x_i = x_i.transpose(1,2)
        x_j = x_j.transpose(1,2)
        time6 = time.time()


        z_i  = net(x_i, use_mask=args.mask, use_continuous_mask=args.cmask) #dim: x_i = torch.Size([104, 50, 3]) and z_i = torch.Size([104, 1000])
        z_j = net(x_j, use_mask=args.mask, use_continuous_mask=args.cmask)
        time7 = time.time()

        #future alignment loss here------------
        time8 =time.time()

        # compute the loss, back-propagate, and update scheduler if required
        loss = contrastive_loss( z_i, z_j, args.temperature ).to( device )
        loss.backward()
        net.optimizer.step()
        losses_e.append( loss.detach().cpu().numpy() )
        
        time9 = time.time()

        # update timiing stats
        td1 += time2 - time1
        td2 += time3 - time2
        td3 += time4 - time3
        td4 += time5 - time4
        td5 += time6 - time5
        td6 += time7 - time6
        td7 += time8 - time7
        td8 += time9 - time8

    loss_e = np.mean( np.array( losses_e ) )
    losses.append( loss_e )


#np.save("/remote/gpu05/rueschkamp/outputs_from_queue/CLR/clr_losses.npy", losses )

print(x_j.shape )
print(z_j.shape )

te1= time.time()

print( "JETCLR TRAINING DONE, time taken: " + str( np.round( te1-te0 , 2 ) ) , file=logfile, flush=True   )
print( f"total time taken: {round( te1-te0, 1 )}s, augmentation: {round(td1+td2+td3+td4+td5,1)}s, forward {round(td6, 1)}s, backward {round(td8, 1)}s, other {round(te1-te0-(td1+td2+td3+td4+td6+td7+td8), 2)}s", flush=True, file=logfile )

tms0 = time.time()
torch.save(net.state_dict(),"/remote/gpu05/rueschkamp/outputs_from_queue/CLR/Model.pt")
tms1 = time.time()
print( f"time taken to save model: {round( tms1-tms0, 1 )}s"  , file=logfile, flush=True )



# Plot the training loss
x = np.linspace(0,args.n_epochs-1,args.n_epochs)

plt.plot(x, losses, label = "loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Contrastive Learning Network Loss')
plt.legend()
plt.savefig("/remote/gpu05/rueschkamp/outputs_from_queue/CLR/CLR-Loss.pdf",format="pdf")

"""
print(len(test_set))

print(test_set[0][0].shape)

# Get one batch of data
data_iter = iter(dl_test)
batch = next(data_iter)

# Get one element from the batch
element = batch[0]

# Print the shape of the element
print(element.shape)
"""
# starting Classifier for evaluation --------------------------------------------------------------------------------------

#Dataloader Setup

data_LCT_train = My_Dataset(args.tr_dat_path,args.val_dat_path,args.test_dat_path,"LCT_train",number_of_jets= 1e5)
data_LCT_test = My_Dataset(args.tr_dat_path,args.val_dat_path,args.test_dat_path,"LCT_test",number_of_jets= 1e5/3)

print(len(data_LCT_test))

dl_LCT_training = DataLoader(data_LCT_train,batch_size=args.batch_size, shuffle=True)
dl_LCT_test = DataLoader(data_LCT_test,batch_size=args.batch_size, shuffle=False)

net.eval();

#Network Setup

linear_input_size = args.output_dim
linear_n_epochs = 300
linear_learning_rate = 0.001
linear_batch_size = 128
linear_opt = "adam"

linear_model = fully_connected_linear_network( linear_input_size, 1, linear_opt, linear_learning_rate )
linear_model = linear_model.to(device)

criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCELoss()
#sigmoid = nn.Sigmoid()
linear_learning_rate = 0.001

# Utilize a named tuple to keep track of scores at each epoch
import collections # whats this???
model_hist = collections.namedtuple('Model','epoch loss')
model_loss = model_hist(epoch = [], loss = [])


def train(epochs, model, criterion, train_dataloader):
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


            outputs = model(point_on_sphere)
            #outputs = sigmoid(model(point_on_sphere))

            loss = criterion(outputs, labels.unsqueeze(-1).float())
            loss.backward()
            model.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        model_loss.epoch.append(c+epoch)
        model_loss.loss.append(loss.item())
        t = time.time()
        print(f"Epoch {epoch}: loss={epoch_loss:.4f} time taken{round(tc1-t,2)}", flush=True, file=logfile)

t_c1 = time.time()
train(linear_n_epochs,linear_model,criterion,dl_LCT_training) 
t_c2 = time.time()



print("Classifier done afeter: " +str( np.round( t_c2 - t_c1, 2 ) ) +" s",file=logfile, flush=True)

# Plot the training loss for the classifier
plt.figure()
x = np.linspace(0,linear_n_epochs-1,linear_n_epochs)

plt.plot(x, model_loss.loss, label = "loss of classifier")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Linear Classifier Loss')
plt.legend()
plt.savefig("/remote/gpu05/rueschkamp/outputs_from_queue/CLR/Loss.pdf",format="pdf")

def calculate_outputs(CLR_model,LCT_model, dataloader):
    errors = []
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

output_net =  calculate_outputs(net, linear_model , dl_LCT_test)
labels_net = get_true_labels(dl_LCT_test)

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
plt.title('Receiver operating characteristic curve, 100 epochs Transformer and 300 Classifier, 1e5 jets')
plt.legend(loc="lower right")
plt.savefig("/remote/gpu05/rueschkamp/outputs_from_queue/CLR//ROC.pdf",format= "pdf" )




