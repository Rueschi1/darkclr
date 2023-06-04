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
import collections

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F


safefile= "/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/AE/"

# initialise logfile
logfile = open("/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/AE/my_AE_logfile.txt", "a" )
print( "logfile initialised"   )

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
print( "device: " + str( device )  , file=logfile, flush=True  )

#loading in data ------------------------------------------------------------

sys.path.insert(1, '/remote/gpu05/rueschkamp/projects/torch_datasets/')
from top_dataset import My_Dataset
from semi_dataset import SemiV_Dataset
from torch.utils.data import DataLoader

t0 = time.time()
#starting training loader --------------------------------------
n_jets = 1e5
ratio = 0.2
n_constits = 50
batch_size = 128

training_set = SemiV_Dataset(
                                    data_path =args.data_path,
                                    signal_origin= "qcd",
                                    usage= "training",
                                    number_constit=  n_constits,#args.n_constit,
                                    number_of_jets= n_jets,
                                    ratio = ratio
                                    )

dl_training = DataLoader(training_set,batch_size=batch_size, shuffle=True)

t1 = time.time()
print( "time taken to load and preprocess data: "+str( np.round( t1-t0, 2 ) ) + " seconds"   , file=logfile, flush=True)



# set up results directory------------------------------------------------------------------------------------------------------------------





base_dir = "/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/AE/" 
expt_tag = "Test_notdroppingall_original" #args.expt
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

loading_model = True
if loading_model :
    # Create an instance of your model
    net = Transformer( input_dim, args.model_dim, args.output_dim, args.n_heads, args.dim_feedforward, args.n_layers, args.learning_rate, args.n_head_layers, dropout=0.1, opt=args.opt )

    # Load the saved state dictionary
    #state_dict = torch.load("/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/Model_21epochs_1e+04Jets.pt")
    epochs = 300
    n_jets = 1e5
    #state_dict = torch.load("/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/Model_21epochs_1e+04Jets.pt")
    Transformer_filename = f"/remote/gpu05/rueschkamp/outputs_from_queue/AnomCLR/plus/experiments/Runwithnotdroppingall/Model_{epochs}epochs_{n_jets:.0e}Jets.pt"
    state_dict = torch.load(Transformer_filename)
    # Load the state dictionary into the model
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8)
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1000),
            #nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder()
model.to(device);
net.eval()

criterion=nn.MSELoss()
learning_rate = 1e-3
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

# Utilize a named tuple to keep track of scores at each epoch
model_hist = collections.namedtuple('Model','epoch loss val_loss')
model_loss = model_hist(epoch = [], loss = [], val_loss = [])

def calculate_errors(model, criterion, dataloader):
    errors = []
    model.eval() # set model to evaluation mode

    with torch.no_grad(): # turn off gradients since we're only evaluating
        for inputs, label in dataloader:
            x_i = inputs

            x_i =  x_i.transpose(1,2)

            point_on_sphere = net(x_i, use_mask=args.mask, use_continuous_mask=args.cmask )
            outputs = model(point_on_sphere)
            #print(point_on_sphere.shape)
            #outputs = sigmoid(model(point_on_sphere))
            for i in range(len(point_on_sphere)):
                
                loss = criterion(outputs[i], point_on_sphere[i]) 
                errors.extend([loss.to(torch.device("cpu")).numpy()])

    return np.array(errors)

def get_true_labels(data_loader):
    labels = []
    for batch in data_loader:
        _, batch_labels = batch
        labels.extend(batch_labels.to(torch.device("cpu")).numpy().tolist())
    return np.array(labels)

def plottingROC(model,criterion,dataloader,Dataorigin,epoch):    

    output_net_top = calculate_errors(model ,criterion, dataloader)
    labels_net_top = get_true_labels(dataloader)

    fpr, tpr, thresholds = roc_curve(labels_net_top, output_net_top) #getting the data needed to plot the ROC curve
    roc_auc = auc(fpr, tpr) #getting the AUC
    # Plot the ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    lw = 2
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([1e-4, 1.0])  # set the lower limit to 0.0001 for logarithmic y-axis
    plt.xscale('linear')     # set x-axis to linear scale
    #plt.yscale('log')        # set y-axis to logarithmic scale
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve '+ Dataorigin)
    plt.legend(loc="lower right")

    if Dataorigin== "aachen":
        AUC_aachen.append(roc_auc)
        plt.savefig(expt_dir+ f"/AUC_aachen_epoch:{epoch}.pdf",format="pdf")

    if Dataorigin== "heidelberg":
        AUC_heidelberg.append(roc_auc)
        plt.savefig(expt_dir+ f"/AUC_heidelberg_epoch:{epoch}.pdf",format="pdf")

    if Dataorigin== "top":
        AUC_top.append(roc_auc)
        plt.savefig(expt_dir+ f"/AUC_top_epoch:{epoch}.pdf",format="pdf")


testing_set_aachen = SemiV_Dataset(
                                    data_path =args.data_path,
                                    signal_origin= "aachen",
                                    usage= "testing",
                                    number_constit= args.n_constit,
                                    number_of_jets= n_jets,
                                    ratio = ratio
                                    )

dl_AE_testing_aachen = DataLoader(testing_set_aachen,batch_size=args.batch_size, shuffle=False)

testing_set_heidelberg = SemiV_Dataset(
                                    data_path =args.data_path,
                                    signal_origin= "heidelberg",
                                    usage= "testing",
                                    number_constit= args.n_constit,
                                    number_of_jets= 10000,
                                    ratio = ratio
                                    )

dl_AE_testing_heidelberg = DataLoader(testing_set_heidelberg,batch_size=args.batch_size, shuffle=False)

testing_set_top = My_Dataset("/remote/gpu05/rueschkamp/data/Jandata/Zendoo/train.h5",
                                  "/remote/gpu05/rueschkamp/data/Jandata/Zendoo/val.h5",
                                  "/remote/gpu05/rueschkamp/data/Jandata/Zendoo/test.h5",
                                  "testing",
                                  50,
                                  number_of_jets= n_jets)

dl_AE_testing_top = DataLoader(testing_set_top,batch_size=args.batch_size, shuffle=False)

print(f"Testing DataLoader Aachen length: {len(dl_AE_testing_aachen)}", file=logfile, flush=True)
print(f"Testing DataLoader Heidelberg length: {len(dl_AE_testing_heidelberg)}", file=logfile, flush=True)
print(f"Testing DataLoader Top length: {len(dl_AE_testing_top)}", file=logfile, flush=True)


checking_iteration = 5


AUC_aachen = []
AUC_heidelberg =[]
AUC_top =[]




def train(epochs, model, model_loss,train_dataloader):
    try: c = model_loss.epoch[-1]
    except: c = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        if epoch%checking_iteration == 0 and epoch > 0:

            model.eval()
            plottingROC(model,criterion,dl_AE_testing_aachen,"aachen",epoch)
            plottingROC(model,criterion,dl_AE_testing_heidelberg,"heidelberg",epoch)
            plottingROC(model,criterion,dl_AE_testing_top,"top",epoch)
            model.train()
            plt.figure()
            x = np.linspace(0, epoch-1 , epoch)
            plt.plot(x, model_loss.loss, label="loss")
            #plt.yscale("log")
            plt.legend()
            plt.savefig(expt_dir+f"AE_loss_epoch:{epoch}.pdf",format="pdf")

            filename = expt_dir + f"/Model_{epoch}epochs_{n_jets:.0e}Jets.pt"
            torch.save(model.state_dict(), filename)

        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            #preprocess
            x_i = inputs
            time1 = time.time()
            x_i = x_i.transpose(1,2) 
            point_on_sphere = net(x_i, use_mask=args.mask, use_continuous_mask=args.cmask )



            outputs = model(point_on_sphere)
            loss = criterion(outputs, point_on_sphere)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        model_loss.epoch.append(c+epoch)
        model_loss.loss.append(loss.item())
        t = time.time()
        print(f"Epoch {epoch}: loss={epoch_loss:.4f} time taken{round(t1-t,2)}", file=logfile, flush=True)

epochs = 101

train(model =model , epochs = epochs, model_loss =model_loss,train_dataloader=dl_training) #AE




def plottingAUCdevelopment(AUC,checking_iteration,Dataorigin):    

    plt.figure()
    x = np.linspace(1,len(AUC),len(AUC)) * checking_iteration
    if Dataorigin== "aachen":
        plt.plot(x, AUC, label="AUC development aachen")
        plt.legend()
        plt.ylim([0.0, 1.0]) 
        plt.show()
        plt.savefig(expt_dir+ f"/AUC_Aachen_development.pdf",format="pdf")
    elif Dataorigin== "heidelberg":
        plt.plot(x, AUC, label="AUC development heidelberg")
        plt.legend()
        plt.ylim([0.0, 1.0]) 
        plt.show()
        plt.savefig(expt_dir+ f"/AUC_Heidelberg_development.pdf",format="pdf")
    elif Dataorigin== "top":
        plt.plot(x, AUC, label="AUC developmenttop")
        plt.legend()
        plt.ylim([0.0, 1.0]) 
        plt.show()
        plt.savefig(expt_dir+ f"/AUC_Top_development.pdf",format="pdf")



plottingAUCdevelopment(AUC_aachen,checking_iteration,"aachen")
plottingAUCdevelopment(AUC_heidelberg,checking_iteration,"heidelberg")
plottingAUCdevelopment(AUC_top,checking_iteration,"top")