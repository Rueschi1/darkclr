import numpy as np

import time
from sklearn import metrics

import torch
import torch.nn as nn

from modules.perf_eval import get_perf_stats

#### function to save out the data and traind model 

######
# Define AE
######

class AutoEncoder( nn.Module ):

    def __init__( self, layer_structure, activation = 'ReLU', use_drop_out=False, 
                    drop_out_rate = 0.1):
        '''
        layer_structure: list or np.array like:
                             [input layer, 1. hidden layer, ... , output layer]
        activation: just jet 'ReLU'
        use_drop_out: True--> after every deep layer ther will be dropout (NOT after input layer)
                      False--> no dropout will be added 
        drop_out_rate: sets the dropoutrate
        
        '''

        super().__init__()
        
        self.layer_structure = layer_structure
        self.activation = activation
        
        self.use_dropout = use_drop_out
        
        if self.use_dropout: 
            self.dropout = nn.Dropout(drop_out_rate)

        if self.activation == 'ReLU': 
            self.activation = nn.ReLU()

        #transform layer_structure list in actual layer structure

        self.module_list = nn.ModuleList()
        for index in range( len(layer_structure) ): 
            if index < ( len(layer_structure)-1 ):
                self.module_list.append(nn.Linear(layer_structure[index], layer_structure[index+1]))
                if index < ( len(layer_structure)-2 ):
                    if index+1 != int(len(layer_structure)/2):
                        if use_drop_out and (index != 0):
                            self.module_list.append(self.dropout)
                        self.module_list.append(self.activation)
        
    def forward(self, data):
        for layer in self.module_list:
            data = layer(data)
        return data

#####
# Define evaluation to see the performance in the log 
#####


def AE_eval(model, val_loader, xdevice, eval_critierion):
    '''Evaluates the AE and returns : AUC, imtafe, loss_eval, label_eval'''

    val_time = time.time()

    model.eval()

    loss_eval = np.array([])
    label_eval = np.array([])

    with torch.no_grad():
        for test_event, test_label in val_loader: 

            test_event = test_event.to(xdevice)
            
            reconst_event = model(test_event)
            loss = eval_critierion(reconst_event, test_event)

            test_label = test_label.numpy()

            loss_per_event = np.mean( loss.detach().cpu().numpy(), axis=1)  #?

            loss_eval = np.concatenate( (loss_eval, loss_per_event), axis=None)
            label_eval = np.concatenate( (label_eval, test_label), axis=None)
    
    auc, imtafe = get_perf_stats(label_eval, loss_eval)
    
    val_fini = time.time()

    val_time = val_fini - val_time

    model.train()
    return auc, imtafe, val_time, loss_eval, label_eval
                
def AE_test(layer_structure, training_data, test_data_dic, rep_layer, sig_names, label_test_data_dic, logfile, saving_dir, activation = 'ReLU', 
                optimizer = 'Adam', learning_rate=0.0001, critierion = 'MSELoss', use_drop_out=False, drop_out_rate = 0.1, 
                training_epochs = 300, AE_batchsize=128 ):
    '''Runs an AE test on the data. Returns: AUC_list, imtafe_list, loss_training, Val_dic
    
        Val_dic: {'Ato4': {epoch_10:{FPR, TPR, threshold, auc, imtafe}, epoch_20:{ ... } ...}}, 'lep': {...}...'''
    #initialise network

    xdevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    AE = AutoEncoder(layer_structure, activation, use_drop_out, drop_out_rate).to(xdevice)

    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(AE.parameters(), lr=learning_rate)
    
    if critierion == 'MSELoss':
        critierion = nn.MSELoss()
        eval_critierion = nn.MSELoss(reduction='none')
    
    #prepare data

    tr_data = torch.from_numpy(training_data).float()
    train_loader = torch.utils.data.DataLoader(tr_data, batch_size=AE_batchsize, shuffle=True)
    
    te_loader_dic = {}
    for key in sig_names:
        if rep_layer == None:
            te_data = torch.from_numpy(test_data_dic[key+'_data']).float()
        else:
            te_data = torch.from_numpy(test_data_dic[key][:, rep_layer, :]).float()
        te_label = torch.from_numpy(label_test_data_dic[key+'_label']).float()

        te_sample = torch.utils.data.TensorDataset(te_data, te_label)


        test_loader = torch.utils.data.DataLoader(te_sample, batch_size=AE_batchsize, shuffle=True)
        te_loader_dic[key] = test_loader
    
    # create validation for background
    if rep_layer == None:
        bg_te_data = test_data_dic['sighp_data'][label_test_data_dic['sighp_label'] == 0]
        bg_te_data = torch.from_numpy(bg_te_data).float()
    else: 
        bg_te_data = test_data_dic['sighp'][:, rep_layer, :][label_test_data_dic['sighp_label'] == 0]
        bg_te_data = torch.from_numpy(bg_te_data).float()
    
    bg_test_loader = torch.utils.data.DataLoader(bg_te_data, batch_size=AE_batchsize, shuffle=True)
    te_loader_dic['bg'] = bg_test_loader

    #train model
    
    AE.train()

    print('--- AE Test ----', file=logfile, flush=True)
    print(' start training ', file=logfile, flush=True)

    training_start = time.time()

    training_loss = []
    AUC_imtafe_dic = {}
    for key in sig_names:
        AUC_imtafe_dic[key+'_AUC'] = []
        AUC_imtafe_dic[key+'_imtafe'] = []
        AUC_imtafe_dic[key+'_loss'] = []
    val_dic = {}
    AUC_imtafe_dic['val_loss'] = []

    ten_ep = time.time()
    for epoch in range(training_epochs):
        # training
        loss = 0
        for batch_features in train_loader: 
            
            batch_features = batch_features.to(xdevice)
            
            optimizer.zero_grad()

            reconstructed = AE(batch_features)

            train_loss = critierion(reconstructed, batch_features)

            train_loss.backward()

            optimizer.step()

            loss += train_loss.detach().cpu().item()     
        training_loss.append(loss/len(train_loader))

        # loging and validation 
        if (epoch+1)%10 == 0: 
            print('····························', file=logfile, flush=True)
            print('   epoch: {}/{}; loss: {}'.format(epoch+1, training_epochs, np.round(loss/len(train_loader), 4) ), file=logfile, flush=True)
            print('   time for 10 epochs: {}'.format(np.round(time.time() - ten_ep, 4)), file=logfile, flush=True)
            
            result = {}
            for key in sig_names:
                auc, imtafe, val_time, loss_val, label_val = AE_eval(AE, te_loader_dic[key], xdevice, eval_critierion)
                AUC_imtafe_dic[key+'_AUC'].append(auc)
                AUC_imtafe_dic[key+'_imtafe'].append(imtafe)
                #computing validation loss
                val_loss = np.mean(loss_val)
                AUC_imtafe_dic[key+'_loss'].append(val_loss)
                # save out data
                result_sig = {}
                fpr, tpr, threshold = metrics.roc_curve( label_val, loss_val )
                result_sig['FPR'] = fpr
                result_sig['TPR'] = tpr
                result_sig['threshold'] = threshold
                result_sig['AUC'] = auc
                result_sig['imtafe'] = imtafe
                result[key] = result_sig
                print('     '+key+': AUC: {}  ;  imtafe: {}'.format(np.round(auc, 4), np.round(imtafe, 4)), file=logfile, flush=True)
            
            val_dic['epoch_'+str( int(epoch+1) )] = result

            # validation loss for bg 
            AE.eval()

            with torch.no_grad():
                loss = 0
                for test_event in te_loader_dic['bg']: 

                    test_event = test_event.to(xdevice)
                    
                    reconst_event = AE(test_event)
                    bg_val_loss = critierion(reconst_event, test_event)

                    loss += bg_val_loss.detach().cpu().item()
                AUC_imtafe_dic['val_loss'].append(loss/len(te_loader_dic['bg']))

            AE.train()


            print('   Validation finished in: {}'.format(np.round(val_time, 4)), file=logfile, flush=True)
            ten_ep = time.time()

    print('    Saving out final AE...', file=logfile, flush=True)
    torch.save(AE.state_dict(), saving_dir + "AE_model_ep" + str(training_epochs) + ".pt")

    training_stop = time.time()

    print(' finished in: {}'.format(np.round(training_stop - training_start, 4)), file=logfile, flush=True)
    print('····························', file=logfile, flush=True)


    return AUC_imtafe_dic, training_loss, val_dic
    






            



