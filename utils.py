import sys
sys.path.append("/cfs/home/u021420/scripts")

import torch
from torch import nn
import torch.optim as optim
import numpy
from torch.utils.data import Dataset
import copy
import random

#mortality dataset for making the dataloaders in data_prep.py
class Mortality_dataset(Dataset):

    def __init__(self, data, padding, seq_len=24, future=6, sines=False, state=False, teacher=False):
        self.data = torch.from_numpy(data).float()
        self.seq_len = seq_len
        self.future = future
        self.padding = padding
        self.sines = sines
        self.state = state
        self.teacher = teacher   

    def __len__(self):
        return len(self.data)-self.seq_len-self.future#-1

    def __getitem__(self,index):

        if self.teacher:
            input = self.data[index : index + self.seq_len, :]
            target = self.data[index : index + self.seq_len + self.future , : ]

        elif self.padding:  
            mask_input =  torch.zeros(self.future , self.data.shape[1])
            
            end_context = index + self.seq_len
            end_pred = index + self.seq_len + self.future

            input = torch.cat(( self.data[ index : end_context , :], mask_input), dim = 0)

            if self.sines and not self.state:
                input[:, -2 : ] = self.data[ index : end_pred , -2 : ] #replacing the padding with the real sine features

            if self.sines and self.state:
                input[:, -7 : ] = self.data[ index : end_pred , -7 : ]#replacing the padding with the real sine and state features
            
            if not self.sines and self.state:
                input[:, -5 : ] = self.data[ index : end_pred , -5 : ]#replacing the padding with the state features

            target =self.data[ index : end_pred]
        
        else:
            input = self.data[index : index + self.seq_len , :]
            target = self.data[index : index + self.seq_len + self.future , : ]

        return input, target

#function that defines the training and testing routines
def Training_Testing(   model_dict, #dictionary with attributes name and model 
                        train_dataloader,
                        test_dataloader,
                        extra_features, #number of extra features that we give the model 2 for sines and 5 for states
                        save_name,
                        future = 6,
                        loss_interval = (None,None), #interval where the loss function is applied in training 
                        teacher_ratio = 0,
                        loss_fn = nn.MSELoss(),
                        raw= False, # if we wish to collect the raw model_out
                        epochs = 100,
                        patience = 500):
                        
    #this function trains the model provided and returns the results of the testing as well as the model itself
    #the fucntion also saves the best model in the directory where it is being run

    #seting up the model, loss function and optimize
    model = model_dict["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = optim.AdamW(model.parameters())

    #initialyzing the results lists
    test_loss_pred1_log = []
    test_loss_pred3_log = []
    test_loss_pred6_log = []
    test_loss_pont_pred6_log = []
    test_loss_pont_pred3_log = []
    train_loss_log = []
    test_loss_log = []

    #epoch counting and stuff for the early stop
    epoch=0
    patience_count = 0
    early_stop = True
    saved_loss = float('inf')

    #stuff that helps in selecting the training pipeline
    SSMs = ["S4", "S4D", "SGconv", "LS4_kb", "LS4_polyb"]
        #extra_features should have 3 possible values -2(sines), -5(state), -7(sin and state), None (neither)
    extra_features = -extra_features if extra_features is not None else extra_features

    while epoch < epochs and early_stop:

        #Training loop
        for batch, (input , target) in enumerate(train_dataloader) : 
            input , target = input.to(device) , target.to(device)
            optimizer.zero_grad() #setting the gradient to 0
            
            if model_dict["name"] == "LSTM":
                teacher = random.random() < teacher_ratio
                if teacher:
                    model_out = model(target, teacher=teacher)
                elif extra_features is not None:
                    model_out = model(input, sine_target = target[:,:,extra_features:])
                else:
                    model_out = model(input)

            else:    
                model_out = model(input) 
                
            loss = loss_fn(torch.squeeze(model_out[ : , loss_interval[0] : loss_interval[1], :extra_features]),
                           torch.squeeze(target[ : , loss_interval[0] : loss_interval[1], :extra_features])) #computing the loss and removing the sine features, since they are known
                          
            loss.backward()
            optimizer.step()

        #Testing loop
        with torch.no_grad():
            input_test , target_test = next(iter(test_dataloader))
            input_test , target_test = input_test.to(device) , target_test.to(device)
            
            if model_dict["name"] == "LSTM":
                
                if extra_features is not None:
                    model_out = model(input_test, sine_target = target_test[:,:,extra_features:])
                else:
                    model_out = model(input_test) 

            else:    
                model_out = model(input_test)
            
            #computing the loss and removing the extra features, since they are known
            loss_test = loss_fn(torch.squeeze(model_out[:, : -future , :extra_features ]), torch.squeeze(target_test[ : , : -future , :extra_features ]))
            test_loss_pred6 = loss_fn(torch.squeeze(model_out[ : , -future : , :extra_features ]), torch.squeeze(target_test[ :, -future : , :extra_features ]))
            test_loss_pred1 = loss_fn(torch.squeeze(model_out[:, -future : -future + 1, :extra_features]), torch.squeeze(target_test[:, -future : -future + 1 , :extra_features]))
            test_loss_pred3 = loss_fn(torch.squeeze(model_out[:, -future : -future + 3, :extra_features]), torch.squeeze(target_test[:, -future : -future + 3, :extra_features]))
            test_loss_pont_pred6 = loss_fn(torch.squeeze(model_out[ : , -1 , :extra_features ]), torch.squeeze(target_test[ :, -1 , :extra_features ]))
            test_loss_pont_pred3 = loss_fn(torch.squeeze(model_out[ : , -4 , :extra_features ]), torch.squeeze(target_test[ :, -4, :extra_features ]))

            test_loss_pred1_log.append(test_loss_pred1.item())
            test_loss_pred3_log.append(test_loss_pred3.item())
            test_loss_pred6_log.append(test_loss_pred6.item())
            test_loss_pont_pred6_log.append(test_loss_pont_pred6.item())
            test_loss_pont_pred3_log.append(test_loss_pont_pred3.item())
            test_loss_log.append(loss_test.item()) 
            train_loss_log.append(loss.item())

            #Early stop 
            if epoch%1 == 0: 
                if test_loss_pred6 <= saved_loss:
                    torch.save(model.state_dict(), save_name)
                    saved_loss = test_loss_pred6
                    saved_epoch = epoch
                    patience_count = 0

                else:
                    patience_count = patience_count + 1
                
                if patience_count >= patience:
                    early_stop = not early_stop

            epoch = epoch + 1

            if not early_stop:
                model.load_state_dict(torch.load(save_name))

    if raw:
        return model, model_out, test_loss_log, test_loss_pred1_log, test_loss_pred3_log, test_loss_pont_pred3_log, test_loss_pred6_log, test_loss_pont_pred6_log, train_loss_log, saved_epoch

    else:   
        return model,test_loss_log, test_loss_pred1_log, test_loss_pred3_log, test_loss_pont_pred3_log, test_loss_pred6_log, test_loss_pont_pred6_log, train_loss_log, saved_epoch

"""
def Training_Testing(   model_dict, #dictionary with attributes name and model 
                        train_dataloader,
                        test_dataloader,
                        extra_features, #number of extra features that we give the model 2 for sines and 5 for states
                        save_name,
                        future = 6,
                        loss_interval = (None,None), #interval where the loss function is applied in training 
                        teacher_ratio = 0,
                        loss_fn = nn.MSELoss(),
                        epochs = 100,
                        patience = 500):
                        
    #this function trains the model provided and returns the results of the testing as well as the model itself
    #the fucntion also saves the best model in the directory where it is being run

    #seting up the model, loss function and optimize
    model = model_dict["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = optim.AdamW(model.parameters())

    #initialyzing the results lists
    test_loss_pred1_log = []
    test_loss_pred3_log = []
    test_loss_pred6_log = []
    test_loss_pont_pred6_log = []
    test_loss_pont_pred3_log = []
    train_loss_log = []
    test_loss_log = []

    #epoch counting and stuff for the early stop
    epoch=0
    patience_count = 0
    early_stop = True
    saved_loss = float('inf')

    #stuff that helps in selecting the training pipeline
    SSMs = ["S4", "S4D", "GConv", "S5"]
        #extra_features should have 3 possible values -2(sines), -5(state), -7(sin and state), None (neither)
    extra_features = -extra_features if extra_features is not None else extra_features

    while epoch < epochs and early_stop:

        #Training loop
        for batch, (input , target) in enumerate(train_dataloader) : 
            input , target = input.to(device) , target.to(device)
            optimizer.zero_grad() #setting the gradient to 0

            if model_dict["name"] == "LSTM":
                teacher = random.random() < teacher_ratio
                if teacher:
                    model_out = model(target, teacher=teacher)
                else:
                    model_out = model(input, sine_target = target[:,:,extra_features:])

            elif model_dict["name"] in SSMs:    
                model_out = model(input) 
                
            loss = loss_fn(torch.squeeze(model_out[ : , loss_interval[0] : loss_interval[1], :extra_features]),torch.squeeze(target[ : , loss_interval[0] : loss_interval[1], :extra_features])) #computing the loss and removing the sine features, since they are known
                          
            loss.backward()
            optimizer.step()

        #Testing loop
        with torch.no_grad():
            input_test , target_test = next(iter(test_dataloader))
            input_test , target_test = input_test.to(device) , target_test.to(device)
            
            if model_dict["name"] == "LSTM":
                model_out = model(input_test, sine_target = target_test[:,:,extra_features:])

            elif model_dict["name"] in SSMs:    
                model_out = model(input_test) 
            
            #computing the loss and removing the extra features, since they are known
            loss_test = loss_fn(torch.squeeze(model_out[:, : -future , :extra_features ]), torch.squeeze(target_test[ : , : -future , :extra_features ]))
            test_loss_pred6 = loss_fn(torch.squeeze(model_out[ : , -future : , :extra_features ]), torch.squeeze(target_test[ :, -future : , :extra_features ]))
            test_loss_pred1 = loss_fn(torch.squeeze(model_out[:, -future : -future + 1, :extra_features]), torch.squeeze(target_test[:, -future : -future + 1 , :extra_features]))
            test_loss_pred3 = loss_fn(torch.squeeze(model_out[:, -future : -future + 3, :extra_features]), torch.squeeze(target_test[:, -future : -future + 3, :extra_features]))
            test_loss_pont_pred6 = loss_fn(torch.squeeze(model_out[ : , -1 , :extra_features ]), torch.squeeze(target_test[ :, -1 , :extra_features ]))
            test_loss_pont_pred3 = loss_fn(torch.squeeze(model_out[ : , -4 , :extra_features ]), torch.squeeze(target_test[ :, -4, :extra_features ]))

            test_loss_pred1_log.append(test_loss_pred1.item())
            test_loss_pred3_log.append(test_loss_pred3.item())
            test_loss_pred6_log.append(test_loss_pred6.item())
            test_loss_pont_pred6_log.append(test_loss_pont_pred6.item())
            test_loss_pont_pred3_log.append(test_loss_pont_pred3.item())
            test_loss_log.append(loss_test.item()) 
            train_loss_log.append(loss.item())

            #Early stop 
            if epoch%1 == 0: 
                if test_loss_pred6 <= saved_loss:
                    torch.save(model.state_dict(), save_name)
                    saved_loss = test_loss_pred6
                    saved_epoch = epoch
                    patience_count = 0

                else:
                    patience_count = patience_count + 1
                
                if patience_count >= patience:
                    early_stop = not early_stop

            epoch = epoch + 1

            if not early_stop:
                model.load_state_dict(torch.load(save_name))


    return model, test_loss_log, test_loss_pred1_log, test_loss_pred3_log, test_loss_pont_pred3_log, test_loss_pred6_log, test_loss_pont_pred6_log, train_loss_log, saved_epoch

#function that defines the training and testing routines
def Training_Testing3(
    model,
    train_dataloader,
    save_name,
    test_dataloader = None,
    loss_interval = (None,None), #interval where the loss function is applied in training 
    teacher_ratio = 0,
    sines = False, #if the sine features are present
    state = False, #if the state features is present
    sinLSTM = False, #if the sine features are present and its a LSTM model
    stateLSTM = False, # if the state feature is present and its an LSTM model
    loss_fn = nn.MSELoss(),
    future = 6,
    epochs = 100,
    patience = 500,
    device = "cuda" if torch.cuda.is_available() else "cpu"):
    #this function trains the model provided and returns the results of the testing as well as the model itself
    #the fucntion also saves the best model in the directory where it is being run
    model.to(device)
    #seting up the model, loss function and optimize
    optimizer = optim.AdamW(model.parameters())
    test_loss_pred1_log = []
    test_loss_pred3_log = []
    test_loss_pred6_log = []
    test_loss_pont_pred6_log = []
    test_loss_pont_pred3_log = []
    train_loss_log = []
    test_loss_log = []
    epoch=0
    patience_count = 0
    early_stop = True
    saved_loss = float('inf')

    while epoch < epochs and early_stop:
      #Training loop
        for batch, (input , target) in enumerate(train_dataloader) : 
            input , target = input.to(device) , target.to(device)
            optimizer.zero_grad() #setting the gradient to 0
            teacher = random.random() < teacher_ratio
            if teacher:
                model_out = model(target, teacher=teacher)

            elif sinLSTM and not stateLSTM:
                model_out = model(input, sine_target = target[:,:,-2:])
            
            elif stateLSTM and not sinLSTM:
                model_out = model(input, sine_target = target[:,:,-5:])
            
            elif sinLSTM and stateLSTM:
                model_out = model(input, sine_target = target[:,:,-7:])

            else:    
                model_out = model(input) #getting the output of the model 
                
            if sines and not state:
                loss = loss_fn(torch.squeeze(model_out[ : , loss_interval[0] : loss_interval[1], :-2 ]),torch.squeeze(target[ : , loss_interval[0] : loss_interval[1], :-2 ])) #computing the loss and removing the sine features, since they are known
            
            elif sines and state:
                loss = loss_fn(torch.squeeze(model_out[ : , loss_interval[0] : loss_interval[1], :-7 ]),torch.squeeze(target[ : , loss_interval[0] : loss_interval[1], :-7 ])) #computing the loss and removing the sine and state features, since they are known
            
            elif not sines and state:
                loss = loss_fn(torch.squeeze(model_out[ : , loss_interval[0] : loss_interval[1], :-5 ]),torch.squeeze(target[ : , loss_interval[0] : loss_interval[1], :-5 ])) #computing the loss and removing the state feature, since they are known

            else:
                loss = loss_fn(torch.squeeze(model_out[ : , loss_interval[0] : loss_interval[1], : ]),torch.squeeze(target[ : , loss_interval[0] : loss_interval[1], : ])) #computing the loss 
                
            loss.backward()
            optimizer.step()

      #esting loop
        with torch.no_grad():
            input_test , target_test = next(iter(test_dataloader))
            input_test , target_test = input_test.to(device) , target_test.to(device)

            if sinLSTM and not stateLSTM:
                model_out = model(input_test, sine_target = target_test[:,:,-2:])
            
            elif stateLSTM and not sinLSTM:
                model_out = model(input_test, sine_target = target_test[:,:,-5:])
            
            elif sinLSTM and stateLSTM:
                model_out = model(input_test, sine_target = target_test[:,:,-7:])

            else:    
                model_out = model(input_test) #getting the output of the model
            
            if sines and not state:
                #computing the loss and removing the sine features, since they are known
                loss_test = loss_fn(torch.squeeze(model_out[:, : -future , :-2 ]), torch.squeeze(target_test[ : , : -future , :-2 ]))
                test_loss_pred6 = loss_fn(torch.squeeze(model_out[ : , -future : , :-2 ]), torch.squeeze(target_test[ :, -future : , :-2 ]))
                test_loss_pred1 = loss_fn(torch.squeeze(model_out[:, -future : -future + 1, :-2]), torch.squeeze(target_test[:, -future : -future + 1 , :-2]))
                test_loss_pred3 = loss_fn(torch.squeeze(model_out[:, -future : -future + 3, :-2]), torch.squeeze(target_test[:, -future : -future + 3, :-2]))
                test_loss_pont_pred6 = loss_fn(torch.squeeze(model_out[ : , -1 , :-2 ]), torch.squeeze(target_test[ :, -1 , :-2 ]))
                test_loss_pont_pred3 = loss_fn(torch.squeeze(model_out[ : , -4 , :-2 ]), torch.squeeze(target_test[ :, -4, :-2 ]))

            elif sines and state:
                #computing the loss and removing the sine and state features, since they are known
                loss_test = loss_fn(torch.squeeze(model_out[:, : -future , :-7 ]), torch.squeeze(target_test[ : , : -future , :-7 ]))
                test_loss_pred6 = loss_fn(torch.squeeze(model_out[ : , -future : , :-7 ]), torch.squeeze(target_test[ :, -future : , :-7 ]))
                test_loss_pred1 = loss_fn(torch.squeeze(model_out[:, -future : -future + 1, :-7]), torch.squeeze(target_test[:, -future : -future + 1 , :-7]))
                test_loss_pred3 = loss_fn(torch.squeeze(model_out[:, -future : -future + 3, :-7]), torch.squeeze(target_test[:, -future : -future + 3, :-7]))
                test_loss_pont_pred6 = loss_fn(torch.squeeze(model_out[ : , -1 , :-7 ]), torch.squeeze(target_test[ :, -1 , :-7 ]))
                test_loss_pont_pred3 = loss_fn(torch.squeeze(model_out[ : , -4 , :-7 ]), torch.squeeze(target_test[ :, -4, :-7 ]))                
            
            elif not sines and state:
                #computing the loss and removing the state feature, since they are known
                loss_test = loss_fn(torch.squeeze(model_out[:, : -future , :-5 ]), torch.squeeze(target_test[ : , : -future , :-5 ]))
                test_loss_pred6 = loss_fn(torch.squeeze(model_out[ : , -future : , :-5 ]), torch.squeeze(target_test[ :, -future : , :-5 ]))
                test_loss_pred1 = loss_fn(torch.squeeze(model_out[:, -future : -future + 1, :-5]), torch.squeeze(target_test[:, -future : -future + 1 , :-5]))
                test_loss_pred3 = loss_fn(torch.squeeze(model_out[:, -future : -future + 3, :-5]), torch.squeeze(target_test[:, -future : -future + 3, :-5]))
                test_loss_pont_pred6 = loss_fn(torch.squeeze(model_out[ : , -1 , :-5 ]), torch.squeeze(target_test[ :, -1 , :-5 ]))
                test_loss_pont_pred3 = loss_fn(torch.squeeze(model_out[ : , -4 , :-5 ]), torch.squeeze(target_test[ :, -4, :-5 ]))

            else:
                #computing the loss 
                loss_test = loss_fn(torch.squeeze(model_out[:, : -future , :]), torch.squeeze(target_test[ : , : -future , : ]))
                test_loss_pred6 = loss_fn(torch.squeeze(model_out[ : , -future : , : ]), torch.squeeze(target_test[ :, -future : , : ]))
                test_loss_pred1 = loss_fn(torch.squeeze(model_out[:, -future : -future + 1, :]), torch.squeeze(target_test[:, -future : -future + 1 , :]))
                test_loss_pred3 = loss_fn(torch.squeeze(model_out[:, -future : -future + 3, :]), torch.squeeze(target_test[:, -future : -future + 3, :]))
                test_loss_pont_pred6 = loss_fn(torch.squeeze(model_out[ : , -1 , : ]), torch.squeeze(target_test[ :, -1 , : ]))
                test_loss_pont_pred3 = loss_fn(torch.squeeze(model_out[ : , -4 , : ]), torch.squeeze(target_test[ :, -4, : ]))

            test_loss_pred1_log.append(test_loss_pred1.item())
            test_loss_pred3_log.append(test_loss_pred3.item())
            test_loss_pred6_log.append(test_loss_pred6.item())
            test_loss_pont_pred6_log.append(test_loss_pont_pred6.item())
            test_loss_pont_pred3_log.append(test_loss_pont_pred3.item())
            test_loss_log.append(loss_test.item()) 
            train_loss_log.append(loss.item())

            #Early stop 
            if epoch%1 == 0: 
                if test_loss_pred6 <= saved_loss:
                    torch.save(model.state_dict(), save_name)
                    saved_loss = test_loss_pred6
                    saved_epoch = epoch
                    patience_count = 0

                else:
                    patience_count = patience_count + 1
                
                if patience_count >= patience:
                    early_stop = not early_stop

            epoch = epoch + 1

            if not early_stop:
            #model_saved.LSTM.flatten_parameters()
                model.load_state_dict(torch.load(save_name))


    return model, test_loss_log, test_loss_pred1_log, test_loss_pred3_log, test_loss_pont_pred3_log, test_loss_pred6_log, test_loss_pont_pred6_log, train_loss_log, saved_epoch
"""

def alt_training(model,
                train_dataloader,
                test_dataloader,
                forecast_loss: int,
                save_name,
                loss_fn = nn.MSELoss(),
                LSTM = False,
                future = 6,
                epochs = 5000*5,
                patience = 500,
                device = "cuda" if torch.cuda.is_available() else "cpu"
                ):
    epoch=0
    patience_count = 0
    early_stop = True
    saved_loss = float('inf')

    model.to(device)    
    optimizer = optim.AdamW(model.parameters())

    while epoch < epochs and early_stop:
        for batch, (input , target) in enumerate(train_dataloader) : 
            input , target = input.to(device) , target.to(device)
            optimizer.zero_grad()
            if LSTM :
                model_out = model(input, sine_target = target[:,:,-2:])
            else :
                model_out = model(input)

            loss = loss_fn(torch.squeeze(model_out[ : , -future : -future + forecast_loss , :-2 ]),torch.squeeze(target[ : ,  : , :-2 ]))

            loss.backward()
            optimizer.step()
    with torch.no_grad():
            input_test , target_test = next(iter(test_dataloader))
            input_test , target_test = input_test.to(device) , target_test.to(device)
            


def testing(model,
            test_dataloader,
            LSTM_bool=False
            ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    future = 6
    MSE = nn.MSELoss()
    MAE = nn.L1Loss()

    with torch.no_grad():
        input_test , target_test = next(iter(test_dataloader))
        input_test , target_test = input_test.to(device) , target_test.to(device)
    
    if LSTM_bool:
        #model_out = model(input_test, pred_target = target_test[:,-future:, :], extra_features = 2, scheduled = 0)
        model_out = model(input_test, sine_target = target_test[:,:,-2:])  
    else:
        model_out = model_out = model(input_test)
    
    MSE_global = MSE(torch.squeeze(model_out[:, : -future , :-2 ]), torch.squeeze(target_test[ : , : -future , :-2 ]))
    MSE_pred6 = MSE(torch.squeeze(model_out[ : , -future : , :-2 ]), torch.squeeze(target_test[ :, -future : , :-2 ]))
    MSE_pred1 = MSE(torch.squeeze(model_out[:, -future : -future + 1, :-2]), torch.squeeze(target_test[:, -future : -future + 1 , :-2]))
    MSE_pred3 = MSE(torch.squeeze(model_out[:, -future : -future + 3, :-2]), torch.squeeze(target_test[:, -future : -future + 3, :-2]))
    MSE_pont_pred6 = MSE(torch.squeeze(model_out[ : , -1 , :-2 ]), torch.squeeze(target_test[ :, -1 , :-2 ]))
    MSE_pont_pred3 = MSE(torch.squeeze(model_out[ : , -4 , :-2 ]), torch.squeeze(target_test[ :, -4, :-2 ]))
    MAE_global = MAE(torch.squeeze(model_out[:, : -future , :-2 ]), torch.squeeze(target_test[ : , : -future , :-2 ]))
    MAE_pred6 = MAE(torch.squeeze(model_out[ : , -future : , :-2 ]), torch.squeeze(target_test[ :, -future : , :-2 ]))
    MAE_pred1 = MAE(torch.squeeze(model_out[:, -future : -future + 1, :-2]), torch.squeeze(target_test[:, -future : -future + 1 , :-2]))
    MAE_pred3 = MAE(torch.squeeze(model_out[:, -future : -future + 3, :-2]), torch.squeeze(target_test[:, -future : -future + 3, :-2]))
    MAE_pont_pred6 = MAE(torch.squeeze(model_out[ : , -1 , :-2 ]), torch.squeeze(target_test[ :, -1 , :-2 ]))
    MAE_pont_pred3 = MAE(torch.squeeze(model_out[ : , -4 , :-2 ]), torch.squeeze(target_test[ :, -4, :-2 ]))
    
    return [MSE_global.item(),#     0 
            MSE_pred6.item(),#      1
            MSE_pred1.item(),#      2
            MSE_pred3.item(),#      3
            MSE_pont_pred6.item(),# 4
            MSE_pont_pred3.item(),# 5
            MAE_global.item(),#     6 
            MAE_pred6.item(),#      7
            MAE_pred1.item(),#      8
            MAE_pred3.item(),#      9
            MAE_pont_pred6.item(),# 10
            MAE_pont_pred3.item()]# 11