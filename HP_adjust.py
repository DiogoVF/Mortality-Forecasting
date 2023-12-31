import argparse
from utils import (Training_Testing,
                   testing,
                   Mortality_dataset)

from itertools import product
import random
import numpy as np
import time
import torch
import os

parser2 = argparse.ArgumentParser(prog='ProgramName')
parser2.add_argument("model", 
                    help = "Select the model to be trained and tested, the hyper parameter values are already defined. Options : S4, S4D, GConv, LSTM",
                    choices = ["S4", "S4D", "GConv", "LSTM", "LS4","DSS"],
                    default = "GConv")
parser2.add_argument("-var1",
                    help = "Receives a list of values that correspond to one hyperparameter of the model. S4, S4D, Gconv => var1 = n_Features | LSTM => var1 = hidden_size",
                    nargs = '*',
                    default = [None],
                    type=int)
parser2.add_argument("-var2",
                    help = "Receives a list of values that correspond to one hyperparameter of the model. S4, S4D, Gconv => var2 = d_SSM | LSTM => var2 = teacher_ratio in %",
                    nargs = '*',
                    default = [None],
                    type=int)
parser2.add_argument("-var3",
                    help = "Receives a list of values that correspond to one hyperparameter of the model. S4, S4D, Gconv => var3 = n_layers | LSTM => var3 = n_layers ",
                    nargs = '*',
                    default = [None],
                    type=int)
parser2.add_argument("-var4",
                    help = "Receives a list of values that correspond to one hyperparameter of the model. LS4=>ld ",
                    nargs = '*',
                    default = [None],
                    type=int)
parser2.add_argument("-e", "--epochs",
                    help = "Receives the number of epochs for training, default=5000",
                    default = 5000*5, 
                    type = int)
parser2.add_argument("-N", 
                    help = "Number of times we train the model for the same combination of hyperparameters, default = 10",
                    default = 10,
                    type = int)
parser2.add_argument("-Seed", 
                    help = "Seed to generate random numbers, used to make comparisons between Hyperparameters. Default no seed",
                    default = None,
                    type = int)
parser2.add_argument("-flname", 
                    help = "Name and directory of the file where the results will be stored. Default = Results",
                    default = "Results")
parser2.add_argument("-sines", 
                    help = "If the features contain the sines",
                    action = "store_true")
parser2.add_argument("-state", 
                    help = "If the features contain the state one hot encoding",
                    action = "store_true")
parser2.add_argument("-Lk","--Liquid_kernel", 
                    help = "The type of liquid kernel to be used",
                    choices = ["kb","polyb"],
                    default = "kb")

args = parser2.parse_args() 
SSMs = ["S4", "S4D", "GConv", "LS4","DSS"]
Others = ["LSTM"]
States = ['California','Texas','Florida','NewYork','Pennsylvania']

if args.Seed is not None:
    torch.manual_seed(args.Seed)
    torch.cuda.manual_seed_all(args.Seed)
    np.random.seed(args.Seed)

start_time = time.time()
#Importing the appropriate data according to the model selected
if args.model in SSMs:
    if  not args.sines and not args.state:
        train_data = torch.load("Preped_data/train_dataloader_JSP.dl") 
        test_data = torch.load("Preped_data/test_dataloader_JSP.dl")
        best_test_data = torch.load("Preped_data/test_dataloader_SP.dl") 
        extra_features = None

    elif args.sines and not args.state:
        train_data = torch.load("Preped_data/train_dataloader_JSPSin.dl") 
        test_data = torch.load("Preped_data/test_dataloader_JSPSin.dl")
        best_test_data = torch.load("Preped_data/test_dataloader_SPSin.dl") 
        extra_features = 2

    elif args.sines and args.state:
        train_data = torch.load("Preped_data/train_dataloader_JSPSinState.dl") 
        test_data = torch.load("Preped_data/test_dataloader_JSPSinState.dl")
        best_test_data = torch.load("Preped_data/test_dataloader_SPSinState.dl") 
        extra_features = 7
    
    elif not args.sines and args.state:
        train_data = torch.load("Preped_data/train_dataloader_JSPState.dl") 
        test_data = torch.load("Preped_data/test_dataloader_JSPState.dl")
        best_test_data = torch.load("Preped_data/test_dataloader_SPState.dl") 
        extra_features = 5

if args.model in Others:
    if not args.sines and not args.state:
        train_data = torch.load("Preped_data/train_dataloader_JS.dl") 
        test_data = torch.load("Preped_data/test_dataloader_JS.dl")
        best_test_data = torch.load("Preped_data/test_dataloader_JS.dl")
        extra_features = None

    elif args.sines and not args.state:
        train_data = torch.load("Preped_data/train_dataloader_JSSin.dl") 
        test_data = torch.load("Preped_data/test_dataloader_JSSin.dl")
        best_test_data = torch.load("Preped_data/test_dataloader_SSin.dl")
        extra_features = 2

    elif args.sines and args.state:
        train_data = torch.load("Preped_data/train_dataloader_JSSinState.dl") 
        test_data = torch.load("Preped_data/test_dataloader_JSSinState.dl")
        best_test_data = torch.load("Preped_data/test_dataloader_SSinState.dl")
        extra_features = 7
    
    elif not args.sines and args.state:
        train_data = torch.load("Preped_data/train_dataloader_JSState.dl") 
        test_data = torch.load("Preped_data/test_dataloader_JSState.dl")
        best_test_data = torch.load("Preped_data/test_dataloader_SState.dl")
        extra_features = 5

import_time = time.time()

n_features = next(iter(train_data))[0].shape[2]

os.chdir("/cfs/home/u021420/scripts/Results/")
num_iter = len(args.var1)*len(args.var2)*len(args.var3)*len(args.var4)*args.N
Results = {}
j=0
model_dict={}
for var1, var2, var3, var4 in product(args.var1, args.var2, args.var3, args.var4):
    Losses = np.ones((8,args.N)) #shape -> ( losses, repetitions i)G
    for i in range(args.N):
        if args.model == "S4":
            from Models import S4_Model
            model_dict["model"] = S4_Model( d_model= var1,
                                            d_input=n_features, 
                                            d_state = var2, 
                                            d_output=n_features, 
                                            n_layers= var3)        
            model_dict["name"] = args.model

        
        if args.model == "DSS":
            from Models import DSS_Model
            model_dict["model"] = DSS_Model(d_model= var1,
                                            d_input=n_features, 
                                            d_state = var2, 
                                            d_output=n_features, 
                                            n_layers= var3)          
            model_dict["name"] = args.model    
        
        if args.model == "S4D":
            from Models import S4D_Model
            model_dict["model"] = S4D_Model(d_model= var1,
                                            d_input=n_features, 
                                            d_state = var2, 
                                            d_output=n_features, 
                                            n_layers= var3)          
            model_dict["name"] = args.model    

        if args.model == "GConv":
            from Models import Gconv_Model
            model_dict["model"] = Gconv_Model(  d_model= var1,
                                                d_input=n_features, 
                                                d_state = var2, 
                                                d_output=n_features, 
                                                n_layers= var3)           
            model_dict["name"] = args.model
        
        if args.model == "LS4":
            from Models import Liquid_S4
            model_dict["model"] = Liquid_S4(    d_model=var1,
                                                d_input=n_features,
                                                d_output=n_features,
                                                d_state=var2,
                                                n_layers=var3,
                                                liquid_kernel=args.Liquid_kernel,
                                                liquid_degree=var4).to("cuda")      
            model_dict["name"] = args.model

        if args.model == "LSTM":
            from Models import LSTM_predictor
            model_dict["model"] = LSTM_predictor(   n_features = n_features,
                                                    hidden_size = var1,
                                                    n_layers = var3)  
            model_dict["name"] = args.model


        print( j+1 , "/", num_iter)
        j=j+1

        model , test_log, loss_pred1, loss_pred3, loss_pont_pred3, loss_pred6, loss_pont_pred6, train_loss, saved_epoch = Training_Testing(
                                                                                                    model_dict = model_dict ,
                                                                                                    train_dataloader = train_data,
                                                                                                    test_dataloader = test_data,
                                                                                                    extra_features = extra_features,
                                                                                                    save_name = "discard_" + args.model + ".pt" if args.model != "LS4"  else "discard_" + args.model + args.Liquid_kernel + ".pt" , 
                                                                                                    epochs = args.epochs,
                                                                                                    teacher_ratio = var2/100 if args.model == "LSTM" else 0)
            
        Losses[ : , i] = [test_log[saved_epoch], 
                           loss_pred1[saved_epoch], 
                           loss_pred3[saved_epoch],
                           loss_pred6[saved_epoch], 
                           loss_pont_pred3[saved_epoch],
                           loss_pont_pred6[saved_epoch],
                           train_loss[saved_epoch],
                           saved_epoch]
        print(Losses[ : , i])

    # training mode to yield the MAE and MSE organised by state
    if args.training_best:
        model_key = args.model
        for state in best_test_data:
                if i == 0:
                    Results[model_key][state] = np.ones((12,args.N))
                
                Results[model_key][state][:,i] = testing(model,
                                                    best_test_data[state],
                                                    LSTM_bool = model_key=="LSTM")
                if i == args.N-1:
                    Results[model_key][state] =np.mean(Results[model_key][state],axis=1)

    else:
        Results[(var1, var2, var3, var4)] = Losses

training_time = time.time()

np.save(args.flname, Results)

saving_time = time.time()

print("import time : ", import_time - start_time )
print("training time : ", training_time - import_time)
print("saving time : ", saving_time - training_time)
print("total time : ", saving_time - start_time)