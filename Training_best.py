from Models import (LSTM_predictor,
                    S4D_Model,
                    S4_Model,
                    Gconv_Model,
                    Liquid_S4,
                    DSS_Model)

from utils import (Training_Testing,
                   Mortality_dataset,
                   testing)
from itertools import product
import argparse
import numpy as np
import time
import torch
import os


parser2 = argparse.ArgumentParser(prog='ProgramName')
parser2.add_argument("-svname",
                    help = "The name of the results file",
                    default = "You_forgot_to_name_this"
                    )
parser2.add_argument("-models",
                    help = "Receives a list of the models to evaluate. Options: S4, S4D, LS4_kb, LS4_polyb, LSTM, SGconv. Default: all models",
                    nargs = '*',
                    default = ["S4", "S4D","DSS" ,"SGconv","LSTM","LS4_kb", "LS4_polyb"]
                    )
parser2.add_argument("-N", 
                    help = "Number of times we train the model for the same combination of hyperparameters, default = 20",
                    default = 20,
                    type = int)
parser2.add_argument("-avg", 
                    help = "If we want the average of the N runs or all the runs",
                    action = "store_true")
args = parser2.parse_args()

Models = args.models
n_features = 59
Model_dict={}

for model_key in Models:

    MSE6_iter = np.ones(args.N) #guarda o MSE6 de cada iteração para ver qual a iteração com o melhor MSE6
    Results = np.ones((args.N,33,5,30,59)) #(iteration,batches,states,features,sequence_length)
    Input = np.ones((args.N,33,5,30,59)) #same shape but this are the input values
    

    for i in range(args.N):

        print("training: " + model_key + " N: ",i)

        if model_key == "LSTM":
            train_data = torch.load("Preped_data/train_dataloader_JSSin.dl") 
            test_data = torch.load("Preped_data/test_dataloader_JSSin.dl")
            real_test_data = torch.load("Preped_data/test_dataloader_SSin.dl")

        else:
            train_data = torch.load("Preped_data/train_dataloader_JSPSin.dl") 
            test_data = torch.load("Preped_data/test_dataloader_JSPSin.dl")
            real_test_data = torch.load("Preped_data/test_dataloader_SPSin.dl")
            
        if model_key=="S4":
            Model_dict["model"] = S4_Model(
                        d_model= 118,
                        d_input=n_features, 
                        d_state = 32, 
                        d_output=n_features, 
                        n_layers= 1)
            Model_dict["name"] = model_key
                
        elif model_key=="S4D":       
            Model_dict["model"] = S4D_Model(
                        d_model= 118,
                        d_input=n_features, 
                        d_state = 16, 
                        d_output=n_features, 
                        n_layers= 1) 
            Model_dict["name"] = model_key
                    
        elif model_key == "DSS":
                Model_dict["model"] = DSS_Model(d_model= 118,
                                                d_input=n_features, 
                                                d_state = 32, 
                                                d_output=n_features, 
                                                n_layers= 1)          
                Model_dict["name"] = model_key

        elif model_key=="SGconv":
            Model_dict["model"] = Gconv_Model(
                        d_model= 118,
                        d_input=n_features, 
                        d_state = 16, 
                        d_output=n_features, 
                        n_layers= 1)
            Model_dict["name"] = model_key
            
        elif model_key=="LSTM":
            Model_dict["model"] = LSTM_predictor(
                n_features = n_features,
                hidden_size = 64,
                n_layers = 1)
            Model_dict["name"] = model_key
                
        elif model_key=="LS4_kb":
            Model_dict["model"] = Liquid_S4(
                d_model=118,
                d_input=n_features,
                d_output=n_features,
                d_state=16,
                n_layers=1,
                liquid_kernel="kb",
                liquid_degree=4)
            Model_dict["name"] = model_key
                
        elif model_key=="LS4_polyb":
            Model_dict["model"] = Liquid_S4(
                d_model=118,
                d_input=n_features,
                d_output=n_features,
                d_state= 128,
                n_layers=1,
                liquid_kernel="polyb",
                liquid_degree=3)
            Model_dict["name"] = model_key

        model,model_out,_,_,_,_,MSE6,_,_,saved_epoch = Training_Testing(model_dict = Model_dict,
                                    train_dataloader = train_data,
                                    test_dataloader = test_data, 
                                    extra_features = 2,
                                    save_name = "Training_best_"+ model_key + "_i"+ str(i) +".pt", 
                                    teacher_ratio = 0.75 if model_key=="LSTM" else 0,
                                    epochs = 5000*5,
                                    raw=True)
        
        MSE6_iter[i] = MSE6[saved_epoch]

        if i == args.N-1:
            best_iter = np.argmin(MSE6_iter)

        with torch.no_grad():
            s=0
            for state in real_test_data:
                input, target = next(iter(real_test_data[state]))
                input, target = input.to("cuda"), target.to("cuda") 

                if model_key == "LSTM":
                    output = model(input, sine_target = target[:,:,-2:])

                else:
                    output = model(input)

                output = output.to("cpu")
                input = input.to("cpu")
                target = target.to("cpu")
                Results[i,:,s,:,:] = output[:,:,:]
                Input[i,:,s,:,:]= target[:,:,:]
                s=s+1

print(best_iter)
np.save(args.svname,(Results,Input))
    

    