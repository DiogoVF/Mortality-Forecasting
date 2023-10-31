from Models import (LSTM_predictor,
                    S4D_Model,
                    S4_Model,
                    DSS_Model,
                    Gconv_Model,
                    Liquid_S4)

from utils import (Training_Testing,
                   Mortality_dataset,
                   testing)

from itertools import product
import argparse
import numpy as np
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
                    default = ["S4", "S4D", "DSS","SGconv","LSTM","LS4_kb", "LS4_polyb"]
                    )
args = parser2.parse_args()

#This code tests the presence os state features and its effects on the models performance 
Models = args.models
Results = {}
N=10
Features = { "SP": None, "SPSin": 2,"SPSinState": 7, "SPState": 5}
Model_dict={}

for fea in Features:
    Results[fea] = {}

    for model_key in Models:
        os.chdir("/cfs/home/u021420/scripts/Preped_data/")

        if model_key == "LSTM":
            train_data = torch.load("train_dataloader_J" + fea[0] + fea[2:] + ".dl") 
            test_data = torch.load("test_dataloader_J" + fea[0] + fea[2:] + ".dl")

        else:
            train_data = torch.load("train_dataloader_J" + fea + ".dl") 
            test_data = torch.load("test_dataloader_J" + fea + ".dl")
        
        n_features = next(iter(train_data))[0].shape[2]

        Losses = np.ones((8,N))
        for i in range(N):
            
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
                Model_dict["model"] = DSS_Model(d_model= 188,
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
                    liquid_degree= 4)
                Model_dict["name"] = model_key
                
            elif model_key=="LS4_polyb":
                Model_dict["model"] = Liquid_S4(
                    d_model=118,
                    d_input=n_features,
                    d_output=n_features,
                    d_state= 128,
                    n_layers=1,
                    liquid_kernel="polyb",
                    liquid_degree= 3)
                Model_dict["name"] = model_key
            
            print("feature: ", fea," Model: " + model_key + " N= ", i+1, "/", N)
            model , test_log, loss_pred1, loss_pred3, loss_pont_pred3, loss_pred6, loss_pont_pred6, train_loss, saved_epoch = Training_Testing(model_dict = Model_dict,
                                    train_dataloader = train_data,
                                    test_dataloader = test_data, 
                                    extra_features = Features[fea],
                                    teacher_ratio=0.75 if model_key=="LSTM" else 0,
                                    save_name = "Fea_var"+ model_key + "_" + fea + ".pt", 
                                    epochs = 5000*5)

            Losses[ : , i] = [test_log[saved_epoch], 
                                loss_pred1[saved_epoch], 
                                loss_pred3[saved_epoch],
                                loss_pred6[saved_epoch], 
                                loss_pont_pred3[saved_epoch],
                                loss_pont_pred6[saved_epoch],
                                train_loss[saved_epoch],
                                saved_epoch]
            print(Losses[:,i])

        Results[fea][model_key] = Losses
np.save(args.svname,Results)

    