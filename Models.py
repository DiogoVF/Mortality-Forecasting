import sys
sys.path.append("/cfs/home/u021420/Env_SSMs/data_env/scripts")

import pandas as pd
import numpy as np 
import torch
from s4 import S4
from gconv_standalone import GConv
from torch import nn
from utils import Mortality_dataset
import random
from LS4_src.models.sequence.ss.s4 import S4 as S4L

class Liquid_S4(nn.Module):
  def __init__( self,
                d_model,
                d_input = None,
                d_state = 64,
                d_output = None,
                liquid_kernel = "kb", #the other option is "polyb"
                liquid_degree = 2,
                n_layers = 1):
      
    super().__init__()

    self.InputOutput = (d_input, d_output)

    if d_input is not None:
      self.encoder = nn.Linear(d_input,d_model)

    # Stack layers as residual blocks
    self.layers = nn.ModuleList()

    for _ in range(n_layers):
        self.layers.append(
            S4L(d_model, d_state = d_state, transposed = False, liquid_kernel = liquid_kernel, liquid_degree = liquid_degree ) #Transposed = False => (B, L, H)
        )
    if d_output is not None:
      self.decoder = nn.Linear(d_model,d_output)


  def forward(self, x):
    #Input x is shape (B, L, d_input)
    if self.InputOutput[0] is not None:
      xi = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
      z = xi

    else:
      z=x

    for layer in self.layers:
        # Each iteration of this loop will map (B, L, d_model) -> (B, L, d_model)
        # Apply S4 block: we ignore the state input and output
        z, _ = layer(z)
            
    # Decode the output
    if self.InputOutput[1] is not None:
      xout = self.decoder(z)  # (B, d_model) -> (B, d_output)
      return xout
        
    else:
      return z





#LSTM class, this was redesign to accept teacher forcing and scheduled sampling
class LSTM_predictor_cell(nn.Module):
  def __init__(self,n_features, hidden_size, n_layers):
    super(LSTM_predictor_cell, self).__init__()
    self.n_features = n_features
    self.hidden_size = hidden_size
    self.n_layers = n_layers

    self.LSTM = nn.LSTM(input_size = n_features,
                        hidden_size = hidden_size,
                        num_layers = n_layers, 
                        batch_first=True)

    self.fc = nn.Linear(hidden_size,n_features) #<-- for a fully connected layer
    
  def forward(self, input, pred_target=None,scheduled=0, extra_features=None,future=6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # input --> shape = (batch_size, sequence_length, feature_size) 
    # pred_target -->  shape = (batch_size,future, feature_size) these contains the real values of the predicitons;
    # extra_features : the number of extra features present nrgative number   
    # scheduled: the probability of using teacher forcing.

    h_0 = torch.zeros(self.n_layers, input.shape[0], self.n_features).to(device) #hidden state
    c_0 = torch.zeros(self.n_layers, input.shape[0], self.n_features).to(device) #internal state


    if random.random() < scheduled:
      output,(h_t,c_t) = self.LSTM(torch.cat((input,pred_target[:,:,:]), dim=1))
      # output --> shape = (batch_size, sequence_length+future, proj_size ou hidden_size)
      # h_t --> shape = (num_layers, batch_size, proj_size ou hidden_size)
      # c_t --> shape = (num_layers, batch_size, proj_size ou hidden_size) 

      out_fc = self.fc(output) 
      #out_fc --> shape = (batch_size, sequence_length+future, feature_size)
      return out_fc

    else:
      output,(h_t,c_t) = self.LSTM(input)
      # output --> shape = (batch_size, sequence_length, hidden_size)
      out_fc = self.fc(output) 
      #out_fc --> shape = (batch_size, sequence_length, feature_size)

      predictions = torch.empty((input.shape[0] ,future + input.shape[1], self.n_features)).to(device)
      # predictions --> shape = (batch_size, sequence_length + future, number_features), [h_0; h_1;...;h_N]
      predictions[: , : input.shape[1] , : ] = out_fc

      # making the predictions  
      for input_t in range(future):
          if extra_features is not None:
            output ,(h_t,c_t) = self.LSTM(torch.cat((out_fc[:,-1:,0:extra_features], pred_target[:, input_t:input_t+1, extra_features: ]),dim=-1),(h_t,c_t))
          else:
            output ,(h_t,c_t) = self.LSTM(out_fc[:,-1:,0:extra_features],(h_t,c_t))             
          out_fc = self.fc(output) #shape out_fc --> (batch_size,1,n_features)
          predictions[ : ,input.shape[1] + input_t , : ] =  torch.squeeze(out_fc)
      
    
      return predictions

class LSTM_predictor(nn.Module):
  def __init__(self,n_features, hidden_size, n_layers):
    super(LSTM_predictor, self).__init__()
    self.n_features = n_features
    self.hidden_size = hidden_size
    self.n_layers = n_layers

    self.LSTM = nn.LSTM(input_size = n_features,
                        hidden_size = hidden_size,
                        num_layers = n_layers, 
                        batch_first=True)

    self.fc = nn.Linear(hidden_size,n_features) #<-- for a fully connected layer
    
  def forward(self, input, sine_target=None,teacher=False, future=6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # input --> shape = (batch_size, sequence_length, feature_size) 
    # -sine_target: the true values of the extra features (sines and state) to be used in the prediction phase, 
    #     instead of the the predicted sines --> shape = (batch_size,seq_len , 2) or (batch_size,seq_len , 1),
    #     if only state feature or (batch_size,seq_len , 7) if state and sin  
    # -teacher: defines if teacher forcing is used, basically bypasses the predictions, since the input now
    #     contains the predicitions as well.

    #out_fc = torch.empty((future + 1, self.n_features)).to(device)
    h_0 = torch.zeros(self.n_layers, input.shape[0], self.n_features).to(device) #hidden state
    c_0 = torch.zeros(self.n_layers, input.shape[0], self.n_features).to(device) #internal state

    output,(h_t,c_t) = self.LSTM(input)
    # output --> shape = (batch_size, sequence_length, proj_size ou hidden_size)
    # h_t --> shape = (num_layers, batch_size, proj_size ou hidden_size)
    # c_t --> shape = (num_layers, batch_size, proj_size ou hidden_size) 

    out_fc = self.fc(output) 
    #out_fc --> shape = (batch_size, sequence_length, feature_size)

    if teacher:
       return out_fc

    if not teacher:
      predictions = torch.empty((input.shape[0] ,future + input.shape[1], self.n_features)).to(device)
      # predictions --> shape = (batch_size, sequence_length + future, number_features), [h_0; h_1;...;h_N]
      predictions[: , : input.shape[1] , : ] = out_fc

      # making the predictions 
      if sine_target is not None:   
        for input_t in range(future):

          if sine_target is not None:
            #output ,(h_t,c_t) = self.LSTM(torch.cat((out_fc[:,:,0:-sine_target.shape[2]], sine_target[:, input_t : input.shape[1] + input_t, : ]), axis=2),(h_t,c_t) )
            output ,(h_t,c_t) = self.LSTM(torch.cat((out_fc[:,-1:,0:-sine_target.shape[2]], sine_target[:, input.shape[1]+input_t : input.shape[1] + input_t+1, : ]), axis=2),(h_t,c_t) )
            out_fc = self.fc(output) 
            predictions[ : ,input.shape[1] + input_t , : ] =  torch.cat((out_fc[:,-1,:-sine_target.shape[2]], sine_target[:, input.shape[1] + input_t  , :]), axis=1)
          else:
             #output ,(h_t,c_t) = self.LSTM(torch.cat((out_fc[:,:,0:-sine_target.shape[2]], sine_target[:, input_t : input.shape[1] + input_t, : ]), axis=2),(h_t,c_t) )
            output ,(h_t,c_t) = self.LSTM(out_fc[:,-1:, :], (h_t,c_t) )
            out_fc = self.fc(output) 
            predictions[ : ,input.shape[1] + input_t , : ] =  out_fc[:,-1,:]
      else:
        for input_t in range(future):
          #output --> shape = (batch_size, sequence_length, hidden_size)
          #hidden state e cell state: h_t --> shape = (num_layers, batch_size, proj_size ou hidden_size)
          #                           c_t --> shape = (num_layers, batch_size, proj_size ou hidden_size) 
          output ,(h_t,c_t) = self.LSTM(out_fc,(h_t,c_t))
          out_fc = self.fc(output) 
          predictions[ : ,input.shape[1] + input_t , : ] =  out_fc[:,-1,:]
    
      return predictions

#Simplified state space model or S5 model
"""class S5_Model(nn.Module):
#The S5 implementation does not support batching, needs to be done by hand
    def __init__(
        self,
        d_model,
        d_state,
        d_input = None,
        d_output = None,
        n_layers = 1
        ):

        self.InputOutput = (d_input, d_output)
        blocks = 8 #default value in the S5 paper, it is the number of block diganonals used to aproximate HiPPO
        block_size = int(d_state/blocks)
        # Initialize state matrix A using approximation to HiPPO-LegS matrix
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
        #Cutting the approximation to the blocksize
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T
        #Make a hiPPO apporxiamtion in each block
        Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
        V = block_diag(*([V] * blocks))
        Vinv = block_diag(*([Vc] * blocks))
        

        self.ssm_init_fn = S5_init( H = d_model,
                                    P = d_state,
                                    Lambda_re_init=Lambda.real,
                                    Lambda_im_init=Lambda.imag,
                                    V=V,
                                    Vinv=Vinv,
                                    C_init="trunc_standard_normal", #default initialization of C
                                    discretization="zoh", #descretization method
                                    dt_min=0.001, #default value
                                    dt_max=0.1, #default value
                                    conj_sym= True,
                                    clip_eigs= False,
                                    bidirectional=False)
        super().__init__()
        if d_input is not None:
          self.encoder = nn.Linear(d_input,d_model)

        # Stack S5 layers as residual blocks
        layer = S5_layer(self.ssm_init_fn, 0, d_model=d_model)
        
        self.s4_layers = [S5_layer(self.ssm_init_fn, 0, d_model=d_model)
                          for _ in range(n_layers)]
        
        if d_output is not None:
          self.decoder = nn.Linear(d_model,d_output)

    def forward(self, x):
        x_out = torch.ones(x.shape)
#       Input x is shape (B, L, d_input)

        for batch in range(x.shape[0]):
          x_batch = x[batch,:,:] # (L, d_input)

          if self.InputOutput[0] is not None:
            z_batch = self.encoder(x_batch)  # (L, d_input) -> (L, d_model)
          
          else:
            z_batch = x_batch # (L, d_input) -> (L, d_model)

          for layer in self.s4_layers:
              # Each iteration of this loop will map (L, d_model) -> (L, d_model)
              # Apply S5 block: we ignore the state input and output
              z_batch, _ = layer(z_batch)

              
          # Decode the output
          if self.InputOutput[1] is not None:
            x_out[batch,:,:]= self.decoder(z_batch)  # (L, d_model) -> (L, d_output)
          
          else:
             x_out[batch,:,:] = z_batch
        
        return x_out"""
            
        
#S4 class, this class has the options of accepting specific outpu and input sizes, if we wish to reduce 
#the number of features, one can also adjust the size of each SSM
class S4_Model(nn.Module):

    def __init__(
        self,
        d_model,
        d_input = None,
        d_state = 64,
        d_output = None,
        n_layers = 1
        ):

        self.InputOutput = (d_input, d_output)

        super().__init__()
        if d_input is not None:
          self.encoder = nn.Linear(d_input,d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.s4_layers.append(
                S4(d_model, d_state = d_state, transposed = False) #Transposed = False => (B, L, H)
            )
        if d_output is not None:
          self.decoder = nn.Linear(d_model,d_output)

    def forward(self, x):

#        Input x is shape (B, L, d_input)

        if self.InputOutput[0] is not None:
          xi = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
          z = xi

        else:
          z=x

        for layer in self.s4_layers:
            # Each iteration of this loop will map (B, L, d_model) -> (B, L, d_model)
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            
        # Decode the output
        if self.InputOutput[1] is not None:
          xout = self.decoder(z)  # (B, d_model) -> (B, d_output)
          return xout
        
        else:
          return z
        
#DSS class, the difference between this and the S4 is the arguments of the s4 function
class DSS_Model(nn.Module):

    def __init__(
        self,
        d_model,
        d_input = None,
        d_state = 64,
        d_output = None,
        n_layers = 1
        ):

        self.InputOutput = (d_input, d_output)
        super().__init__()
        if d_input is not None:
          self.encoder = nn.Linear(d_input,d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.s4_layers.append(
                S4(
                  d_model,
                  d_state = d_state,
                  mode = "diag",
                  measure = "diag-legs",
                  bidirectional = False,
                  disc = "dss",
                  real_type = "none",  
                  transposed = False) #Transposed = False => (B, L, H)
            )
        if d_output is not None:
          self.decoder = nn.Linear(d_model,d_output)

    def forward(self, x):

#        Input x is shape (B, L, d_input)

        if self.InputOutput[0] is not None:
          xi = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
          z = xi

        else:
          z=x

        for layer in self.s4_layers:
            # Each iteration of this loop will map (B, L, d_model) -> (B, L, d_model)
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            
        # Decode the output
        if self.InputOutput[1] is not None:
          xout = self.decoder(z)  # (B, d_model) -> (B, d_output)
          return xout
        
        else:
          return z

#S4d class, the difference between this and the S4 is the arguments of the s4 function
class S4D_Model(nn.Module):

    def __init__(
        self,
        d_model,
        d_input = None,
        d_state = 64,
        d_output = None,
        n_layers = 1
        ):

        self.InputOutput = (d_input, d_output)
        super().__init__()
        if d_input is not None:
          self.encoder = nn.Linear(d_input,d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.s4_layers.append(
                S4(
                  d_model,
                  d_state = d_state,
                  mode = "diag",
                  measure = "diag-lin",
                  bidirectional = False,
                  disc = "zoh",
                  real_type = "exp",  
                  transposed = False) #Transposed = False => (B, L, H)
            )
        if d_output is not None:
          self.decoder = nn.Linear(d_model,d_output)

    def forward(self, x):

#        Input x is shape (B, L, d_input)

        if self.InputOutput[0] is not None:
          xi = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
          z = xi

        else:
          z=x

        for layer in self.s4_layers:
            # Each iteration of this loop will map (B, L, d_model) -> (B, L, d_model)
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            
        # Decode the output
        if self.InputOutput[1] is not None:
          xout = self.decoder(z)  # (B, d_model) -> (B, d_output)
          return xout
        
        else:
          return z

class Gconv_Model(nn.Module):
   
  def __init__(
        self,
        d_model,
        d_input = None,
        d_state = 64,     #d_state é na verdade dimensão do kernel
        d_output = None,
        n_layers = 1,
        seq_len = 30
        ):
      
    super().__init__()

    self.InputOutput = (d_input, d_output)

    if d_input is not None:
        self.encoder = nn.Linear(d_input,d_model)

      # Stack gconv layers as residual blocks
    self.gconv_layers = nn.ModuleList()

    for _ in range(n_layers):
          self.gconv_layers.append(
              GConv(
                d_model = d_model,
                d_state = -2, #isto não serve para nada
                kernel_dim = d_state,
                l_max = seq_len,  
                transposed = False) #Transposed = False => (B, L, H)
          )

    if d_output is not None:
        self.decoder = nn.Linear(d_model,d_output)
      
  def forward(self, x):
  #Input x is shape (B, L, d_input)

    if self.InputOutput[0] is not None:
      xi = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
      z = xi

    else:
      z=x

    for layer in self.gconv_layers:
        # Each iteration of this loop will map (B, L, d_model) -> (B, L, d_model)
        # Apply Gconv block: we ignore the state input and output
        z, _ = layer(z)

    # Decode the output
    if self.InputOutput[1] is not None:
      xout = self.decoder(z)  # (B, d_model) -> (B, d_output)
      return xout

    else:
      return z     
    
