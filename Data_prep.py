import argparse
import numpy as np 
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader

from torch.utils.data import ConcatDataset
from utils import Mortality_dataset

parser = argparse.ArgumentParser()
#optional arguments
parser.add_argument("-names", "--File_names", help = "Names of the data files to be saved, format (train_set, test_set)", nargs = "*", default = ["trainning_dataloader.dl","testing_dataloader.dl"], type = str)
parser.add_argument("-sv_dir", "--Save_dir", help = "directory path where the dataloaders are going to be saved", default = "/cfs/home/u021420/scripts/Preped_data")
parser.add_argument("-dt_dir", "--Data_dir", help = "directory path where the data is fetched ", default = "/cfs/home/u021420/Data")
parser.add_argument("-Pd", "--Padding", help = "If the data requires padding, used in SSSMs (default false)", action = "store_true")
parser.add_argument("-bs", "--Batch_size", help = "The size of each batch", type=int ,default = 64)
parser.add_argument("-f", "--Future", help = "The amount of time steps (months) we want to predict", type = int ,default = 6)
parser.add_argument("-sl", "--Sequence_len", help = "How many months of context we have", type = int ,default = 24)
parser.add_argument("-shf", "--Shuffle", help = "If we want to shuffle the dataloaders", action = "store_true")
parser.add_argument("-join", "--join", help = "if we want to join all the states into a single dataloader", action = "store_true")
parser.add_argument("-sine", "--sine", help = "introduces two sine features to the dataset to enable the model time perception", action = "store_true")
parser.add_argument("-state", "--state", help = "Introduce a one hot encoding feature to identify the state where the data is from", action = "store_true")
parser.add_argument("-teacher", "--teacher", help = "Makes the predictions available in the input for teacher forcing", action = "store_true")

#parser arguments
args = parser.parse_args()

#parameters
batch_size = args.Batch_size            
future = args.Future                    
seq_len = args.Sequence_len                  
shuffle_bool = args.Shuffle

#Setting the data directory
directory = args.Data_dir
os.chdir(directory)

year=['99','00_05','06_11','12_16','17_19']
State=['California','Texas','Florida','NewYork','Pennsylvania']

files={}

#Getting all the data from the different states in the same dictionary;
for S in range(len(State)):
    for y in range(len(year)):
        #preparing the file 
        data_interval=pd.read_csv('{state}_{years}_CoD.txt'.format(years=year[y],state=State[S]), sep='\t')
        data_interval=data_interval[~data_interval['Month'].isnull()]
        data_interval=data_interval.pivot(index=["Cause of death","Cause of death Code"],columns="Month Code", values="Deaths")

        if y==0:
            data=data_interval
        else:
            data=pd.merge(data, data_interval, how='inner', on= ["Cause of death","Cause of death Code"])
    data=data.dropna()

    #Normalizing the data according to the population
    #Getting the incidence per 100 000 inhabitants
    population = pd.read_csv('{state}_Population.txt'.format(state=State[S]), sep='\t')
    population = population[~population['Year'].isnull()]
    population = population['Population']

    for i in range(int(data.shape[1]/12)):
      data.iloc[ : , i*12 : ( i + 1 )*12] = (data.iloc[ : , i*12 : ( i + 1 )*12]/population.iloc[i])*100000

    #Collecting the common indexes to all the states
    if S==0:
        common_index=data.index
    else:
        common_index= np.intersect1d(common_index,data.index)
    files[State[S]]=data
    
#Keeping only the causes of death that are common to all the states 
for k in files:
    files[k]=files[k].loc[common_index]



#scalers={}
if args.arima:
    train_dataloaders = {}
    test_dataloaders = {}
    for k in files:
        
        death_array=files[k].values.transpose()
        #death_array of shape --> (total of months, number features)

        #75% for training, 25% for testing
        datasets = Mortality_dataset(death_array[:, : ], padding = args.Padding, future=future, seq_len=seq_len, sines=args.sine, state= args.state, teacher = args.teacher)
        dataloader = DataLoader(datasets,batch_size = batch_size, drop_last = False, shuffle = shuffle_bool, pin_memory_device="")

        for batch, (input , target) in enumerate(dataloader):
            if batch==0:
                train_array = input.numpy()
                test_array = target.numpy()[:,-future:,:]
            else:
                train_array = np.concatenate((train_array, input.numpy()), axis=0)
                test_array = np.concatenate((test_array, target.numpy()[:,-future:,:]), axis=0)
        #train_array.shape()--> (batch_size, sequence length 24, feature_size 57)
        #test_array.shape()--> (batch_size, future length 6, feature size 57)

        train_dataloaders[k] = train_array
        test_dataloaders[k] = test_array

    os.chdir(args.Save_dir)
    train_name, test_name = args.File_names
    np.save(train_name, train_dataloaders)
    np.save(test_name, test_dataloaders)    

elif args.join:
    train_datasets={}
    test_datasets ={}
    for S in range(len(State)):
        #scalers[k] = pre.MinMaxScaler()
        death_array = files[State[S]].values.transpose() #death_array of shape --> (total of months, number features)

        if args.sine:
        #creating two sine artificial features to give the model a sense of seasonality and adding them to the death array
            x = np.array(range(252))
            sines = np.append([np.sin(x*np.pi*2/12)],[np.sin(x*np.pi*2/6)], axis=0).transpose()
            death_array = np.append(death_array, sines, axis=1)

        if args.state:
        #creating a one hot encoding for the state where the example is from
            state_num = np.zeros((252,5))
            state_num[:,S] = np.ones((252,))
            death_array = np.append(death_array, state_num, axis=1) #cali->1 Texas->2 Florida->3 New York->4 Pensilvania->5

        #75% for training, 25% for testing
        train_datasets[State[S]] = Mortality_dataset(death_array[:189, : ], padding = args.Padding, future=future, seq_len=seq_len, sines=args.sine, state= args.state, teacher = args.teacher)
        test_datasets[State[S]] = Mortality_dataset(death_array[189 : , : ], padding = args.Padding, future=future, seq_len=seq_len, sines=args.sine, state= args.state)
    train_dataset = ConcatDataset([train_datasets[i] for i in train_datasets.keys() ])
    test_dataset = ConcatDataset([test_datasets[i] for i in test_datasets.keys() ])

    train_dataloaders = DataLoader(train_dataset,batch_size = batch_size, drop_last = False, shuffle = shuffle_bool, pin_memory_device="")
    test_dataloaders = DataLoader(test_dataset,batch_size = len(test_dataset)+1, drop_last = False, shuffle = shuffle_bool, pin_memory_device="")

    os.chdir(args.Save_dir)
    train_name, test_name = args.File_names
    torch.save(train_dataloaders, train_name)
    torch.save(test_dataloaders, test_name)    

else:
    #making the dataloaders
    train_dataloaders={}
    test_dataloaders={}
    #Creating the dataloaders for each state
    for k in files:
        #scalers[k] = pre.MinMaxScaler()

        death_array=files[k].values.transpose()
        #death_array of shape --> (total of months, number features)

        if args.sine:
        #creating two sine artificial features to give the model a sense of seasonality
            x = np.array(range(252))
            sines = np.append([np.sin(x*np.pi*2/12)],[np.sin(x*np.pi*2/6)], axis=0).transpose()
            death_array = np.append(death_array, sines, axis=1)

        if args.state:
        #creating a one hot encoding for the state where the example is from
            state_num = np.zeros((252,5))
            state_num[:,S] = np.ones((252,))
            death_array = np.append(death_array, state_num, axis=1) #cali->1 Texas->2 Florida->3 New York->4 Pensilvania->5

        #75% for training, 25% for testing
        train_dataset = Mortality_dataset(death_array[:189, : ], padding = args.Padding, future=future, seq_len=seq_len, sines=args.sine)
        test_dataset = Mortality_dataset(death_array[189 : , : ], padding = args.Padding, future=future, seq_len=seq_len, sines=args.sine)

        train_dataloaders[k] = DataLoader(train_dataset,batch_size = batch_size, drop_last = False, shuffle = shuffle_bool, pin_memory_device="")
        test_dataloaders[k] = DataLoader(test_dataset,batch_size = batch_size, drop_last = False, shuffle = shuffle_bool, pin_memory_device="")

        os.chdir(args.Save_dir)
        train_name, test_name = args.File_names

        torch.save(train_dataloaders, train_name)
        torch.save(test_dataloaders, test_name)
        