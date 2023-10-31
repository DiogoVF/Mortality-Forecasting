# Thesis-Code-Base
This repository contains the datasets and source code used in my M.Sc. research project regarding the forecast of mortality rates.

## Data Preparation
The raw Data is present in the Data folder, this data results from querying directly the U.S. underlying cause of death database (https://wonder.cdc.gov/ucd-icd10.html). The Data_prep.py script is responsible for transforming this data into the training and test sets for the models. The results are in teh Preped_data folder.
Example of usage:
```python
python Data_prep.py -names ["trainning_dataloader.dl","testing_dataloader.dl"] -join -shf -sine -Pd
#this command create two dataloaders with batch size 64, with data from all the sates (-join), with padding for the SSMs (-Pd),
#using the sinusoidal features (-sine) and the batches are shufled (-shf)
```
## Hyper-parameter adjustment
Having the appropriate data loaders we can run the HP_adjust.py script to fine tune the hyper-parameters. This script receives the model to be used along with their respective hyper-parameters. The script saves the results as a dictionary whose key is (var1, var2, var3, var4), each var takes a value of their respective hyper-parameter, when the hyper-parameter is not used the value of the respective var is None. Each element of the dictionary is a list containing the test losses, the overall test loss (over the 30 months context+forecast), the 1, 2, and 3 month test losses both punctual and up until that month and the training loss.
Example of usage:
```python
python HP_adjust.py S4D -var1 30 59 118 -var2 16 32 64 128 -var3 1 2 3 -sines -flname HP_adjust_S4D.pt
#this is adjusting the hyper-parameter of the S4D model, where var1 => number of features, var2 => SSM dimention and var3 => number of layers
#-sines indicates we are using the sinusoidal features
```
## Training best
To perform model comparison we first need to train the best models on the optimal hyper-parameters. The script Training_best.py does just that, it trains the given model with the optimal hyper-parameters (these are hardcoded) 20 times, saving the parameters of each iteration. It prints the number of best iteration based on the MSE of the 6 month forecast. The results are saved into a matrix of shape (iterations,batches,states,sequence_length,features), usually (20,33,5,30,59) that correspond to the models ouputs on the test set.
Example of usage:
```python
python Training_best.py -models S4D -N 20 -svname Best_of_S4D
#this trains 20 iteration of the S4D model and saves it under the name Best_of_S4D
```
## Feature Variation
Again with the optimal hyper-parameters hardcoded, the Feature_Variations.py script runs every combination of extra features, without both the sinusoidal and state features, with only the state features, with only the sinusoidal features and with no extra features. It receives the intended model and returns a dictionary with keys "SP" (no extra features), "SPSin" (only the sinusoidal features), "SPSinState" (both state and sinusoidal features), "SPState"(only the state features), where each element of the dictionary is a list containing the test losses, the overall test loss (over the 30 months context+forecast), the 1, 2, and 3 month test losses both punctual and up until that month and the training loss.
Example of usage:
```python
python Feature_Variation.py -models S4D -svname FeaVar_S4D
#This performs the feature variation on the S4D model
```
