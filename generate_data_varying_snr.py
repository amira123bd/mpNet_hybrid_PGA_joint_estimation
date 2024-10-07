#%%
import torch
from pathlib import Path
import numpy as np 
import generate_channels_realizations
import utils_function
from tqdm import tqdm
import tensorflow as tf



import sys
from pathlib import Path

#%%
batch_size=500
nb_channels_train=100
nb_channels_test= 1000
batch_train=400



#%%
#get sionna data and preprocess it 
path_init=Path.cwd()
data_path='Data/datasionna'
norm_factor=utils_function.preprocess(data_path,batch_size)

#%% 
# get data normalize it and stack it in a single Dataset
file_name=f"batch_0.npz"
data = np.load(path_init/'Data/datasionna'/file_name)
channel_0 = data['channel_train']/norm_factor

Dataset=channel_0

i=1
while i<batch_size:
    #get channels

    file_name=f"batch_{i}.npz"
    data = np.load(path_init/'Data/datasionna'/file_name)
    channel_train= data['channel_train']/norm_factor
    Dataset=np.vstack((Dataset,channel_train))

    i+=1
 


#%%
sigma=1e-3
N_BS_antenna=Dataset.shape[1]


#%%
SNR_linear=np.linalg.norm(Dataset,2,axis=1)**2/ (N_BS_antenna * sigma)
 
#%%
SNR_dB= 10*np.log10(SNR_linear)
print(SNR_dB.mean())



#%%
#save test data
test_data={}

#Save data
channel_test=Dataset[:nb_channels_test]

#generate noisy channels,  the channel norm and sigma  
h_noisy,sigma_2=generate_channels_realizations.generate_noisy_channels_varying_SNR(channel_test,sigma)


test_data['h']          =torch.tensor(channel_test,dtype=torch.complex128)
test_data['h_noisy']    =h_noisy
test_data['sigma_2']    =sigma_2


file_name = f"test_data.npz"  #Save data per batch
np.savez(path_init / 'Data/1e-3/data_var_snr' / file_name, **test_data) 



# %%
batch=0
stop=True

while batch<batch_train and stop==True:
    #get channels
    train_data={}
    
    file_name=f"batch_{batch}.npz" 

    channel_train = Dataset[(batch*nb_channels_train)+nb_channels_test:((batch+1)*nb_channels_train)+nb_channels_test]


    #If number of channels in the batch is not enough 
    if channel_train.shape[0]!=nb_channels_train:
        stop=False
    else:

        # generate noisy channels
        h_noisy,sigma_2=generate_channels_realizations.generate_noisy_channels_varying_SNR(channel_train,sigma)


        train_data['h']             =torch.tensor(channel_train,dtype=torch.complex128)
        train_data['h_noisy']       =h_noisy
        train_data['sigma_2']       =sigma_2
       


        #Save data
        np.savez(path_init / 'Data/1e-3/data_var_snr' / file_name, **train_data)
        batch=batch+1
    
 





#%%
'''# remove data with negative SNR values
index_to_remove=[]
for i in range(SNR_dB.shape[0]):
    if(SNR_dB[i]<0):
        print(SNR_dB[i])
        index_to_remove.append(i)
        

#new_dataset_ = np.delete(Dataset,index_to_remove,axis=0) 
#shuffle data 
#new_dataset = new_dataset_[np.random.permutation(new_dataset_.shape[0])]

#new_dataset = Dataset[np.random.permutation(Dataset.shape[0])]
'''
