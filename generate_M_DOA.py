#%%
import numpy as np
import torch
from pathlib import Path
import utils_function


#%%
#Load test data 
M={}
M_t={}
batch_size=300
nb_channels_test=1000
nb_channel_train=100
nb_BS_antenna=64
T=1
L=10
m=T*L

path_init=Path.cwd()
file_name = f'Data/Measurement_matrix/L_{L}_T_{T}'  

# generate Measurement matrix for test 
phases=np.random.uniform(0,2*np.pi,(nb_channels_test,nb_BS_antenna,m))
M_test= utils_function.generate_M(phases)

M_t['M_test']=M_test

#save data 
np.savez(path_init /file_name / 'test.npz', **M_t) 


i=0

while i<batch_size:

    # generate Measurement matrix for test 
    phases=np.random.uniform(0,2*np.pi,(nb_channel_train,nb_BS_antenna,m))
    M_train= utils_function.generate_M(phases)

    M['M_train']=M_train

    #save data 
    np.savez(path_init /file_name / f"batch_{i}.npz", **M) 

    i+=1

#%%
# generate DOA
DoA  =np.zeros((1200,3))
DoA[:,1]= np.random.uniform(-np.pi,np.pi,1200)

# save azimuth angles 
doa={}
doa['DoA']=DoA
np.savez(path_init / 'Data'/'DoA', **doa)





# %%
