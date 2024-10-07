#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import ProjGradAscent
import mpnet_model
import End_to_End_model
import utils_function
import matplotlib as mpl  
#import plots
import time
import generate_steering
mpl.rcParams.update(mpl.rcParamsDefault)


torch.manual_seed(42)


# %% Parameters defining
k=5 # iteration mpnet sc
num_of_iter_pga_unf=10 # iteration pga
N     = 4        # Num of users
L     = 10       # RF chains
T     = 1        # Instant
M     = 64       # BS antennas
m     = T*L      # mesures
noise_var = 1e-3  # noise variance
 
train_size = 2000
test_size  = 1000
val_size = 1000
nb_channels_train=100 # number of channels 
batch_size = 100
epochs=100
max_batch=300



path_init=Path.cwd()

#%%
# load test data 

file_name_dataset = 'Data/1e-3/data_var_snr'  
test_data = np.load(path_init/file_name_dataset/'test_data.npz')
                
h_test        =    torch.tensor(test_data['h'],dtype=torch.complex128)       
h_noisy_test  =    torch.tensor(test_data['h_noisy'],dtype=torch.complex128)      
sigma_2_test  =    torch.tensor(test_data['sigma_2'])     
norm_test     =    torch.norm(h_noisy_test,p=2,dim=1)

## preprocessing 

# normalize channels 
h_noisy_test   = h_noisy_test / norm_test[:,None]


# Get Measurement matrix
M_test = torch.tensor(np.load(path_init/'Data'/f'Measurement_matrix/L_{L}_T_{T}'/'test.npz')['M_test'],dtype=torch.complex128)

# Multiply channels by M 
h_noisy_test_M = torch.matmul(h_noisy_test.unsqueeze(1), torch.conj(M_test)).squeeze()
channel_norm_M = torch.norm(h_noisy_test_M, p=2, dim=1)
h_noisy_test_M = h_noisy_test_M / channel_norm_M[:, None]

# Mpnet Estimation
sigma_norm_test = torch.sqrt(sigma_2_test) / channel_norm_M

# reshape channels for pga 
H_test = h_test.view(-1,N,M)


#%%
# train channels 
h_noisy_train_M= torch.empty(0,dtype=torch.complex128)
norm_train= torch.empty(0)
norm_train_M= torch.empty(0)
h= torch.empty(0,dtype=torch.complex128)
h_noisy = torch.empty(0,dtype=torch.complex128)
M_train=torch.empty(0,dtype=torch.complex128)
batch_idx=0
while batch_idx < max_batch:

    train_data = np.load(path_init/file_name_dataset/f'batch_{batch_idx}.npz')
   
    # true channels 
    h_chan = torch.tensor(train_data['h'], dtype=torch.complex128)

    # noisy channels
    h_noisy_chan = torch.tensor(train_data['h_noisy'], dtype=torch.complex128)
    h_norm  = torch.norm(h_noisy_chan,p=2,dim=1)
    h_noisy_chan=h_noisy_chan/h_norm[:,None]

    # generate Measurement matrix and multiply channels 
    #phases=np.random.uniform(0,2*np.pi,(h_noisy_chan.shape[0],h_noisy_chan.shape[1],m))
    #m_train = torch.tensor(train_data['M_train'], dtype=torch.complex128)
    #m_train= utils_function.generate_M(phases)

    m_train = torch.tensor(np.load(path_init/'Data'/f'Measurement_matrix/L_{L}_T_{T}'/f'batch_{batch_idx}.npz')['M_train'],dtype=torch.complex128)


    h_noisy_M  =torch.matmul((h_noisy_chan).unsqueeze(1),torch.conj(m_train)).squeeze()
    norm_M     =    torch.norm(h_noisy_M,p=2,dim=1)
    h_noisy_M  = h_noisy_M/ norm_M[:,None]

    h_noisy_train_M = torch.cat((h_noisy_train_M, h_noisy_M), dim=0)
    norm_train= torch.cat((norm_train, h_norm), dim=0)
    h= torch.cat((h, h_chan), dim=0)
    h_noisy= torch.cat((h_noisy, h_noisy_chan), dim=0)
    M_train= torch.cat((M_train, m_train), dim=0)
    norm_train_M = torch.cat((norm_train_M, norm_M), dim=0)

    batch_idx+=1

##############################Define the model structure #####################################
#%%
#Mpnet Init [init_ nominal, real or pretrained]

file_name = r'Data2/datasionna/antenna_position.npz'  
antenna_pos = np.load(path_init/file_name)
nominal_ant_positions=antenna_pos['nominal_position']
real_ant_positions=antenna_pos['real_position']
DoA= np.load(path_init/'Data2/DoA.npz')['DoA']
g_vec= np.ones(64)
lambda_ =  0.010706874

# load pretrained model 

#MPNET
constrained_mpnet=torch.load('pretrained_models/mpnet_L_10_T_1_noise_1e-3.pth',map_location='cpu')
for _,param in constrained_mpnet.named_parameters():
    antenna_position= param.data
#PGA
pga=torch.load('pretrained_models/pga_sigma_1e-3_L_10_T_1.pth')


#%%
Mpnet_pga=End_to_End_model.CompositeModel(pga , constrained_mpnet,antenna_position,DoA,g_vec,lambda_,'Mpnet')
 

#%%
# Define loss function 
def sum_loss(wa, wd, h, n, batch_size, sigma):
    a1 = torch.transpose(wa, 1, 2).conj() @ torch.transpose(h, 1, 2).conj()
    a2 = torch.transpose(wd, 1, 2).conj() @ a1
    a3 = h @ wa @ wd @ a2
    g = torch.eye(n).reshape((1, n, n)) + (a3 / (n * sigma))  # g = Ik + H*Wa*Wd*Wd^(H)*Wa^(H)*H^(H)
    s = torch.log(g.det())  # s = log(det(g))

    loss = sum(torch.abs(s)) / batch_size
    return -loss

#%%
device='cpu'
# Define evaluation function 
def evaluate(h, WA, WD, n, sigma, batch_size, num_iter):
    sum_rate = torch.zeros(num_iter)
    for i in range(num_iter):
        sum_rate[i] = sum_loss(WA[i], WD[i], h, n, batch_size, sigma)
       
        
    return -sum_rate

#%% ploting the results
def plot_sum_rate(sum_rate,num_of_iter_pga,pga_type,channel_type):

    y = sum_rate.detach().numpy()
    x = np.array(list(range(num_of_iter_pga))) + 1

    
    plt.figure()
    plt.plot(x, y, '+--')
    plt.title(f'Sum rate, {pga_type}, {channel_type} M={M} N={N} L={L} T={T}')
    plt.xlabel('Number of Iteration')
    plt.ylabel('Achievable Rate')
    plt.grid()
    plt.show()



#%%
#optimizer = torch.optim.Adam(Mpnet_pga.parameters(), lr=1e-5)

# custom optimizer
optimizer=torch.optim.Adam([
     {'params': Mpnet_pga.mpnet.parameters(),'lr':1e-4},
     {'params': Mpnet_pga.pga.parameters()  ,'lr':1e-4}

])

#Mpnet_pga.freeze_parameters(Mpnet_pga.pga)

#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)





#%%
# model training 

epochs= 1
batch_size=200
train_size=1000
train_losses, valid_losses = [], []
 




for i in tqdm(range(epochs)):

    
    

    for b in range(0, train_size, batch_size):
 
        print(f'batch: {b}')


        #------Get Data-------

        h_batch                    = h[b:b+batch_size]  # true channels not normalized
        h_noisy_batch              = h_noisy[b:b+batch_size] # noisy channels normalizd
        h_noisy_train_M_batch      = h_noisy_train_M[b:b+batch_size]
        M_train_batch              = M_train[b:b+batch_size]
        norm_train_batch           = norm_train[b:b+batch_size]
        norm_train_M_batch         = norm_train_M[b:b+batch_size]

        sigma_2_train_batch        = torch.tensor([noise_var] * batch_size)

       

        sigma_norm_train = torch.sqrt(sigma_2_train_batch)/ norm_train_M_batch
 
        #H_train true channels

        H_train_init=h_batch.view(-1,N,M)
 
 
        #------ Forward Pass--------------  
       
        sum_rate , wa, wd,_,_ = Mpnet_pga.forward(h_noisy_train_M_batch,h_noisy_batch,h_batch,norm_train_batch,k,sigma_norm_train,M_train_batch,N,M,L,num_of_iter_pga_unf,noise_var)

        #------ Backward Pass-------------
        loss = sum_loss(wa, wd, H_train_init, N,  H_train_init.shape[0],noise_var) 
        #print(loss)


        optimizer.zero_grad()
       
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(Mpnet_pga.parameters(), 20)


        optimizer.step()


        #for i,j in Mpnet_pga.named_parameters():
            #print(i)
            #print(j)
       

        #for name, param in Mpnet_pga.named_parameters():
            #if param.grad is not None:
                #print(f"Grad for {name}: {param.grad.norm()}")
            #print(f"Param {name}: {param.norm()}")        

    #scheduler.step()

    with torch.no_grad():
        sum_rate_unf, wa, wd,WA,WD = Mpnet_pga.forward(h_noisy_test_M,h_noisy_test,h_test,norm_test,k,sigma_norm_test,M_test,N,M,L,num_of_iter_pga_unf,noise_var)
        #sum_rate=evaluate(H_test,WA,WD,N,noise_var,H_test.shape[0],10)
        #plot_sum_rate(sum_rate,10,'unfolded','true')
        # train loss
        #__, wa, wd,_,_ = Mpnet_pga.forward(h_noisy_train_M[:train_size],h_noisy[:train_size],h[:train_size],norm_train[:train_size],k,torch.sqrt(torch.tensor([noise_var] * train_size))/ norm_train_M[:train_size],M_train[:train_size],N,M,L,num_of_iter_pga_unf,noise_var)
        #train_losses.append(sum_loss(wa, wd, (h_noisy[:train_size]).view(-1,N,M), N, train_size),noise_var)
 
        # validation loss
        #__, wa, wd,_,_  = Mpnet_pga.forward(h_noisy_val_M,h_noisy_val,val_channels_clean,channel_norm_val,k,sigma_norm_val,M_val,N,M,L,num_of_iter_pga_unf,noise_var)
        #valid_losses.append(sum_loss(wa, wd, h_noisy_val.view(-1,N,M), N, val_size))
 
#end_time=time.time()




'''#%%
import plots
plots.plot_learning_curve(train_losses,
                        valid_losses,
                        num_of_iter_pga_unf,
                        epochs,
                        batch_size,
                        train_size,
                        test_size,
                        val_size,L)'''


#%%
sum_rate_unf, wa, wd,WA,WD = Mpnet_pga.forward(h_noisy_test_M,h_noisy_test,h_test,norm_test,k,sigma_norm_test,M_test,N,M,L,num_of_iter_pga_unf,noise_var)


#%%
sum_rate=evaluate(H_test,WA,WD,N,noise_var,H_test.shape[0],10)


# %%
plot_sum_rate(sum_rate,10,'unfolded','true')

#%%
def save_sum_rate(file_name,sum_rate_):
    # save sum rate
    sum_rate={}
    sum_rate[f'{file_name}']=sum_rate_.detach().numpy()
    np.savez(path_init / 'sumRate'/'1e-3'/f'{file_name}', **sum_rate)
#%%
save_sum_rate('End_To_End',sum_rate)

