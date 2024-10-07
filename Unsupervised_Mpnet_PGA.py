
#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import ProjGradAscent
import plots
import utils_function            
import generate_steering
torch.manual_seed(42)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'


# %%
N = 4   # Num of users
L = 10 # RF chains
T = 1  # Mesures
M = 64   # Tx antennas
sigma=1e-3
train_size = 5000
valid_size = 1000
max_batch  = 300
epochs = 10
batch_size = 10  # batch size

 
#%%
#Get data
path_init=Path.cwd()
file_name_dataset = f'Data/1e-3/data_var_snr'
file_name_mpnet = f'Data/1e-3/Estimation/L_10_T_1/mpnet'
#file_name_nominal = f'Data/1e-3/Estimation/nominal'  
#file_name_real = f'Data/1e-3/Estimation/real'  
#file_name_ls=  f'Data/1e-3/Estimation/ls'  

batch_idx=0

stop=True

H_Mpnet= torch.empty(0,N,M,dtype=torch.complex128)
H_nominal=torch.empty(0,N,M,dtype=torch.complex128)
H_true= torch.empty(0,N,M,dtype=torch.complex128)
H_noisy= torch.empty(0,N,M,dtype=torch.complex128)
H_ls= torch.empty(0,N,M,dtype=torch.complex128)
H_real= torch.empty(0,N,M,dtype=torch.complex128)


#%%
#####

# h: true channels without noise
# h_noisy: noisy channels 


#####

# load train data 
while stop:
    # Load true channels
    true_data = np.load(path_init / file_name_dataset/ f'batch_{batch_idx}.npz')
    h = torch.tensor(true_data['h'], dtype=torch.complex128)
    H_true = torch.cat((H_true, h.view(-1, N, M)), dim=0)

    # Load noisy channels
    h_noisy = torch.tensor(true_data['h_noisy'], dtype=torch.complex128)
    H_noisy = torch.cat((H_noisy, h_noisy.view(-1, N, M)), dim=0)
    h_norm  = torch.norm(h_noisy,p=2,dim=1)

    # Load Mpnet Estimated channels
    Mpnet_data = np.load(path_init / file_name_mpnet / f'batch_{batch_idx}.npz')
    Mpnet_channels = torch.tensor(Mpnet_data['channels'], dtype=torch.complex128)
    Mpnet_channels = Mpnet_channels * h_norm[:, None]
    H_Mpnet = torch.cat((H_Mpnet, Mpnet_channels.view(-1, N, M)), dim=0)
    
    

    ''' 
    # Load dict real Estimated channels 
    real_data = np.load(path_init / file_name_real / f'batch_{batch_idx}.npz')
    channels_real= torch.tensor(real_data['channels'], dtype=torch.complex128)
    channels_real = channels_real * h_norm[:, None]
    H_real = torch.cat((H_real, channels_real.view(-1, N, M)), dim=0)

    
   # Load dict nominal Estimated channels 
    nominal_data = np.load(path_init / file_name_nominal / f'batch_{batch_idx}.npz')
    channels_nominal = torch.tensor(nominal_data['channels'], dtype=torch.complex128)
    channels_nominal = channels_nominal * h_norm[:, None]
    H_nominal = torch.cat((H_nominal, channels_nominal.view(-1, N, M)), dim=0)

    
    # Load ls Estimated channels 
    ls_data = np.load(path_init / file_name_ls / f'batch_{batch_idx}.npz')
    channels_ls= torch.tensor(ls_data['channels'], dtype=torch.complex128)
    channels_ls = channels_ls * channel_norm[:, None]
    H_ls = torch.cat((H_ls, channels_ls.view(-1, N, M)), dim=0)      
    ''' 


    if H_true.shape[0] == valid_size + train_size :
        stop = False
    batch_idx += 1

#%%
# load test data and denormalize channels
data = np.load(path_init / file_name_dataset/ f'test_data.npz')
data_test_mpnet = np.load(path_init / file_name_mpnet / f'test.npz')
#data_test_real=  np.load(path_init / file_name_real / f'test.npz')
#data_test_nominal = np.load(path_init / file_name_nominal/ f'test.npz')

h_mpnet = torch.tensor(data_test_mpnet['channels'], dtype=torch.complex128)
#h_real= torch.tensor(data_test_real['channels'], dtype=torch.complex128)
#h_nominal = torch.tensor(data_test_nominal['channels'], dtype=torch.complex128)
h_noisy_test=torch.tensor(data['h_noisy'],dtype=torch.complex128)
h_test=torch.tensor(data['h'],dtype=torch.complex128)

h_norm_test= torch.norm(h_noisy_test,p=2,dim=1)

# denormalize estimations
h_mpnet   = h_mpnet        * h_norm_test[:, None]
#h_real     = h_real       * h_norm_test[:, None]
#h_nominal  = h_nominal    * h_norm_test[:, None]




#%%
# Get train, test and validation data 

#H_true, H_mpnet, H_nominal

# train data 
H_true_train    = H_true [0:train_size].to(device)
H_noisy_train   = H_noisy[0:train_size].to(device)
H_mpnet_train   = H_Mpnet [0:train_size].to(device)
#H_nominal_train = H_nominal [0:train_size].to(device)
#H_real_train    = H_real[0:train_size].to(device)
#H_ls_train=H_ls[0:train_size].to(device)

# validation data 
H_true_val      = H_true [train_size:train_size+int(valid_size/N)].to(device)
H_noisy_val     = H_noisy[train_size:train_size+int(valid_size/N)].to(device)
H_mpnet_val     = H_Mpnet [train_size:train_size+int(valid_size/N)].to(device)
#H_nominal_val   = H_nominal [train_size:train_size+int(valid_size/N)].to(device)
#H_real_val      = H_real[train_size:train_size+int(valid_size/N)].to(device)
#H_ls_val=H_ls[train_size+test_size:].to(device)

#%%
# test data 
H_true_test     = h_test.view(-1,N,M).to(device)
H_noisy_test    = h_noisy_test.view(-1,N,M).to(device)
H_mpnet_test    = h_mpnet.view(-1,N,M).to(device)
#H_nominal_test  = h_nominal.view(-1,N,M).to(device)
#H_real_test     = h_real.view(-1,N,M).to(device)
#H_ls_test=H_ls[train_size:train_size+test_size].to(device)



#%% 
# Define loss function 

def sum_loss(wa, wd, h, n, batch_size, sigma):
    a1 = torch.transpose(wa, 1, 2).conj() @ torch.transpose(h, 1, 2).conj()
    a2 = torch.transpose(wd, 1, 2).conj() @ a1
    a3 = h @ wa @ wd @ a2
    g = torch.eye(n,device=device).reshape((1, n, n)) + a3 / (n * sigma)  # g = Ik + H*Wa*Wd*Wd^(H)*Wa^(H)*H^(H)
    s = torch.log(g.det())  # s = log(det(g))

  
    loss = sum(torch.abs(s)) / batch_size
    return -loss

#%%

# Define evaluation function 
def evaluate(h, WA, WD, n, sigma, batch_size, num_iter):
    sum_rate = torch.zeros(num_iter)
    for i in range(num_iter):
        sum_rate[i] = sum_loss(WA[i], WD[i], h, n, batch_size, sigma)
       
        
    return -sum_rate



#%%
def Classical_PGA(hyp,num_of_iter_pga,h,test_size):
    mu = torch.tensor([[hyp] * (2)] * num_of_iter_pga, requires_grad=True)


    #Object defining
    classical_model = (ProjGradAscent.ProjGA(mu))

    # executing classical PGA 
    s,wa,wd,WA,WD = classical_model.forward(h,N,L,  num_of_iter_pga,sigma)

    # evaluate classical PGA 
    sum_rate= evaluate(H_true_test, WA,WD,N,sigma,test_size,num_of_iter_pga)
    

    return sum_rate


#%% ploting the results
def plot_sum_rate(sum_rate,num_of_iter_pga,pga_type,channel_type):

    y = sum_rate.cpu().detach().numpy()
    x = np.array(list(range(num_of_iter_pga))) + 1

    plt.figure()
    plt.plot(x, y, '+--')
    plt.title(f'Sum rate, {pga_type}, {channel_type} M={M} N={N} L={L} T={T}')
    plt.xlabel('Number of Iteration')
    plt.ylabel('Achievable Rate')
    plt.grid()

    plt.show()


def save_sum_rate(file_name,sum_rate_):
    
    # save sum rate
    sum_rate={}
    sum_rate[f'{file_name}']=sum_rate_.detach().numpy()
    np.savez(path_init / 'sumRate'/'1e-3'/f'{file_name}', **sum_rate)

#%%
# -------------------------------- Classical PGA --------------------------------------------------
classical_iter=50

#%%
#----------TRUE CHANNELS ------------
sum_rate_classical_true_chan=Classical_PGA(1.5e-2,classical_iter,H_true_test,H_true_test.shape[0])
plot_sum_rate(sum_rate_classical_true_chan,classical_iter,'Classical_PGA','Perfect CSI')

#%%
#----------MPNET ESTIMATED MPNET CHANNELS ------------
sum_rate_classical_Mpnet_chan=Classical_PGA(1e-2,classical_iter,H_mpnet_test,H_mpnet_test.shape[0])
plot_sum_rate(sum_rate_classical_Mpnet_chan,classical_iter,'Classical_PGA','MpNet Estimation')

#%%
#----------MP NOMINAL ESTIMATED CHANNELS ------------
#sum_rate_classical_Mp_nominal_chan=Classical_PGA(3e-4,classical_iter,H_nominal_test,H_nominal_test.shape[0])
#plot_sum_rate(sum_rate_classical_Mp_nominal_chan,classical_iter,'Classical_PGA','Mp Nominal Estimation')

#%%
#----------MP REAL ESTIMATED CHANNELS ------------
#sum_rate_classical_Mp_real_chan=Classical_PGA(4e-4,classical_iter,H_real_test,H_real_test.shape[0])
#plot_sum_rate(sum_rate_classical_Mp_real_chan,classical_iter,'Classical_PGA','Mp real Estimation')

#%%
#----------LS CHANNELS ------------------
#sum_rate_classical_ls_chan=Classical_PGA(6e-4,classical_iter,H_ls_test,test_size)
#plot_sum_rate(sum_rate_classical_ls_chan,classical_iter,'Classical_PGA','LS Estimation')

#%% save all estimations 
save_sum_rate('class_true_channel',sum_rate_classical_true_chan)
save_sum_rate('class_est_mpnet_channel',sum_rate_classical_Mpnet_chan)
#save_sum_rate('class_est_mp_nominal_channel',sum_rate_classical_Mp_nominal_chan)
#save_sum_rate('class_est_mp_real_channel',sum_rate_classical_Mp_real_chan)
#save_sum_rate('class_est_ls_channel',sum_rate_classical_ls_chan)


#%%
# -----------------------------------Unfolded PGA ------------------------------------------
# parameters defining
num_of_iter_pga_unf = 10 
mu_unf = torch.tensor([[1e-2] * (2)] * num_of_iter_pga_unf, requires_grad=True)
#mu_unf= torch.randn(num_of_iter_pga_unf,2, requires_grad=True)*1e-3


#%% Object defining
unfolded_model = ProjGradAscent.ProjGA(mu_unf)

# %% optimizer

optimizer = torch.optim.Adam(unfolded_model.parameters(), lr=1e-3)
scheduler= torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.2)



#%%
epochs = 30
batch_size = 500 # batch size
train_losses, valid_losses = [], []

for i in tqdm(range(epochs)):


    permut=np.random.permutation(H_true_train.shape[0])
    

    H_shuffeld_Mpnet      = H_mpnet_train  [permut]
    H_shuffeld_true       = H_true_train   [permut]
    #H_shuffeld_Mp_nominal = H_nominal_train[permut]
    #H_shuffeld_Mp_real    = H_real_train[permut]
    #H_shuffeld_ls   = H_ls_train[permut]

    for b in range(0, H_true_train.shape[0], batch_size):
        H_Mpnet =   H_shuffeld_Mpnet         [b:b+batch_size].to(device)
        H_true  =   H_shuffeld_true          [b:b+batch_size].to(device)
        #H_Mp_nominal = H_shuffeld_Mp_nominal [b:b+batch_size].to(device)
        #H_Mp_real = H_shuffeld_Mp_real  [b:b+batch_size].to(device)
        #H_ls = H_shuffeld_ls  [b:b+batch_size].to(device)
       
        sum_train, wa, wd , _,_ = unfolded_model.forward(H_Mpnet, N, L,  num_of_iter_pga_unf,sigma)

        loss = sum_loss(wa, wd, H_Mpnet, N,  batch_size,sigma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       

    with torch.no_grad():
        # train loss
        sum_rate, wa, wd,_,_ = unfolded_model.forward(H_mpnet_train, N, L,  num_of_iter_pga_unf,sigma)
        train_losses.append(sum_loss(wa, wd, H_mpnet_train, N,  H_mpnet_train.shape[0],sigma))
       
        #y_unf = [r.detach().numpy() for r in sum(sum_rate)/sum_rate.shape[0]]
        #print(y_unf)
        # validation loss
        __, wa, wd,_,_ = unfolded_model.forward(H_mpnet_val, N, L,  num_of_iter_pga_unf,sigma)
        valid_losses.append(sum_loss(wa, wd, H_mpnet_val, N,  H_mpnet_val.shape[0],sigma))


#%% 
#plotting learning curve
plots.plot_learning_curve(train_losses,valid_losses,num_of_iter_pga_unf,epochs,batch_size,train_size,1000,valid_size,L)

#%% 
# executing unfolded PGA on the test set
s,_,_,WA,WD = unfolded_model.forward(H_true_test, N, L,  num_of_iter_pga_unf,sigma)


#%%
# Evaluate results 
sum_rate= evaluate(H_true_test, WA,WD,N,sigma,H_true_test.shape[0],num_of_iter_pga_unf)

#%%
plot_sum_rate(sum_rate,10,'Unfolded PGA','MpNet Estimation')

#%%
# save sum rate

#unf_est_mpnet_channel
#unf_true_channel
#unf_est_mp_nominal_channel
#unf_est_mp_real_channel
#unf_est_ls
save_sum_rate('unf_true_channel',sum_rate)

#%%
torch.save(unfolded_model,f'pretrained_models/pga_sigma_1e-3_L_10_T_1.pth')
#torch.save(unfolded_model.state_dict(), 'pretrained_model/pga_sigma_1e-3_L_10_T_1.pth')





