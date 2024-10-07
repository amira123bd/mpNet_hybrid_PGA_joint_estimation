import matplotlib.pyplot as plt
from typing import Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
import generate_steering
import  mpnet_model
from torch import autograd
import sparse_recovery
import sys
import utils_function
from torch.nn.utils import parameters_to_vector
import matplotlib as mpl
from tqdm import trange
mpl.rcParams.update(mpl.rcParamsDefault)
#torch.manual_seed(42)
# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
 
class UnfoldingModel_Sim:
    def __init__(self,
                 BS_position: np.ndarray,
                 antenna_pos: np.ndarray,
                 real_antenna_pos: np.ndarray,
                 sigma_ant:float,
                 DoA: np.ndarray,
                 g_vec: np.ndarray,
                 lambda_,
                 snr_in_dB: int = 20,                  
                 lr: float = 0.001,
                 lr_constrained: float =0.1,
                 momentum: float = 0.9,                
                 optimizer = 'adam',
                 epochs: int = 10,
                 batch_size: int = 100,              
                 k: int = None,
                 batch_subsampling: int = 20,
                 train_type: str = 'online',
                 f0:int=28e9,
                 nb_channels_test: int=200,
                 nb_channels_train: int=1000,
                 nb_channels_val: int=200,
                 nb_channels_per_batch: int=20,
                 BS_antenna: int=64,
                 L: int=10,
                 T: int=10,
                 noise_var: int=1e-02
                 ) -> None:
       
       
        self.snr_in_dB=snr_in_dB
        self.lr=lr
        self.lr_constrained=lr_constrained
        self.momentum=momentum
        self.epochs=epochs
        self.k=k
        self.batch_subsampling=batch_subsampling
        self.train_type=train_type
        self.BS_position=BS_position
        self.antenna_pos=torch.tensor(antenna_pos).type(torch.FloatTensor).to(device)
        self.DoA=torch.tensor(DoA).type(torch.FloatTensor).to(device)
        self.g_vec=torch.tensor(g_vec).to(device)
        self.lambda_= lambda_
        self.train_type=train_type
        self.f0=f0
        self.batch_size=batch_size
        self.nb_channels_test=nb_channels_test
        self.nb_channels_train= nb_channels_train
        self.nb_channels_val=nb_channels_val
        self.optimizer=optimizer
        self.nb_channels_per_batch=nb_channels_per_batch
        self.sigma_ant=sigma_ant
        self.real_antenna_pos=real_antenna_pos
        self.m=L*T
        self.T=T
        self.L=L
        self.noise_var=noise_var
        self.BS_antenna = BS_antenna
 
 
        # to store the channel realizations
        self.dataset=None
     
        #sigma noise
        self.sigma_noise = None
       
        #stopping criteria
        self.SC2 = None
 
 
        #generate nominal steering vectors using nominal antenna positions
        dict_nominal=generate_steering.steering_vect_c(self.antenna_pos,
                                                       self.DoA,
                                                       self.g_vec,
                                                       self.lambda_).type(torch.complex128).to(device)
        #
        dict_real = generate_steering.steering_vect_c(torch.tensor(self.real_antenna_pos).type(torch.FloatTensor).to(device),
                                                       self.DoA,
                                                       self.g_vec,
                                                       self.lambda_).type(torch.complex128).to(device)
       
       
 
        self.dict_nominal = dict_nominal
        self.dict_real=dict_real
        weight_matrix=dict_nominal
       
        #initialization of the mpnet model with the weight matrix
        self.mpNet = mpnet_model.mpNet(weight_matrix)
        self.mpNet_Constrained= mpnet_model.mpNet_Constrained(self.antenna_pos,self.DoA,self.g_vec,self.lambda_, True).to(device)
       
        #Initialization of the optimizer
        self.optimizer= optim.Adam(self.mpNet.parameters(),lr=self.lr)
        self.constrained_optimizer = optim.Adam(self.mpNet_Constrained.parameters(), lr = self.lr_constrained)
 
       
        #Result table for every batch size over the whole epochs
        #cost function over train
        dim_results = (epochs*batch_size)
        self.cost_func = np.zeros(dim_results)
        self.cost_func_c=np.zeros(dim_results)
       
       
       
        dim_per_batch=int(np.ceil(batch_size/batch_subsampling))
 
        #Initialization of SNR out table
        self.snr_out_c_mpnet=np.zeros(dim_per_batch)
        self.snr_out_dB_mpNet=np.zeros(dim_per_batch)
        self.snr_out_dB_omp_real=np.zeros(dim_per_batch)
        self.snr_out_dB_omp_nominal=np.zeros(dim_per_batch)
       
        #cost function over test    
        self.cost_func_test = np.zeros(dim_per_batch)
        self.cost_func_test_c = np.zeros(dim_per_batch)
        self.cost_func_ls =np.zeros(dim_per_batch)
        self.cost_func_mp_nominal=np.zeros(dim_per_batch)
        self.cost_func_mp_real=np.zeros(dim_per_batch)
       
       
       
       
    def train_online_test_inference(self):

        if self.train_type=='Online':
 
            path_init=Path.cwd()
            file_name = 'Data/1e-3/data_var_snr/test_data.npz'  
            test_data = np.load(path_init/file_name)
                
            h_test        =    torch.tensor(test_data['h'],dtype=torch.complex128).to(device)         
            h_noisy_test  =    torch.tensor(test_data['h_noisy'],dtype=torch.complex128).to(device)       
            sigma_2_test  =    torch.tensor(test_data['sigma_2']).to(device)      
            norm_test     =    torch.norm(h_noisy_test,p=2,dim=1).to(device)

            print(h_test.shape)

            ## preprocessing 

            # normalize channels 
            h_noisy_test   = h_noisy_test / norm_test[:,None]
            h_test         = h_test / norm_test[:,None]


            # generate Measurement matrix and multiply channels 
            M_test = torch.tensor(np.load(path_init/'Data'/f'Measurement_matrix/L_{self.L}_T_{self.T}'/'test.npz')['M_test'],dtype=torch.complex128).to(device)

            
            h_noisy_test_M =torch.matmul((h_noisy_test).unsqueeze(1),torch.conj(M_test)).squeeze().to(device)
            norm_test_M     =torch.norm(h_noisy_test_M,p=2,dim=1).to(device)
            h_noisy_test_M  =h_noisy_test_M/ norm_test_M[:,None]


            #Stopping criteria
            sigma_norm_test = (torch.sqrt(sigma_2_test)/norm_test_M).to(device)
            #SC2= pow(sigma_norm_test,2) * self.L * self.BS_antenna

            SC2= pow((torch.sqrt(sigma_2_test)/norm_test),2) * self.BS_antenna

            #the first inference before training the channel to get estimation with the initial nominal dictionnary(no gradient is computed)
            idx_subs = 1
            for batch in range(self.batch_size):
    
                if batch == 0:
                
                    with torch.no_grad():  
                        #residuals_mp_test_c, est_chan_test, test_temp_test_c = self.mpNet(h_noisy_test_M,h_noisy_test,M_test,self.L, self.k,sigma_norm_test,2)
                        residuals_mp_test_c, est_chan_test_c, test_temp_test_c= self.mpNet_Constrained(h_noisy_test_M,h_noisy_test,M_test,self.L, self.k,sigma_norm_test,2)
                
                    #Estimate channel reconstruction
                    h_hat_mpNet_test_c = test_temp_test_c.detach().cpu().numpy()
   
                    #Cost function  
                    self.cost_func_test_c[0]=torch.mean(torch.sum(torch.abs(h_test.cpu()-h_hat_mpNet_test_c)**2,1)/torch.sum(torch.abs(h_test.cpu())**2,1))
    

                    #MP with real and nominal dictionnary
                    #rel_err_nominal, _= UnfoldingModel_Sim.run_mp_omp(self,h_test,h_noisy_test_M,h_noisy_test,M_test,SC2,'nominal')
                    #rel_err_real,_    = UnfoldingModel_Sim.run_mp_omp(self,h_test,h_noisy_test_M,h_noisy_test,M_test,SC2,'real')
                
    
                
                    #Add mp results to tables
                    #self.cost_func_mp_nominal[:] = rel_err_nominal * np.ones(len(self.cost_func_mp_nominal))
                    #self.snr_out_dB_omp_nominal[:] = mean_snr_out_dB_nominal * np.ones(len(self.snr_out_dB_omp_nominal))
                
                    #self.cost_func_mp_real[:] = rel_err_real * np.ones(len(self.cost_func_mp_real))
                    #self.snr_out_dB_omp_real[:] = mean_snr_out_dB_real * np.ones(len(self.snr_out_dB_omp_real))  
                
                    #LS estimation
                    #Mh=torch.conj(torch.transpose(M_test, 1, 2))
                    #ls_est=(torch.linalg.inv(M_test @ Mh) @ M_test @ h_noisy_test_M.unsqueeze(2)).squeeze(2).cpu().numpy()
                    #cost_func_ls=np.mean(np.linalg.norm(h_test.cpu().numpy()-ls_est,2,axis=1)**2/np.linalg.norm(h_test.cpu().numpy(),2,axis=1)**2)
                    #self.cost_func_ls[:]=cost_func_ls*np.ones(len(self.cost_func_ls))
                
                    print(f'batch {batch} MP Nominal: {self.cost_func_mp_nominal[0]} ')
                    print(f'batch {batch} MP Real: {self.cost_func_mp_real[0]} ')
                    print(f'batch {batch} LS: {self.cost_func_ls[0]}')
                    #print(f'batch {batch} MpNet: {self.cost_func_test[0]}')
                    print(f'batch {batch} Constrained MpNet: {self.cost_func_test_c[0]} ')
                
                
                
                            
                #Train channel
                path_init=Path.cwd()
                file_name = f'Data/1e-3/data_var_snr/batch_{batch}.npz'  
                train_data = np.load(path_init/file_name)
                
                h_train        =    torch.tensor(train_data['h'],dtype=torch.complex128).to(device)         
                h_noisy_train  =    torch.tensor(train_data['h_noisy'],dtype=torch.complex128).to(device)       
                sigma_2_train  =    torch.tensor(train_data['sigma_2']).to(device)      
                norm_train     =    torch.norm(h_noisy_train,p=2,dim=1).to(device)

                ##preprocessing

               # normalize channels 
                h_noisy_train   = h_noisy_train / norm_train[:,None]
                h_train         = h_train / norm_train[:,None]


                # get Measurement matrix and multiply channels 
                M_train = torch.tensor(np.load(path_init/'Data'/f'Measurement_matrix/L_{self.L}_T_{self.T}'/f'batch_{batch}.npz')['M_train'],dtype=torch.complex128).to(device)
               
                
                h_noisy_train_M  =torch.matmul((h_noisy_train).unsqueeze(1),torch.conj(M_train)).squeeze().to(device)
                norm_train_M     = torch.norm(h_noisy_train_M,p=2,dim=1).to(device)
                h_noisy_train_M  = h_noisy_train_M/ norm_train_M[:,None]


                #Stopping criteria
                self.sigma_noise = (torch.sqrt(sigma_2_train  )/norm_train_M).to(device)


                ##forward propagation
    
                # Reset gradients to zero
            
                self.constrained_optimizer.zero_grad()
                #self.optimizer.zero_grad()
    
            
                #residuals_mp, est_chan, test_temp= self.mpNet(h_noisy_train,self.M,self.k,self.sigma_noise,2)
                residuals_mp_c, est_chan_c, test_temp_c= self.mpNet_Constrained(h_noisy_train_M,h_noisy_train,M_train,self.L, self.k,self.sigma_noise,2)
    
                
                ##Backward propagation
                #out_mp = torch.abs(residuals_mp).pow(2).sum()/train_channels.size()[0]
                out_mp_c = torch.abs(residuals_mp_c).pow(2).sum()/h_noisy_train.size()[0]
 
                out_mp_c.backward()
                #out_mp.backward()      
                
            
            
    
    
                #Gradient calculation
            
                self.constrained_optimizer.step()
                #self.optimizer.step()
            
                '''
                for name, param in self.mpNet_Constrained.named_parameters():
                    # Vérifiez si le paramètre a un gradient non nul
                    if param.grad is not None:
                        print(f'Paramètre : {name}, Gradient : \n{param.grad}')
                    else:
                        print(f'Paramètre : {name}, Pas de gradient calculé')                
            
    
                #print(parameters_to_vector(self.mpNet_Constrained.parameters()))
    
                    #print('out_mp1___in training',out_mp1)
                for name, param in self.mpNet_Constrained.named_parameters():
                    print(f'Paramètre :a {name} : {param.is_leaf}, Nécessite des gradients : {param.requires_grad} , grad_fn {param.grad_fn} ')
                
                '''
            
    
    
                if batch == self.batch_size-1:
                    #----save the model----
                    torch.save(self.mpNet_Constrained,f'pretrained_models/mpnet_L_{self.L}_T_{self.T}_noise_{self.noise_var}.pth')
                    # torch.save(self.mpNet_Constrained.state_dict(), f'pretrained_model/mpnet_sigma_{self.noise_var}_L_{self.L}_T_{self.T}.pth')

            
                if batch%self.batch_subsampling == 0 and batch != 0:
    
    
                    with torch.no_grad():  
        
                        #residuals_mp_test, est_chan_test, test_temp_test = self.mpNet(h_noisy_test_M,h_noisy_test,M_test, self.k,sigma_norm_test,2)
                        residuals_mp_test_c, est_chan_test_c, test_temp_test_c= self.mpNet_Constrained(h_noisy_test_M,h_noisy_test,M_test,self.L, self.k,sigma_norm_test,2)
                
                    #Estimate channel reconstruction
        
                    #h_hat_mpNet_test =  test_temp_test.detach().numpy()
                    h_hat_mpNet_test_c = test_temp_test_c.detach().cpu().numpy()
    
                
    
                    #Cost function  
    
                    #self.cost_func_test[idx_subs]=torch.mean(torch.sum(torch.abs(test_channels_clean-h_hat_mpNet_test)**2,1)/torch.sum(torch.abs(test_channels_clean)**2,1))
                    self.cost_func_test_c[idx_subs]=torch.mean(torch.sum(torch.abs(h_test.cpu()-h_hat_mpNet_test_c)**2,1)/torch.sum(torch.abs(h_test.cpu())**2,1))
    
    
                    #Channel metrics  
                    #mpnet        
                    #rel_er_mpNet_test = np.mean(np.linalg.norm(test_channels_clean-h_hat_mpNet_test,2,axis=1)**2/np.linalg.norm(test_channels_clean,2,axis=1)**2)
                    ##self.snr_out_dB_mpNet[idx_subs] = 10*np.log10(1/rel_er_mpNet_test)
                
                    #mpnet constrained
                    rel_er_mpNet_test_c = np.mean(np.linalg.norm(h_test.cpu()-h_hat_mpNet_test_c,2,axis=1)**2/np.linalg.norm(h_test.cpu(),2,axis=1)**2)
                    self.snr_out_c_mpnet[idx_subs] = 10*np.log10(1/rel_er_mpNet_test_c)
                
    
    
                
                    #print(f'batch {batch} MP Nominal: {self.cost_func_mp_nominal[idx_subs]} ')
                    print(f'batch {batch} MP Real: {self.cost_func_mp_real[idx_subs]} ')
                    #print(f'batch {batch} LS: {self.cost_func_ls[idx_subs]}')
                    #print(f'batch {batch} MpNet: {self.cost_func_test[idx_subs]}')
                    print(f'batch {batch} Constrained MpNet: {self.cost_func_test_c[idx_subs]} ')
    
                
                
    
    
    
                    idx_subs += 1          


        elif self.train_type=='Offline':
            #Load test, train and validation   data
            path_init=Path.cwd()
            file_name = 'Data/1e-3'  
            data = np.load(path_init/file_name/'data.npz')
            idx_test  = self.nb_channels_test
            idx_val   = self.nb_channels_val
            idx_train = self.nb_channels_train


            test_channels_clean=torch.tensor(data['channels'][:idx_test],dtype=torch.complex128) # clean channels not normalized
            channel_norm_test=torch.tensor(data['channels_norm'][:idx_test]) #channels norm
            h_noisy_test=torch.tensor(data['h_noisy'][:idx_test],dtype=torch.complex128)# normalized
            sigma_2_test=torch.tensor(data['sigma_2'][:idx_test])

            val_channels_clean=torch.tensor(data['channels'][idx_test:idx_test+idx_val],dtype=torch.complex128) # clean channels not normalized
            channel_norm_val=torch.tensor(data['channels_norm'][idx_test:idx_test+idx_val]) #channels norm
            h_noisy_val=torch.tensor(data['h_noisy'][idx_test:idx_test+idx_val],dtype=torch.complex128)# normalized
            sigma_2_val=torch.tensor(data['sigma_2'][idx_test:idx_test+idx_val])            

            train_channels_clean=torch.tensor(data['channels'][idx_test+idx_val:idx_test+idx_val+idx_train],dtype=torch.complex128) # clean channels not normalized
            channel_norm_train=torch.tensor(data['channels_norm'][idx_test+idx_val:idx_test+idx_val+idx_train]) #channels norm
            h_noisy_train=torch.tensor(data['h_noisy'][idx_test+idx_val:idx_test+idx_val+idx_train],dtype=torch.complex128)# normalized
            sigma_2_train=torch.tensor(data['sigma_2'][idx_test+idx_val:idx_test+idx_val+idx_train])            


            # generate Measurements 
            phases=np.random.uniform(0,2*np.pi,(idx_test+idx_val+idx_train,h_noisy_train.shape[1],self.m))
            M= utils_function.generate_M(phases)

            M_test = M[:idx_test]
            M_val = M[idx_test:idx_test+idx_val]
            M_train = M[idx_test+idx_val:idx_test+idx_val+idx_train]

            #Multiply channels 
            h_noisy_train_M=torch.matmul((h_noisy_train).unsqueeze(1),torch.conj(M_train)).squeeze()
            norm_train=torch.norm(h_noisy_train_M,p=2,dim=1)
            h_noisy_train_M=h_noisy_train_M/norm_train[:,None] # normalized

            h_noisy_test_M=torch.matmul((h_noisy_test).unsqueeze(1),torch.conj(M_test)).squeeze()
            norm_test=torch.norm(h_noisy_test_M,p=2,dim=1)
            h_noisy_test_M=h_noisy_test_M/norm_test[:,None] # normalized

            h_noisy_val_M=torch.matmul((h_noisy_val).unsqueeze(1),torch.conj(M_val)).squeeze()
            norm_val=torch.norm(h_noisy_val_M,p=2,dim=1)
            h_noisy_val_M=h_noisy_val_M/norm_val[:,None] # normalized


            # Sigma for SC2

            sigma_test  = torch.sqrt(sigma_2_test  )/norm_test
            sigma_train = torch.sqrt(sigma_2_train  )/norm_train
            sigma_val   = torch.sqrt(sigma_2_val )/norm_val

            # costs train and validation
            train_cost=[]
            val_cost=[]
            test_cost=[]

            for epochs in trange(self.epochs):
                   
                    for b in range(0, self.nb_channels_train, self.batch_size):
                        chan_noisy_M= h_noisy_train_M[b:b+self.batch_size]
                        chan_clean = train_channels_clean[b:b+self.batch_size]
                        chan_noisy = h_noisy_train[b:b+self.batch_size]
                        chan_norm  = channel_norm_train[b:b+self.batch_size]
                        chan_sigma = sigma_train[b:b+self.batch_size]
                        chan_M= M_train[b:b+self.batch_size]


                        #forward propagation
                        residuals_mp_c, est_chan_c, test_temp_c= self.mpNet_Constrained(chan_noisy_M,chan_noisy,chan_M,self.L, self.k,chan_sigma,2)
            
                        #Backward propagation
                        self.constrained_optimizer.zero_grad()
                        out_mp_c = torch.abs(residuals_mp_c).pow(2).sum()/chan_clean.size()[0]
                        out_mp_c.backward()
                        self.constrained_optimizer.step()  
                        
                    with torch.no_grad():
                        _, _, train_temp_c= self.mpNet_Constrained(h_noisy_train_M,h_noisy_train,M_train,self.L, self.k,sigma_train,2)
                         
                        train_cost.append(torch.mean(torch.sum(torch.abs(train_channels_clean-train_temp_c)**2,1)/torch.sum(torch.abs(train_channels_clean)**2,1)))
                        
                        _, _, val_temp_c= self.mpNet_Constrained(h_noisy_val_M,h_noisy_val,M_val,self.L, self.k,sigma_val,2)

                        val_cost.append(torch.mean(torch.sum(torch.abs(val_channels_clean-val_temp_c)**2,1)/torch.sum(torch.abs(val_channels_clean)**2,1)))

                        _, _, test_temp_c= self.mpNet_Constrained(h_noisy_test_M,h_noisy_test,M_test,self.L, self.k,sigma_test,2)

                        test_cost.append(torch.mean(torch.sum(torch.abs(test_channels_clean-test_temp_c)**2,1)/torch.sum(torch.abs(test_channels_clean)**2,1)))
   

            return     train_cost,val_cost,test_cost
                                  

                        
                       
                       

                       
                          
    def run_mp_omp(self,clean_channels:np.ndarray, noisy_channels:np.ndarray,h_noisy:np.ndarray,M:np.ndarray, SC: np.ndarray, type_dict: str) -> Tuple[float,float]:
       
 
 
           
 
            out_chans= torch.zeros_like(h_noisy,dtype=torch.complex128)
   
 
            for i in range(noisy_channels.shape[0]):
                if type_dict == 'nominal':
                    
 
                    dict_nominal_M=torch.matmul(torch.conj(M.mT) , self.dict_nominal).type(torch.complex128).to(device)
   
                    for j in range(dict_nominal_M.shape[0]):
   
                            
                            norm = torch.tensor(np.sqrt(np.sum(np.abs(dict_nominal_M[j].detach().cpu().numpy())**2,0))).to(device)
                   
                            dict_nominal_M[j] = (dict_nominal_M[j]/norm).type(torch.complex128).to(device)
                       
                  
                    out_chans[i,:]= sparse_recovery.mp(noisy_channels[i,:],h_noisy[i,:],dict_nominal_M[i,:],self.dict_nominal,self.k,False,SC[i])
               
               
                elif type_dict == 'real':
                    dict_real_M=torch.matmul(torch.conj(M.mT) , self.dict_real).type(torch.complex128).to(device)
   
                    for j in range(dict_real_M.shape[0]):
   
                       
                            norm = torch.tensor(np.sqrt(np.sum(np.abs(dict_real_M[j].detach().cpu().numpy())**2,0))).to(device)
                   
                            dict_real_M[j] = (dict_real_M[j]/norm).type(torch.complex128).to(device)
                       
   
                               
                   
                    out_chans[i,:]= sparse_recovery.mp(noisy_channels[i,:],h_noisy[i,:],dict_real_M[i,:],self.dict_real,self.k,False,SC[i])
           
                else:
               
                    sys.exit('undefined dictionnary type, either real or nominal!')
               
            rel_err=np.mean(np.linalg.norm(out_chans.cpu()-clean_channels.cpu(),2,axis=1)**2/np.linalg.norm(clean_channels.cpu(),2,axis=1)**2)
       
            snr_out_db = 10*np.log10(1/rel_err)
           
       
            return rel_err,snr_out_db
                 
       
 
