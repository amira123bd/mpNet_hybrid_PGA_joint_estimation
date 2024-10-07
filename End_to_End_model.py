#%%
import torch.nn as nn
import torch
seed_value=42
import numpy as np
import mpnet_model
import generate_steering


#%% Define CompositeModel
class CompositeModel(nn.Module):
    
    def __init__(self, pga, constrained_mpnet,antenna_position,DoA,g_vec,lambda_, config_type):

        w_init=generate_steering.steering_vect_c(torch.tensor(antenna_position).type(torch.FloatTensor),
                                                       torch.tensor(DoA).type(torch.FloatTensor),
                                                       torch.tensor(g_vec),
                                                       lambda_).type(torch.complex128)

        
        super(CompositeModel, self).__init__()
        if config_type == 'C_Mpnet':

             self.mpnet = constrained_mpnet

        elif config_type == 'Mpnet':

             self.mpnet = mpnet_model.mpNet(w_init)

        else:
            print('"Mpnet" or "C_Mpnet"')
        self.pga = pga


    def forward(self, x,h,h_true,norm,k,sigma,M_matrix, N,M,L,num_of_iter,noise_var):
        torch.manual_seed(seed_value)
       #input--------------------------------------------------
       #x= noisy channels multiplied by the measurement matrix
       #h= noisy channels
       #h_true = true channels
       #k= max iteration of iterative mpnet 
       #M_matrix= measurement matrix
       #N=number of UE
       #M=BS antenna
       #L=RF chains
       #num_of_iter=number of iteration for PGA
       #noise_var= the noise variance 
       #-------------------------------------------------------


        residual , _, h_hat= self.mpnet(x,h,M_matrix,L,k,sigma,2)
    
        h_hat_mpNet_test_c = torch.tensor(h_hat.detach().numpy(),dtype=torch.complex128)
        print('NMSE: ',torch.mean(torch.sum(torch.abs(h_true-h_hat_mpNet_test_c*norm[:,None])**2,1)/torch.sum(torch.abs(h_true)**2,1)))
        #print(h_hat_mpNet_test_c*norm[:,None])

        #print(((h_hat*norm[:,None]).view(-1,N,M)))

        sum_rate , wa, wd,WA,WD = self.pga(((h_hat*norm[:,None]).view(-1,N,M)).type(torch.complex128) ,N,L,num_of_iter,noise_var)
        

   
        

        return sum_rate , wa, wd,WA,WD
    


    def freeze_parameters(self,model):
        for param in model.parameters():
            param.requires_grad=False


    def unfreeze_parameters(self,model):
        for param in model.parameters():
            param.requires_grad=True
# %%
