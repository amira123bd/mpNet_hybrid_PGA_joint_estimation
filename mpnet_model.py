from __future__ import print_function
from typing import Optional, Tuple
import numpy as np
import torch
import sys
from torch.functional import Tensor
import torch.nn as nn
from torch.autograd.function import Function, FunctionCtx
import math
import generate_steering
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

#device='cpu'
device='cuda'

 
 
# Custom autograd Functions
class keep_k_max(Function):
   
    @staticmethod
    def forward(ctx: FunctionCtx, activations_in: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input = activations_in.clone().detach()
        
        if input.dim() == 1:
            input = input.unsqueeze(0)
        
        n_samples = input.shape[0]
        d_a = input.shape[1]

        # create activations out 
        activations_out = torch.zeros_like(input, dtype=torch.complex128)
        
        # find the top-k elements
        abs_input = torch.abs(input)
        _, topk_indices = torch.topk(abs_input, k, dim=1, largest=True, sorted=False)
        
        for i in range(n_samples):
            activations_out[i, topk_indices[i]] = input[i, topk_indices[i]]
        
        # Save activations that correspond to the selected atoms for backward propagation

        ctx.save_for_backward(activations_out)
        return activations_out, topk_indices
 
    @staticmethod
    def backward(ctx: FunctionCtx,grad_output: Tensor, id_k_max: Tensor) -> Tuple[Tensor, None, None]:
       
 
        activations_out, = ctx.saved_tensors
       
        grad_input = grad_output.clone()
        grad_input[activations_out == 0] = 0
        return grad_input, None, None
   



class mpNet_Constrained(nn.Module):
    def __init__(self, ant_position: torch.float, DoA: Tensor, g_vec: np.ndarray, lambda_, normalize: bool = True) -> None:
        super().__init__()
        
        
        self.ant_position = nn.Parameter((ant_position).to(device))
        
       
        self.DoA = DoA.to(device)
        self.g_vec = g_vec
        self.normalize = normalize
        self._lambda_ = lambda_
    
    def forward(self, x: Tensor, x_m: Tensor, M: Tensor, L: int, k: int, sigma: Optional[float] = None, sc: int = 2) -> Tuple[Tensor, Tensor, Optional[np.ndarray]]:
       
        residual = x.clone()
        res = x_m.clone()

        if self.normalize:
            W = generate_steering.steering_vect_c(self.ant_position, self.DoA, self.g_vec, self._lambda_).type(torch.complex128).to(device)
            M_D = torch.matmul(torch.conj(M.mT), W).type(torch.complex128).to(device)
            for i in range(M_D.shape[0]):
                norm = torch.tensor(np.sqrt(np.sum(np.abs(M_D[i].detach().cpu().numpy()) ** 2, 0))).to(device)
                M_D[i] = (M_D[i] / norm).type(torch.complex128).to(device)
        else:
            W = generate_steering.steering_vect_c(self.ant_position, self.DoA, self.g_vec, self._lambda_).type(torch.complex128).to(device)
            M_D = torch.matmul(torch.conj(M.mT), W).type(torch.complex128).to(device)



  
        E = W.clone()
        norm_E = torch.tensor(np.sqrt(np.sum(np.abs(E.detach().cpu().numpy()) ** 2, 0))).to(device)
        E = (E / norm_E).type(torch.complex128).to(device)

        if sigma is None:  # No stopping criterion
            residuals = []
            for iter in range(k):
                z, id_k_max = keep_k_max.apply(residual @ torch.conj(M_D), 1)
                residual = residual - (z @ M_D.T)
                residuals.append(residual)

            x_hat = x - residual
            return residual, x_hat, None
        
       


        else:  # Use stopping criterion
            m = np.shape(residual)[1] # mesures
            N = np.shape(res)[1] # BS antenna
            if sc == 1:  # SC1
                threshold = pow(sigma, 2) * ( N*L + 2 * math.sqrt(N*L * math.log(N*L)))
            elif sc == 2:  # SC2
                threshold = pow(sigma, 2) * N * L 
            

            current_ids = list(range(residual.size()[0]))
      
            depths = np.zeros(residual.size()[0])
            iter = 0

            while bool(current_ids) and iter < 20:

                res_norm_2 = torch.norm(residual, p=2, dim=1) ** 2

         

                for i in current_ids[:]:
                    
                    if res_norm_2[i] < threshold[i]:
                        depths[i] = iter
                        current_ids.remove(i)
                        
                    else:
                 
                        # X_hat estimation [x= h @ M]
                        z, id_k_max = keep_k_max.apply(residual[i].clone() @ torch.conj(M_D[i]), 1)
                        residual[i] = residual[i].clone() - (z @ M_D[i].T).clone()

                        
                        # X_hat_m estimation [ x = h ]
                        res[i] = res[i] - (z @ E.T)
                

                iter += 1

            x_hat = x - residual
            x_hat_m = x_m - res

            print(iter)

        


            return residual, x_hat, x_hat_m
   





class mpNet(nn.Module):
   
   
       
    def __init__(self, W_init: Tensor) -> None:
            # W shape: [N,A]
            super().__init__()
 
           
             
            self.W = nn.Parameter(W_init).to(device)
 
 
 
       
           
           
               
    def forward(self, x: Tensor,x_m: Tensor,M:Tensor, L:int,  k: int,sigma: Optional[float] = None, sc: int = 2) -> Tuple[Tensor, Tensor, Optional[np.ndarray]]:
            #X shape: (nb_samples,N,nb_users)
           
        residual = x.clone()
        res = x_m.clone()
       
 
        ##M@ D
        M_D=torch.matmul(torch.conj(M.mT),self.W).type(torch.complex128).to(device)
 
        for i in range(M_D.shape[0]):
 
            norm = torch.tensor(np.sqrt(np.sum(np.abs(M_D[i].detach().cpu().numpy())**2,0))).to(device)
           
            M_D[i] = (M_D[i]/norm).type(torch.complex128).to(device)
 
 
        ##temprary dict to store chosen index
        E=self.W.clone()
        norm_E = torch.tensor(np.sqrt(np.sum(np.abs(E.detach().cpu().numpy())**2,0))).to(device)
        E= (E/ norm_E).type(torch.complex128).to(device)
 
        #print('E',E)
 
 
       
       
       
        if sigma is None:  # no stopping criterion
            residuals = []
            for iter in range(k):
                z, id_k_max = keep_k_max.apply(residual @ torch.conj(self.W), 1)
 
                residual = residual -  (z  @   self.W.T)
     
                residuals.append(residual)
               
 
            x_hat = x - residual
           
 
            return residual,x_hat,None
 
 
 
        else:  # Use stopping criterion
            m = np.shape(residual)[1] # mesures
            N = np.shape(res)[1] # BS antenna
            if sc == 1:  # SC1
                threshold = pow(sigma, 2) * ( N*L + 2 * math.sqrt(N*L * math.log(N*L)))
            elif sc == 2:  # SC2
                threshold = pow(sigma, 2) * N * L 
            

            current_ids = list(range(residual.size()[0]))
      
            depths = np.zeros(residual.size()[0])
            iter = 0

            while bool(current_ids) and iter < 20:

                res_norm_2 = torch.norm(residual, p=2, dim=1) ** 2
         

                for i in current_ids[:]:
                    
                    if res_norm_2[i] < threshold[i]:
                        depths[i] = iter
                        current_ids.remove(i)
                    else:
                 
                        # X_hat estimation [x= h @ M]
                        z, id_k_max = keep_k_max.apply(residual[i].clone() @ torch.conj(M_D[i]), 1)
                        residual[i] = residual[i].clone()- (z @ M_D[i].T).clone()

                   

                        
                        # X_hat_m estimation [ x = h ]

                        res[i] = res[i] - (z @ E.T)
                

                iter += 1

            x_hat = x - residual
            x_hat_m = x_m - res

            print(iter)


            return residual, x_hat, x_hat_m