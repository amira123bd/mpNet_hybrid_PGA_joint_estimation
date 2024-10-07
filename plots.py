import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
   
   
path_init = Path.cwd()
def plot_SNR(self):
       
 
        plt.rcParams['text.usetex'] = True
       
       
        plt.figure()
        vec_base = np.arange(0,self.batch_size/self.batch_subsampling,1)
        #plt.plot(vec_base,self.snr_out_dB_mpNet,'x-',color='red',linewidth=0.7,label='mpNet ')
        plt.plot(vec_base,self.snr_out_c_mpnet,'>-',color='red',linewidth=0.7,label='mpNet Constrained ')
        plt.plot(vec_base,self.snr_out_dB_omp_nominal,'*-',color='c',linewidth=0.7,label='MP (Nominal dictionnary) ')
        plt.plot(vec_base,self.snr_out_dB_omp_real,'+-',color='k',linewidth=0.7,label='MP (Real dictionnary)')
        plt.ylabel('SNR_out (dB)')
        plt.xlabel(f'Number of seen channels (*{self.batch_subsampling*self.nb_channels_per_batch})')
        plt.legend(loc = 'lower right')
        plt.title(f'SNR_out evolution ')
        plt.grid()
        plt.xlim(left=0)
       
        plt.savefig(path_init/'Results'/('SNR_out_SNR_in_' + str(self.snr_in_dB)+'dB_'+\
           str(self.epochs)+'epoch_'+str(self.batch_size)+'_batch_'+\
               str(self.nb_channels_per_batch)+'_channels_per_batch_'),dpi=500)
        plt.show()
 
 
def plot_NMSE_Online(self,L,T):
        plt.rcParams['text.usetex'] = True
        plt.figure()
        vec_base = np.arange(0,self.batch_size/self.batch_subsampling,1)
        #plt.plot(vec_base,self.cost_func_test,'p-',linewidth=0.8,label='mpNet')  
        plt.plot(vec_base,self.cost_func_test_c,'>-',linewidth=0.8,label='mpNet Constrained ')
        plt.plot(vec_base,self.rel_err_omp_nominal,'*-',linewidth=0.8,label='MP (Nominal dictionnary) ')
        plt.plot(vec_base,self.rel_err_omp_real,'+-',linewidth=0.8,label='MP (Real dictionnary) ')  
        #plt.plot(vec_base,self.cost_func_ls,'o--',linewidth=0.8,label='LS')  
        plt.grid()
        plt.legend(loc = 'best')
        plt.xlabel(f'Number of seen channels (*{self.batch_subsampling*self.nb_channels_per_batch})')
        plt.ylabel('NMSE')
        plt.title(f'NMSE evolution L={L} T={T} ')
        plt.xlim(left=0)
 
        plt.savefig(path_init/'Results'/('NMSE_SNR_in_' + str(self.snr_in_dB)+'dB_'+\
            str(self.epochs)+'epoch_'+str(self.batch_size)+'_batch_'+\
                str(self.nb_channels_per_batch)+'_channels_per_batch_'),dpi=500)
       
        plt.show()
       
def plot_NMSE_r_Online(self):
        plt.rcParams['text.usetex'] = True
        plt.figure()
        vec_base = np.arange(0,self.batch_size/self.batch_subsampling,1)
        plt.plot(vec_base,self.cost_func_test,'p-',linewidth=0.8,label='mpNet')  
        plt.plot(vec_base,self.cost_func_test_c,'^-',linewidth=0.8,label='mpNet Constrained ')
        plt.plot(vec_base,self.cost_func_test_c_r,'>-',linewidth=0.8,label='mpNet Constrained (Random initialization)')
        plt.plot(vec_base,self.rel_err_omp_nominal,'*-',linewidth=0.8,label='OMP (Nominal dictionnary) ')
        plt.plot(vec_base,self.rel_err_omp_real,'+-',linewidth=0.8,label='OMP (Real dictionnary) ')  
        plt.plot(vec_base,self.cost_func_ls,'o--',linewidth=0.8,label='LS')  
        plt.grid()
        plt.legend(loc = 'best')
        plt.xlabel(f'Number of seen channels (*{self.batch_subsampling*self.nb_channels_per_batch})')
        plt.ylabel('NMSE')
        plt.title(f'NMSE evolution ')
        plt.xlim(left=0)
 
        plt.savefig(path_init/'Results'/('NMSE_SNR_in_' + str(self.snr_in_dB)+'dB_'+\
            str(self.epochs)+'epoch_'+str(self.batch_size)+'_batch_'+\
                str(self.nb_channels_per_batch)+'_channels_per_batch_'),dpi=500)
       
        plt.show()
 
 
def plot_NMSE_positions(self):
        plt.rcParams['text.usetex'] = True
        plt.figure()
        vec_base = np.arange(0,self.batch_size/self.batch_subsampling,1)
        plt.plot(vec_base,self.cost_func_position,'p-',linewidth=0.7)  
 
        plt.grid()
        #plt.legend(loc = 'best')
        plt.xlabel(f'Number of seen channels (*{self.batch_subsampling*self.nb_channels_per_batch})')
        plt.ylabel('NMSE')
        plt.title(f'Position NMSE evolution ')
        plt.xlim(left=0)
 
        plt.savefig(path_init/'Results'/('Position_NMSE_' + str(self.snr_in_dB)+'dB_'+\
            str(self.epochs)+'epoch_'+str(self.batch_size)+'_batch_'+\
                str(self.nb_channels_per_batch)+'_channels_per_batch_'),dpi=500)      
       
        plt.show()
       
       
def plot_NMSE_Offline(self):
        plt.rcParams['text.usetex'] = True
        plt.figure()
        vec_base = np.arange(0,self.epochs+1,1)
        #plt.plot(vec_base,self.cost_func_offline,'p-',linewidth=0.9,label='mpNet')
        plt.plot(vec_base,self.cost_func_r_offline,'^-',linewidth=0.9,label='Mpnet(Random initialization)')
        plt.plot(vec_base,self.cost_func_c_offline,'>-',linewidth=0.9,label='Constrained Mpnet')
         
 
        plt.grid()
        #plt.legend(loc = 'best')
        plt.xlabel(f'epochs ')
        plt.ylabel('NMSE')
        plt.title(f'NMSE evolution in Offline learning mode')
        plt.xlim(left=0)
 
        plt.savefig(path_init/'Results'/('NMSE_offline' + str(self.snr_in_dB)+'dB_'+\
            str(self.epochs)+'epoch_'+str(self.batch_size)+'_batch_'+\
                str(self.nb_channels_per_batch)+'_channels_per_batch_'),dpi=500)      
       
        plt.show()  
        
def plot_learning_curve(train_losses,
                        valid_losses,
                        num_of_iter_pga_unf,
                        epochs,
                        batch_size,
                        train_size,
                        test_size,
                        valid_size,L):
        y_t = [r.detach().numpy() for r in train_losses]
        x_t = np.array(list(range(len(train_losses))))
        y_v = [r.detach().numpy() for r in valid_losses]
        x_v = np.array(list(range(len(valid_losses))))
        plt.figure()
        plt.plot(x_t, y_t, 'o-', label='Train')
        plt.plot(x_v, y_v, '*-', label='Valid')
        plt.grid()
        plt.title(f'Loss Curve, Num Epochs = {epochs}, Batch Size = {batch_size} \n Num of Iterations of PGA = {num_of_iter_pga_unf}')
        plt.xlabel('Epoch')
        plt.legend(loc='best')
        plt.savefig(path_init/'Results'/('Loss_Curve_L_' + str(L)+'_train_size'+\
                str(train_size)+'_test_size'+str(test_size)+'_val_size_'+\
                str(valid_size)+'_epochs_'+str(epochs)+'_batch_size_'+str(batch_size)),dpi=500)

        plt.show()
        
def plot_Sum_Rate_Unfolded_PGA(sum_rate_unf,
                               num_of_iter_pga_unf,
                               train_size,
                               test_size,
                               valid_size,
                               epochs,
                               batch_size,L,name):
        plt.figure()
        y_unf = [r.detach().numpy() for r in (sum(sum_rate_unf)/sum_rate_unf.shape[0])]
        x = np.array(list(range(num_of_iter_pga_unf))) +1
        plt.plot(x, y_unf, 'o-')
        plt.title(f'The Average Achievable Sum-Rate\n of the unfolded PGA with {name}')
        plt.xlabel('Number of Iteration')
        plt.ylabel('Achievable Rate')
        plt.grid()
        plt.savefig(path_init/'Results'/('Mpnet_unfolded_PGA_L_' + str(L)+'_train_size'+\
                str(train_size)+'_test_size'+str(test_size)+'_val_size_'+\
                str(valid_size)+'_epochs_'+str(epochs)+'_batch_size_'+str(batch_size)),dpi=500)
        plt.show()
        
def plot_Classical_vs_unfolded_PGA(sum_rate_class,
                                   sum_rate_unf,
                                   num_of_iter_pga,
                                   num_of_iter_pga_unf,
                                   test_size,train_size,valid_size,
                                   epochs,batch_size,L,name):
        # ploting the results
        plt.figure()
        y = [r.detach().numpy() for r in (sum(sum_rate_class)/sum_rate_class.shape[0])]
        y_unf = [r.detach().numpy() for r in (sum(sum_rate_unf)/sum_rate_unf.shape[0])]

        x = np.array(list(range(num_of_iter_pga))) +1
        x_unf = np.array(list(range(num_of_iter_pga_unf))) +1

        plt.plot(x_unf, y_unf, 'X-',label='Unfolded PGA')
        plt.plot(x, y, 'o-',label='Classical PGA ')
        plt.title(f'The Average Achievable Sum-Rate  \n of the unfolded PGA vs classical PGA with {name}')
        plt.xlabel('Number of Iteration')
        plt.ylabel('Achievable Rate')
        plt.legend(loc = 'lower right')
        plt.grid()
        plt.savefig(path_init/'Results'/('Mpnet_unfolded_vs_classical_PGA_L_' + str(L)+'_train_size'+\
                str(train_size)+'_test_size'+str(test_size)+'_val_size_'+\
                str(valid_size)+'_epochs_'+str(epochs)+'_batch_size_'+str(batch_size)),dpi=500)
        plt.show()