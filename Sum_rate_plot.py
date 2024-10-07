#%%
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
#%%

path_init=Path.cwd()
L=10
N=4
T=1
#%%

unf_true_channels= np.load(path_init/'sumRate'/'1e-3'/'unf_true_channel.npz')['unf_true_channel']
unf_estimated_channels= np.load(path_init/'sumRate'/'1e-3'/'unf_est_mpnet_channel.npz')['unf_true_channel']
unf_nominal=np.load(path_init/'sumRate'/'1e-3'/'unf_est_mp_nominal_channel.npz')['unf_est_mp_nominal_channel']
unf_real=np.load(path_init/'sumRate'/'1e-3'/'unf_est_mp_real_channel.npz')['unf_est_mp_real_channel']
#unf_ls=np.load(path_init/'sumRate'/'1e-3'/'unf_est_ls.npz')['unf_est_ls']


class_true_channels=np.load(path_init/'sumRate'/'1e-3'/'class_true_channel.npz')['class_true_channel']
class_estimated_channels=np.load(path_init/'sumRate'/'1e-3'/'class_est_mpnet_channel.npz')['class_est_mpnet_channel']
#class_nominal=np.load(path_init/'sumRate'/'1e-3'/'class_est_mp_nominal_channel.npz')['class_est_mp_nominal_channel']
#class_real=np.load(path_init/'sumRate'/'1e-3'/'class_est_mp_real_channel.npz')['class_est_mp_real_channel']
#class_ls=np.load(path_init/'test/'sumRate'/'1e-3'/'class_est_ls_channel.npz')['class_est_ls_channel']

end_to_end=np.load(path_init/'sumRate'/'1e-3'/'End_To_End.npz')['End_To_End']



WF=np.load(path_init/'sumRate'/'1e-3'/'wf.npz')['wf']
#FD=np.load(path_init/'sumRate'/'1e-3'/'fully_digital.npz')['fully_digital']

# %%
plt.rcParams['text.usetex'] = True
plt.figure()

num_of_iter_pga_unf=10
num_iter_pga_class=50
x_unf = np.array(list(range(num_of_iter_pga_unf))) +1
x_class = np.array(list(range(num_iter_pga_class))) +1
#y_WF=WF*np.ones(num_iter_pga_class)

#y_FD=FD*np.ones(num_iter_pga_class)

#plt.plot(x_class, class_true_channels, '*--',label='Classical PGA - Perfect CSI',color='#FFA500')
plt.plot(x_class, class_estimated_channels, '+--',label='Classical PGA - MpNet Estimation',color='g')
#plt.plot(x_class, class_nominal, 'p--',label='Classical PGA - Nominal MP Estimation',color='b')
#plt.plot(x_class, class_real, 'p--',label='Classical PGA - Real MP Estimation',color='red')
#plt.plot(x_class, class_ls, 'p--',label='Classical PGA - LS Estimation',color='#AAA510')

#plt.plot(x_unf, unf_true_channels, marker='+', markersize=7, markeredgewidth=1, linestyle='-', label='Unfolded PGA - Perfect CSI',color='#FFA500')
plt.plot(x_unf, unf_estimated_channels, 'x-',label='Unfolded PGA - MpNet Estimation')
#plt.plot(x_unf, unf_nominal, 'p-',label='Unfolded PGA - Nominal MP Estimation',color='b')
#plt.plot(x_unf, unf_real, 'x-',label='Unfolded PGA - Real MP Estimation',color='red')
#plt.plot(x_unf, unf_ls, 'x-',label='Unfolded PGA - LS Estimation',color='#AAA510')

plt.plot(x_unf, end_to_end, 'x-',label='Supervised End to End learning')
#plt.plot(x_class, y_WF, '--',label='Water_filling')
#plt.plot(x_class, y_FD, '--',label='Fully Digital',color='red')
plt.xlim(1,50) 
#plt.ylim(bottom=0)

plt.title(f'Sum-Rate\n L={L} T={T} N={N}  $\\sigma^2$ = 1e-3')
plt.xlabel('Number of Iteration')
plt.ylabel('Achievable Rate')

plt.grid()
plt.legend(loc='lower right')
plt.show()











####################### SUM RATE WITH DIFFERENT VARIANCE #############################

#%%
class_est_3=np.load(path_init/'sumRate'/'1e-3'/'class_est_mpnet_channel.npz')['class_est_mpnet_channel']
unf_est_3=np.load(path_init/'sumRate'/'1e-3'/'unf_est_mpnet_channel.npz')['unf_est_mpnet_channel']
end_to_end_3= np.load(path_init/'sumRate'/'1e-3'/'End_To_End.npz')['End_To_End']
class_true_3= np.load(path_init/'sumRate'/'1e-3'/'class_true_channel.npz')['class_true_channel']
unf_true_3= np.load(path_init/'sumRate'/'1e-3'/'unf_true_channel.npz')['unf_true_channel']
WF_3=np.load(path_init/'sumRate'/'1e-3'/'wf.npz')['wf']

#%%
class_est_8_4=np.load(path_init/'sumRate'/'8e-4'/'class_est_mpnet_channel.npz')['class_est_mpnet_channel']
unf_est_8_4=np.load(path_init/'sumRate'/'8e-4'/'unf_est_mpnet_channel.npz')['unf_est_mpnet_channel']
end_to_end_8_4= np.load(path_init/'sumRate'/'8e-4'/'End_To_End.npz')['End_To_End']
class_true_8_4= np.load(path_init/'sumRate'/'8e-4'/'class_true_channel.npz')['class_true_channel']
unf_true_8_4= np.load(path_init/'sumRate'/'8e-4'/'unf_true_channel.npz')['unf_true_channel']
WF_8_4=np.load(path_init/'sumRate'/'8e-4'/'wf.npz')['wf']

#%%
class_est_6_4=np.load(path_init/'sumRate'/'6e-4'/'class_est_mpnet_channel.npz')['class_est_mpnet_channel']
unf_est_6_4=np.load(path_init/'sumRate'/'6e-4'/'unf_est_mpnet_channel.npz')['unf_est_mpnet_channel']
end_to_end_6_4= np.load(path_init/'sumRate'/'6e-4'/'End_To_End.npz')['End_To_End']
class_true_6_4= np.load(path_init/'sumRate'/'6e-4'/'class_true_channel.npz')['class_true_channel']
unf_true_6_4= np.load(path_init/'sumRate'/'6e-4'/'unf_true_channel.npz')['unf_true_channel']
WF_6_4=np.load(path_init/'sumRate'/'6e-4'/'wf.npz')['wf']

#%%
class_est_4_4=np.load(path_init/'sumRate'/'4e-4'/'class_est_mpnet_channel.npz')['class_est_mpnet_channel']
unf_est_4_4=np.load(path_init/'sumRate'/'4e-4'/'unf_est_mpnet_channel.npz')['unf_est_mpnet_channel']
end_to_end_4_4= np.load(path_init/'sumRate'/'4e-4'/'End_To_End.npz')['End_To_End']
class_true_4_4= np.load(path_init/'sumRate'/'4e-4'/'class_true_channel.npz')['class_true_channel']
unf_true_4_4= np.load(path_init/'sumRate'/'4e-4'/'unf_true_channel.npz')['unf_true_channel']
WF_4_4=np.load(path_init/'sumRate'/'4e-4'/'wf.npz')['wf']



#%%

x=[1e-3,8e-4,6e-4,4e-4]
y1=[class_est_3[49],class_est_8_4[49],class_est_6_4[49],class_est_4_4[49]]
y2=[unf_est_3[9],unf_est_8_4[9],unf_est_6_4[9],unf_est_4_4[9]]
y3=[end_to_end_3[9],end_to_end_8_4[9],end_to_end_6_4[9],end_to_end_4_4[9]]
y4=[class_true_3[49],class_true_8_4[49],class_true_6_4[49],class_true_4_4[49]]
y6=[unf_true_3[9],unf_true_8_4[9],unf_true_6_4[9],unf_true_4_4[9]]

y5=[WF_3,WF_8_4,WF_6_4,WF_4_4]



#%%
#plt.plot(x, y1, 'x--',label='Classical PGA(50 iter) - MpNet Estimation ')
plt.plot(x, y2, 'o-',label='Unfolded PGA(10 iter) - MpNet Estimation')
plt.plot(x, y3, 'x--',label='End To End supervised')
#plt.plot(x, y4, 'p-',label='Classical PGA - Perfect CSI')
plt.plot(x, y6, 'v-',label='Unfolded PGA(10 iter) - Perfect CSI')
plt.plot(x, y5, 'p-',label='Water filling')

# Personnalisation des ticks de l'axe des x
plt.xticks([1e-3,8e-4,6e-4,4e-4], ['1e-3','8e-4','6e-4','4e-4'])
plt.xlim(4e-4,1e-3) 
# Inversion de l'axe des x
plt.gca().invert_xaxis()


plt.title(f'L={L}, T={T}, N={N}')
plt.xlabel('Noise variance')
plt.ylabel('Achievable Rate')
plt.grid(True)
plt.legend(loc='best')

plt.show()
