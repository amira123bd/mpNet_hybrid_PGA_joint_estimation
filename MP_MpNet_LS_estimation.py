import numpy as np
import torch
import utils_function
from tqdm import trange
from pathlib import Path
import sparse_recovery
import mpnet_model
device='cuda'

def save_estimation(model, data_file, data_pred_mpnet, data_pred_MP_nominal, data_pred_MP_real, data_pred_LS, batch_size, k, T, L, dict_nominal, dict_real
                    ,antenna_position,DoA,g_vec,lambda_ ,BS_antenna):
    
    path_init = Path.cwd()

    # Load trained model
    #constrained_mpnet=mpnet_model.mpNet_Constrained(torch.tensor(antenna_position).type(torch.FloatTensor),torch.tensor(DoA).type(torch.FloatTensor),torch.tensor(g_vec),lambda_, True)
    #constrained_mpnet.load_state_dict(torch.load(model))
    constrained_mpnet=torch.load(model)
    
    #constrained_mpnet = torch.load(model)
    #constrained_mpnet.to('cuda')

    # Prepare dictionaries on GPU
    dict_nominal = dict_nominal.to('cuda')
    dict_real = dict_real.to('cuda')



    ####################################           test                   ############################################
    ####################################           Data                   ############################################
    ######################################      MPNET estimation #############################################################
    
    ## estimate test channels 

    # load test data 
    
    test_data = np.load(path_init/data_file/'test_data.npz')
                    
    h_test        =    torch.tensor(test_data['h'],dtype=torch.complex128).to(device)         
    h_noisy_test  =    torch.tensor(test_data['h_noisy'],dtype=torch.complex128).to(device)       
    sigma_2_test  =    torch.tensor(test_data['sigma_2']).to(device)      
    norm_test     =    torch.norm(h_noisy_test,p=2,dim=1).to(device)

    ## preprocessing 

    # normalize channels 
    h_noisy_test   = h_noisy_test / norm_test[:,None]
    h_test         = h_test / norm_test[:,None]

    # Get Measurement matrix
    M_test = torch.tensor(np.load(path_init/'Data'/f'Measurement_matrix/L_{L}_T_{T}'/'test.npz')['M_test'],dtype=torch.complex128).to(device)

    # Multiply channels by M 
    h_noisy_M = torch.matmul(h_noisy_test.unsqueeze(1), torch.conj(M_test)).squeeze()
    channel_norm_M = torch.norm(h_noisy_M, p=2, dim=1)
    h_noisy_M = h_noisy_M / channel_norm_M[:, None]

    # Mpnet Estimation
    sigma_norm = torch.sqrt(sigma_2_test) / channel_norm_M
    SC2 = pow(sigma_norm,2) * BS_antenna * L  

    _, _, est_chan = constrained_mpnet(h_noisy_M, h_noisy_test, M_test, L, k, sigma_norm, 2)
    est_chan = est_chan.detach().cpu().numpy()

    # Save estimation
    est_c_mpnet = {'channels': est_chan}

    # Cost function
    norm_test_dataset = torch.sum(torch.abs(h_test.cpu() )**2, 1)
    mp_error_norm_test = torch.sum(torch.abs(h_test.cpu() - est_chan)**2, 1)
    print('NMSE MpNet Test data: ', torch.mean(mp_error_norm_test / norm_test_dataset))

    # Save data
    np.savez(path_init / data_pred_mpnet / f'test.npz', **est_c_mpnet)

    '''
    print(data_pred_MP_real)
    ######################################### Mp real Estimation ##################################################
    out_chans = torch.zeros_like(h_noisy_test, dtype=torch.complex128).to('cuda')

    for i in range(h_noisy_M.shape[0]):
            dict_real_M = torch.matmul(torch.conj(M_test.mT), dict_real.to('cuda')).to('cuda')

            for j in range(dict_real_M.shape[0]):
                    norm = torch.tensor(np.sqrt(np.sum(np.abs(dict_real_M[j].detach().cpu().numpy())**2, 0)), device='cuda')
                    dict_real_M[j] = (dict_real_M[j] / norm).type(torch.complex128)

            out_chans[i, :] = sparse_recovery.mp(h_noisy_M[i, :], h_noisy_test[i, :], dict_real_M[i, :], dict_real.to('cuda'), k, False, SC2[i])

    # Save estimation
    est_c_mp_r = {'channels': out_chans.cpu().numpy()}

    # Cost function
    norm_test_dataset = torch.sum(torch.abs(h_test.cpu())**2, 1)
    mp_error_norm_test = torch.sum(torch.abs(h_test.cpu() - out_chans.cpu())**2, 1)
    print('NMSE MP REAL test data : ', torch.mean(mp_error_norm_test / norm_test_dataset))

    np.savez(path_init / data_pred_MP_real /'test.npz', **est_c_mp_r)
    
    ######################################### Mp nominal Estimation ##################################################

    out_chans = torch.zeros_like(h_noisy_test, dtype=torch.complex128).to('cuda')

    for i in range(h_noisy_M.shape[0]):
            dict_nominal_M = torch.matmul(torch.conj(M_test.mT), dict_nominal.to('cuda')).to('cuda')

            for j in range(dict_nominal_M.shape[0]):
                norm = torch.tensor(np.sqrt(np.sum(np.abs(dict_nominal_M[j].detach().cpu().numpy())**2, 0)), device='cuda')
                dict_nominal_M[j] = (dict_nominal_M[j] / norm).type(torch.complex128).to('cuda')

            out_chans[i, :] = sparse_recovery.mp(h_noisy_M[i, :], h_noisy_test[i, :], dict_nominal_M[i, :], dict_nominal, k, False, SC2[i])

    # Save estimation
    est_c_mp_n = {'channels': out_chans.cpu().numpy()}

    # Cost function
    norm_test_dataset = torch.sum(torch.abs(h_test.cpu())**2, 1)
    mp_error_norm_test = torch.sum(torch.abs(h_test.cpu() - out_chans.cpu())**2, 1)
    print('NMSE MP NOMINAL test data : ', torch.mean(mp_error_norm_test / norm_test_dataset))

    np.savez(path_init / data_pred_MP_nominal / 'test.npz', **est_c_mp_n)

    

    ####################################           train                  ############################################
    ####################################           Data                   ############################################

    ###################################### MPNET estimation #############################################################


    for batch_idx in trange(batch_size):

        data = np.load(path_init / data_file / f'batch_{batch_idx}.npz')
        h = torch.tensor(data['h'], dtype=torch.complex128).to('cuda')
        h_noisy = torch.tensor(data['h_noisy'], dtype=torch.complex128).to('cuda')
        sigma_2 = torch.tensor(data['sigma_2']).to('cuda')
        channel_norm = torch.norm(h_noisy,p=2,dim=1).to('cuda')
        h = h / channel_norm[:, None]
        h_noisy=h_noisy/channel_norm[:, None]

        # Get Measurement matrix
        #phases_train = np.random.uniform(0, 2 * np.pi, (h_noisy.shape[0], h_noisy.shape[1], m))
        #M = utils_function.generate_M(phases_train).to('cuda')
        M = torch.tensor(np.load(path_init/'Data2'/f'Measurement_matrix/L_{L}_T_{T}'/f'batch_{batch_idx}.npz')['M_train'],dtype=torch.complex128).to(device)

        # Multiply channels by M 
        h_noisy_M = torch.matmul(h_noisy.unsqueeze(1), torch.conj(M)).squeeze().to('cuda')
        channel_norm_M = torch.norm(h_noisy_M, p=2, dim=1).to('cuda')
        h_noisy_M = h_noisy_M / channel_norm_M[:, None]

        # Mpnet Estimation
        sigma_norm = torch.sqrt(sigma_2) / channel_norm_M
        SC2 = pow(sigma_norm,2) * BS_antenna * L 

        _, _, est_chan = constrained_mpnet(h_noisy_M, h_noisy, M, L, k, sigma_norm, 2)
        est_chan = est_chan.detach().cpu().numpy()

        # Save estimation
        est_c = {'channels': est_chan}

        # Cost function
        norm_test_dataset = torch.sum(torch.abs(h.cpu() )**2, 1)
        mp_error_norm_test = torch.sum(torch.abs(h.cpu() - est_chan)**2, 1)
        print('NMSE MpNet', torch.mean(mp_error_norm_test / norm_test_dataset))

        # Save data
        np.savez(path_init / data_pred_mpnet / f'batch_{batch_idx}.npz', **est_c)







        ########################################### Mp nominal Estimation ##############################################
        out_chans = torch.zeros_like(h_noisy, dtype=torch.complex128).to('cuda')

        for i in range(h_noisy_M.shape[0]):
            dict_nominal_M = torch.matmul(torch.conj(M.mT), dict_nominal)

            for j in range(dict_nominal_M.shape[0]):
                norm = torch.tensor(np.sqrt(np.sum(np.abs(dict_nominal_M[j].detach().cpu().numpy())**2, 0)), device='cuda')
                dict_nominal_M[j] = (dict_nominal_M[j] / norm).type(torch.complex128).to('cuda')

            out_chans[i, :] = sparse_recovery.mp(h_noisy_M[i, :], h_noisy[i, :], dict_nominal_M[i, :], dict_nominal, k, False, SC2[i])

        # Save estimation
        est_c_mp = {'channels': out_chans.cpu().numpy()}

        # Cost function
        norm_test_dataset = torch.sum(torch.abs(h.cpu())**2, 1)
        mp_error_norm_test = torch.sum(torch.abs(h.cpu() - out_chans.cpu())**2, 1)
        print('NMSE MP NOMINAL', torch.mean(mp_error_norm_test / norm_test_dataset))

        np.savez(path_init / data_pred_MP_nominal / f'batch_{batch_idx}.npz', **est_c_mp)
        
        
        ######################################### Mp real Estimation ##################################################
        out_chans = torch.zeros_like(h_noisy, dtype=torch.complex128).to('cuda')

        for i in range(h_noisy_M.shape[0]):
            dict_real_M = torch.matmul(torch.conj(M.mT), dict_real)

            for j in range(dict_real_M.shape[0]):
                norm = torch.tensor(np.sqrt(np.sum(np.abs(dict_real_M[j].detach().cpu().numpy())**2, 0)), device='cuda')
                dict_real_M[j] = (dict_real_M[j] / norm).type(torch.complex128)

            out_chans[i, :] = sparse_recovery.mp(h_noisy_M[i, :], h_noisy[i, :], dict_real_M[i, :], dict_real, k, False, SC2[i])

        # Save estimation
        est_c_mp = {'channels': out_chans.cpu().numpy()}

        # Cost function
        norm_test_dataset = torch.sum(torch.abs(h.cpu())**2, 1)
        mp_error_norm_test = torch.sum(torch.abs(h.cpu() - out_chans.cpu())**2, 1)
        print('NMSE MP REAL', torch.mean(mp_error_norm_test / norm_test_dataset))

        np.savez(path_init / data_pred_MP_real / f'batch_{batch_idx}.npz', **est_c_mp)

        
        
        
        ######################################### LS estimation####################################################
        #Mh = torch.conj(torch.transpose(M, 1, 2))
        #est_c_ls = {'channels': (torch.linalg.inv(M @ Mh) @ M @ h_noisy_M.unsqueeze(2)).squeeze(2).cpu().numpy()}

        # Cost function
        #norm_test_dataset = torch.sum(torch.abs(h.cpu())**2, 1)
        #mp_error_norm_test = torch.sum(torch.abs(h.cpu() - est_c_ls['channels'])**2, 1)
        #print('ls cost', torch.mean(mp_error_norm_test / norm_test_dataset))

        #np.savez(path_init / data_pred_LS / f'batch_{batch_idx}.npz', **est_c_ls)
        '''