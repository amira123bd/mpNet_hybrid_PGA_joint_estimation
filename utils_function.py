import numpy as np
import time
import math
import tensorflow as tf
import torch# Import Sionna RT components
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from sionna.channel import cir_to_ofdm_channel

from pathlib import Path


def init_scene(BS_position,f0):
    '''
    generate scene that contain  BS position 
    
    input: tx_pos => corresponds to the user's positions
           rx_pos => The BS position 
    
    output: scene
            nominal antenna positions
            real antenna positions
    
    '''
    
    scene = load_scene(sionna.rt.scene.etoile)
    nb_BS_antenna=64
    #nb_BS_antenna=256

    # change frequency
    scene.frequency=f0
    lambda_ = 0.010706874

    
    # Configuration of transmitters
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")
    
    # Configuration of recievers 
    # the antenna by default is located in the y-z plane 
    # This config creates antennas that are aligned over y axis 
    scene.rx_array = PlanarArray(num_rows=1, num_cols=nb_BS_antenna,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")
    
    
    #generate real and nominal positions
    nominal_ant_positions = tf.Variable(scene.rx_array.positions)
    scene.rx_array.positions=tf.Variable(scene.rx_array.positions)
    #print(scene.rx_array.positions)
    scene.rx_array.positions[:,1].assign(scene.rx_array.positions[:,1]+ 0.1*lambda_*np.random.randn(nb_BS_antenna))
    real_ant_positions = scene.rx_array.positions
    #print(scene.rx_array.positions)
    
    
    # Add BS to the scene
    rx = Receiver("rx", position=BS_position, orientation=[0,0,0])
    scene.add(rx)
    

        
    return scene,nominal_ant_positions,real_ant_positions 


        
def generate_M(phases):
    
    M_test = []

    for phase in phases:
        
        M = torch.tensor(np.exp(1j * phase), dtype=torch.complex128)
        
        #M_test.append(M/torch.linalg.norm(M,axis=0)[None,:])
        
        M_test.append(M)
        

    M_t = torch.stack(M_test)



    return M_t


def preprocess(data_path,batch_size):
    #Preprocessing:Dataset normalization 
    
    batch_idx=1
    path_init = Path.cwd()

    file_name=f"batch_0.npz" 
    data = np.load(path_init/data_path/file_name)
    channel_0= data['channel_train']

    Dataset=channel_0


    while(batch_idx<batch_size):
        
        #get channels       
        file_name=f"batch_{batch_idx}.npz" 
        data = np.load(path_init/data_path/file_name)
        channel_train= data['channel_train']

        Dataset=np.vstack((Dataset,channel_train))      
        batch_idx+=1
        



    norm_channels=np.linalg.norm(Dataset,axis=1)
    norm_max=np.max(norm_channels)
    norm_min=np.min(norm_channels)
    norm_factor=norm_max - norm_min
    
    return norm_factor