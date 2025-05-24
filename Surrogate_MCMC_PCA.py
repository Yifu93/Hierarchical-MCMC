import numpy as np
import h5py
import unet_uae_filter_16_32_32_64 as vae_util
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
from tensorflow import keras
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, concatenate, TimeDistributed, RepeatVector, ConvLSTM3D
from keras.models import Model
from keras.optimizers import Adam
from Markov_chain import *
import random
from pca import PCA

# check tensorflow version
print("tensorflow version:", tf.__version__)
# check available gpu
gpus =  tf.config.list_physical_devices('GPU')
print("available gpus:", gpus)
# limit the gpu usage, prevent it from allocating all gpu memory for a simple model
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# check number of cpus available
print("available cpus:", os.cpu_count())

def load_data(data_path, array_name_list):
    hf_r = h5py.File(data_path, 'r')
    result = []
    for name in array_name_list:
        result.append(np.array(hf_r.get(name)))
    hf_r.close()
    return result
    
nx, ny, nz = 80, 80, 20
k_max = np.log(1e4)
phi_max = 0.4
nr  = 1000
dim = 900

# ======================================================================================================================
# ======================================================== Surrogate ===================================================
# ======================================================================================================================
obs_loc_x = [21, 22, 61, 61]
obs_loc_y = [22, 61, 22, 62]

time_step_data_to_use    = np.array([0, 1, 2])                 
pressure_location_to_use = -1                           

depth = 10                                             
input_shape=(20, 80, 80, 3)
batch_size = 1 

vae_model_p,_ = vae_util.create_vae(input_shape, depth) # load pressure surrogate model
vae_model_p.summary(line_length=150)

output_dir_p = '/saved_models_0/'
vae_model_p.load_weights(output_dir_p + 'saved-model-10steps-lr3e-4-pressure-detrend-hd-0-filter_16_32_32_64-mse-300-41.44.h5')

vae_model_s,_ = vae_util.create_vae(input_shape, depth) # load saturation surrogate model
vae_model_s.summary(line_length=150)

output_dir_s = '/saved_models_0/'
vae_model_s.load_weights(output_dir_s + 'saved-model-10-steps-lr3e-4-saturation-hd-0-filter_16_32_32_64-mse-500-711.30.h5') 

P_initial = load_data('P_initial.h5', ['pressure'])
P_initial = np.array(P_initial)
P_initial = P_initial[0,...]

max_p = load_data('max_p.h5', ['pressure'])
max_p = np.array(max_p)
max_p = max_p[0,...]

min_p = load_data('min_p.h5', ['pressure'])
min_p = np.array(min_p)
min_p = min_p[0,...]

# ======================================================================================================================
# ======================================================== PCA =========================================================
# ======================================================================================================================
multi_Gaussian = load_data('logk_1.h5', ['logk'])
multi_Gaussian = np.array(multi_Gaussian)
multi_Gaussian = multi_Gaussian[0, :, :, :, :]

pca_model = PCA(nc = nx * ny * nz, nr = nr, l = 1000)
pca_model.construct_pca(multi_Gaussian.reshape((nr, nx * ny * nz)).T)

def compute_surrogate_prediction(logk_mean, logk_std, a, b, log_kvkh, m_pca): 
       
    base         = m_pca.reshape((batch_size, nz, ny, nx, 1))    
    permeability = base * logk_std + logk_mean
    porosity     = permeability * a + b
    
    permeability = np.exp(permeability);
    permeability[permeability < 1e-3] = 1e-3  # mD
    permeability[permeability > 1e4]  = 1e4   # mD
    permeability = np.log(permeability);
    permeability = permeability / k_max 
    
    porosity[porosity < 0.05] = 0.05
    porosity[porosity > 0.4]  = 0.4   
    porosity = porosity / phi_max
    
    k_ratio = np.power(10, log_kvkh)
    kvkh    = np.zeros(((batch_size, 20, 80, 80, 1)))
    kvkh[0, ...] = k_ratio
    
    input_x = np.concatenate([permeability, kvkh, porosity], axis = -1)
    
    pressure_predictions = vae_model_p.predict(input_x)    # pressure surrogate prediction
    saturation_predictions = vae_model_s.predict(input_x)  # saturation surrogate prediction
    
    pressure_predictions = pressure_predictions[:, :, :, :, :, 0]
    pressure_predictions = pressure_predictions * (max_p - min_p + 1e-6) + min_p
    pressure_predictions = pressure_predictions + P_initial
    
    saturation_predictions[saturation_predictions < 0.025] = 0.0
    saturation_predictions[saturation_predictions > 0.8]   = 0.8
    
    observation_saturation = np.zeros((4, 10, 20))          # get saturation predictions at observation wells
    observation_pressure = np.zeros((4, 10, 20))            # get pressure predictions at observation wells
    
    for i in range(len(obs_loc_x)):
        y, x = obs_loc_y[i], obs_loc_x[i]
        observation_saturation[i, :, :] = saturation_predictions[:, :, :, y, x, 0]
        observation_pressure[i, :, :] = pressure_predictions[:, :, :, y, x]
        
    observation_pressure_to_use = observation_pressure[:, time_step_data_to_use, :]
    observation_pressure_to_use = observation_pressure_to_use[:, :, pressure_location_to_use]
    observation_pressure_to_use = observation_pressure_to_use / 1e6    # MPa
    
    observation_saturation_to_use = observation_saturation[:, time_step_data_to_use, :]
    
    return observation_saturation_to_use, observation_pressure_to_use
    
# ======================================================================================================================
# ======================================================== MCMC ========================================================
# ======================================================================================================================
max_iter = 5000000               # maximum number of samples (for each chain)
m        = 1                     # number of chains
n_theta  = 5                     # number of inversion parameters
theta_labels = ["mean(logk)", "std(logk)", "d", "e", "kv/kh"]
accept_count = 0

# Meta parameters: mean(logk), std(logk), d, e, log10(kv/kh)
thetaMin = np.array([1.5, 1.0, 0.02, 0.05, -2.0])         # permissible range: min
thetaMax = np.array([4.0, 2.5, 0.04, 0.10,  0.0])         # permissible range: max

proposal_std = (thetaMax - thetaMin) / 16.0               # standard deviation for each parameter

beta = 0.15                                  

true_pressure, true_saturation, saturation_std, pressure_std = load_true_simulation_data()
isCheckConvergence = True
iter_num = 0

def propose(proposal_mean):
    
    proposed_parameters = np.zeros_like(proposal_mean)
    
    for i in range(n_theta):
        
        proposed_parameters[i] = np.random.normal(loc=proposal_mean[i], scale=proposal_std[i])
        
        while (proposed_parameters[i] < thetaMin[i]) or (proposed_parameters[i] > thetaMax[i]):
            
            proposed_parameters[i] = np.random.normal(loc=proposal_mean[i], scale=proposal_std[i])
    
    return proposed_parameters

for chain_num in range(m):

    print('---------------- Running chain {:} of {:}. ----------------'.format(chain_num + 1, m))

    xi = np.random.normal(0, 1, dim) 
    
    m_pca = pca_model.generate_pca_realization(xi, dim).T 
    
    theta = np.zeros_like(thetaMax)                           # initial guess: uniformly distributed between allowable range  
     
    for i in range(n_theta):
        
        theta[i] = np.random.uniform(thetaMin[i], thetaMax[i])
          
    print ('Iteration number is: ', iter_num, flush = True)
    
    observation_saturation, observation_pressure_to_use = compute_surrogate_prediction(theta[0], theta[1], theta[2], theta[3], theta[4], m_pca)   
    loglikelihood = compute_loglikelihood(observation_pressure_to_use, observation_saturation, true_pressure, true_saturation, pressure_std, saturation_std)
    
    likelihood_chain = np.zeros(max_iter)                  # likelihood at each iteration
    likelihood_chain[0] = loglikelihood

    theta_chain = np.zeros([max_iter, n_theta])       # inversion parameters at each iteration for current chain
    theta_chain[0, :] = theta
    
    pca_chain = np.zeros([max_iter, dim])
    pca_chain[0, :] = xi
    
    is_accept = [True]
    accept_count += 1
    convergence_frequency = 250                 # frequency at which convergence is checked
                                 
    PDF_mean_logk_previous, PDF_std_logk_previous, PDF_a_previous, PDF_b_previous, PDF_kvkh_previous = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)    # PDF values for continous variables

    n_proposed = np.zeros(2, dtype=int)         # number of times update is proposed for each parameter
    n_accepted = np.zeros(2, dtype=int)         # number of times update is accepted for each parameter
    
    # ========================= START MCMC LOOP ============================================================================
    for iter_num in range(1, max_iter):
        
        print ('Iteration number is: ', iter_num, flush = True)      
        
        previous_xi = xi
        previous_theta = theta
        previous_m_pca = m_pca
        prev_loglikelihood = loglikelihood
        
        # Two-stage hierarchical MCMC. 
        # Stage 1: propose a new xi, which is PCA variables 
        n_proposed[0] += 1
        
        xk = np.random.normal(0, 1, dim) 
        xi = np.power((1 - beta **2), 1/2) * xi + beta * xk
        
        m_pca = pca_model.generate_pca_realization(xi, dim).T  # Construct PCA realizations
        
        observation_saturation, observation_pressure = compute_surrogate_prediction(theta[0], theta[1], theta[2], theta[3], theta[4], m_pca)
        
        loglikelihood = compute_loglikelihood(observation_pressure, observation_saturation, true_pressure, true_saturation, pressure_std, saturation_std)
        
        accept_ratio = loglikelihood - prev_loglikelihood
        
        # accept/reject
        if np.log(np.random.random()) <= accept_ratio:                   
            
            print('Accept PCA variables.', flush = True)  # update Markov chain
            n_accepted[0] += 1
            pca_chain[iter_num, :] = xi
            prev_loglikelihood = loglikelihood
            
            pca_chain_save = pca_chain[0 : iter_num + 1, ...]            
            np.save('pca_chain_pca_1', pca_chain_save)       
                      
        else:
            
            xi = previous_xi          
            pca_chain[iter_num, :] = xi
            m_pca = previous_m_pca
            
        # Stage 2: propose a new metaparameters             
        proposal_mean = theta
        
        theta = propose(proposal_mean)
                  
        n_proposed[1] += 1    
         
        observation_saturation, observation_pressure = compute_surrogate_prediction(theta[0], theta[1], theta[2], theta[3], theta[4], m_pca)        
        
        loglikelihood = compute_loglikelihood(observation_pressure, observation_saturation, true_pressure, true_saturation, pressure_std, saturation_std)
        
        accept_ratio = loglikelihood - prev_loglikelihood
        
        # accept/reject
        if np.log(np.random.random()) <= accept_ratio:          
                
            n_accepted[1] += 1           
            is_accept.append(True)          
            accept_count += 1 
            
            # update Markov chain
            print('Accept Meta Parameters:', theta, flush = True)
            theta_chain[iter_num, :] = theta
            likelihood_chain[iter_num] = loglikelihood
            
            theta_chain_save = theta_chain[0 : iter_num + 1, :]
            likelihood_chain_save = likelihood_chain[0 : iter_num + 1]
            
            np.save('theta_chain_pca',      theta_chain_save)
            np.save('likelihood_chain_pca', likelihood_chain_save)
            np.save('is_accept_pca',        is_accept)
            np.save('n_proposed_pca',       n_proposed)
            np.save('n_accepted_pca',       n_accepted)          
            isCheckConvergence = True
                      
        else:
            
            theta = previous_theta
            loglikelihood = prev_loglikelihood
            is_accept.append(False)
            
            # update Markov chain
            theta_chain[iter_num, :] = theta
            likelihood_chain[iter_num] = loglikelihood

        # check for convergence
        if (accept_count % convergence_frequency == 0 and isCheckConvergence == True):   
            
            print('Check convergence: ', flush = True)
            isCheckConvergence = False
            
            thetaTemp    = theta_chain[0 : iter_num + 1, :]       
            theta_accept = thetaTemp[is_accept, :]
            
            mean_logk_accept = theta_accept[:, 0]
            mean_logk_accept = np.unique(mean_logk_accept)
            std_logk_accept  = theta_accept[:, 1]
            std_logk_accept  = np.unique(std_logk_accept)
            a_accept         = theta_accept[:, 2]
            a_accept         = np.unique(a_accept)
            b_accept         = theta_accept[:, 3]
            b_accept         = np.unique(b_accept)
            kvkh_accept      = theta_accept[:, 4]
            kvkh_accept      = np.unique(kvkh_accept)
            
            # Check convergence for continuous variables
            PDF_mean_logk, _ = np.histogram(mean_logk_accept, bins = 10, range=((thetaMin[0], thetaMax[0])), density = True)
            PDF_std_logk, _  = np.histogram(std_logk_accept, bins = 10, range=((thetaMin[1], thetaMax[1])), density = True)
            PDF_a, _         = np.histogram(a_accept, bins = 10, range=((thetaMin[2], thetaMax[2])), density = True)
            PDF_b, _         = np.histogram(b_accept, bins = 10, range=((thetaMin[3], thetaMax[3])), density = True)
            PDF_kvkh, _      = np.histogram(kvkh_accept, bins = 10, range=((thetaMin[4], thetaMax[4])), density = True)

            mean_difference_mean_logk, mean_difference_std_logk, mean_difference_a, mean_difference_b, mean_difference_kvkh = 0, 0, 0, 0, 0
            count_mean_logk, count_std_logk, count_a, count_b, count_kvkh = 10, 10, 10, 10, 10
            
            for i in range(10):  
                
                if PDF_mean_logk_previous[i] != 0:
                    mean_difference_mean_logk += np.abs(PDF_mean_logk[i] - PDF_mean_logk_previous[i]) / PDF_mean_logk_previous[i]
                    
                if PDF_std_logk_previous[i] != 0:
                    mean_difference_std_logk  += np.abs(PDF_std_logk[i] - PDF_std_logk_previous[i]) / PDF_std_logk_previous[i]
                    
                if PDF_a_previous[i] != 0:
                    mean_difference_a         += np.abs(PDF_a[i] - PDF_a_previous[i]) / PDF_a_previous[i]
                    
                if PDF_b_previous[i] != 0:
                    mean_difference_b         += np.abs(PDF_b[i] - PDF_b_previous[i]) / PDF_b_previous[i]
                    
                if PDF_kvkh_previous[i] != 0:
                    mean_difference_kvkh      += np.abs(PDF_kvkh[i] - PDF_kvkh_previous[i]) / PDF_kvkh_previous[i]
              
            if (accept_count > convergence_frequency):
                mean_difference_mean_logk = mean_difference_mean_logk / count_mean_logk
                mean_difference_std_logk  = mean_difference_std_logk / count_std_logk
                mean_difference_a         = mean_difference_a / count_a
                mean_difference_b         = mean_difference_b / count_b
                mean_difference_kvkh      = mean_difference_kvkh / count_kvkh
            
            difference = [mean_difference_mean_logk, mean_difference_std_logk, mean_difference_a, mean_difference_b, mean_difference_kvkh]            
            print('Convergence check results: ', difference, flush = True)            
            total_difference = np.mean(difference)        

            eps = 0.01
            if (total_difference < eps) and (accept_count > convergence_frequency):
                print('MCMC results are convergenced!', flush = True)
                theta_chain      = theta_chain[0 : iter_num + 1, :]
                likelihood_chain = likelihood_chain[0 : iter_num + 1]
                pca_chain        = pca_chain[0 : iter_num + 1, ...]
                break

            PDF_mean_logk_previous, PDF_std_logk_previous, PDF_a_previous, PDF_b_previous, PDF_kvkh_previous = PDF_mean_logk, PDF_std_logk, PDF_a, PDF_b, PDF_kvkh
            
            np.save('PDF_mean_logk_pca', PDF_mean_logk_previous)
            np.save('PDF_std_logk_pca',  PDF_std_logk_previous)
            np.save('PDF_a_pca',         PDF_a_previous)
            np.save('PDF_b_pca',         PDF_b_previous)
            np.save('PDF_kvkh_pca',      PDF_kvkh_previous)
            
print("\n============== MCMC results ==============")
print("\t\t\t\t\t\tMean \t St. dev.")
for i in range(n_theta):
    print("{:15s} \t {:7.3f} \t {:7.3f}".format(theta_labels[i], np.mean(theta_chain[:, i]), np.std(theta_chain[:, i])))
    
np.save('theta_chain_pca', theta_chain)
np.save('pca_chain_pca',   pca_chain)
np.save('is_accept_pca',   is_accept)
np.save('n_proposed_pca',  n_proposed)
np.save('n_accepted_pca',  n_accepted)
np.save('likelihood_chain_pca', likelihood_chain)
