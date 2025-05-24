import numpy as np
import h5py
import math

time_step_data_to_use    = np.array([0, 1, 2])                 
pressure_location_to_use = -1                           

saturation_measurement_resolution_error = 0.1
saturation_surrogate_error = 0.055  # Estimated at the observation locations

pressure_measurement_resolution_error = 0.1
pressure_surrogate_error = 0.290    # Estimated at the observation locations

saturation_std = pow((saturation_measurement_resolution_error ** 2 + saturation_surrogate_error ** 2), 0.5)         
pressure_std   = pow((pressure_measurement_resolution_error ** 2 + pressure_surrogate_error ** 2), 0.5)

def load_data(data_path, array_name_list):
    
    hf_r = h5py.File(data_path, 'r')
    result = []
    for name in array_name_list:
        result.append(np.array(hf_r.get(name)))
    hf_r.close()
    
    return result

def load_true_simulation_data():
    
    # Read noisy pressure
    true_pressure_dir = 'Pressure_Obeservation_1.h5'
    true_pressure = load_data(true_pressure_dir, ['pressure'])
    true_pressure = np.array(true_pressure)
    true_pressure = true_pressure.reshape(4, 10, 20)
    
    # Read noisy saturation
    true_saturation_dir = 'Saturation_Obeservation_1.h5'
    true_saturation = load_data(true_saturation_dir, ['saturation'])
    true_saturation = np.array(true_saturation)
    true_saturation = true_saturation.reshape(4, 10, 20)
      
    # Saturation at observation data
    true_saturation = true_saturation[:, time_step_data_to_use, :]
    
    # Pressure at observation data
    true_pressure = true_pressure[:, time_step_data_to_use, :]
    true_pressure = true_pressure[:, :, pressure_location_to_use]            
  
    return true_pressure, true_saturation, saturation_std, pressure_std

def compute_loglikelihood(predicted_pressure, predicted_saturation, true_pressure, true_saturation, pressure_std, saturation_std):
    
    log_likelihood = -0.5 * (np.sum((predicted_pressure - true_pressure) ** 2 / pressure_std ** 2, axis = (0, 1)) + np.sum((predicted_saturation - true_saturation) ** 2 / saturation_std ** 2, axis = (0, 1, 2)))
                       
    return log_likelihood
