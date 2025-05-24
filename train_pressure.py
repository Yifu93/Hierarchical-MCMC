import numpy as np
import h5py
import unet_uae_filter as vae_util
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf 
import pickle

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

def load_data(data_path, array_name_list):
    hf_r = h5py.File(data_path, 'r')
    result = []
    for name in array_name_list:
        result.append(np.array(hf_r.get(name)))
    hf_r.close()
    return result

data_dir = '/Dataset'
# load training data
simulation_data = os.path.join(data_dir, 'P_train_10_steps.h5')
p_t = load_data(simulation_data, ['pressure'])
p_t = np.array(p_t)
p_t = p_t[0,...]

permeability = os.path.join(data_dir, 'logk_train.h5')
k = load_data(permeability, ['logk'])
k = np.array(k)
k = k.reshape(2000, 20, 80, 80)

k_max = np.max(k)
k     = k / k_max

print('k max is ', np.max(k))
print('k min is ', np.min(k))

permeability = os.path.join(data_dir, 'logk_validation.h5')
k_v = load_data(permeability, ['logk'])
k_v = np.array(k_v)
k_v = k_v.reshape(100, 20, 80, 80)
k_v = k_v / k_max

print('k_v max is ', np.max(k_v))
print('k_v min is ', np.min(k_v))

porosity = os.path.join(data_dir, 'phi_train.h5')
phi = load_data(porosity, ['phi'])
phi = np.array(phi)
phi = phi.reshape(2000, 20, 80, 80)
phi_max = np.max(phi)
phi = phi / phi_max

print('porosity max is ', np.max(phi))
print('porosity min is ', np.min(phi))

porosity = os.path.join(data_dir, 'phi_validation.h5')
phi_v = load_data(porosity, ['phi'])
phi_v = np.array(phi_v)
phi_v = phi_v.reshape(100, 20, 80, 80)
phi_v = phi_v / phi_max

Meta_Parameters_path = os.path.join(data_dir, 'Meta_Parameters_train.npy')
Meta_Parameters = np.load(Meta_Parameters_path)

log_kvkh  = Meta_Parameters[2, :]

kvkh = np.zeros(((2000, 20, 80, 80, 1)))

for i in range(2000):
    kvkh[i, ...] = np.power(10, log_kvkh[i])
    
print('kvkh min is ', np.min(kvkh))
print('kvkh max is ', np.max(kvkh))

Meta_Parameters_path = os.path.join(data_dir, 'Meta_Parameters_validation.npy')
Meta_Parameters_v = np.load(Meta_Parameters_path)

log_kvkh_v  = Meta_Parameters_v[2, :]

kvkh_v = np.zeros(((100, 20, 80, 80, 1)))

for i in range(100):
    kvkh_v[i, ...] = np.power(10, log_kvkh_v[i])
    
print('kvkh_v min is ', np.min(kvkh_v))
print('kvkh_v max is ', np.max(kvkh_v))

print('p_t shape is ', p_t.shape)
print('k shape is ', k.shape)
print('phi shape is ', phi.shape)

# load testing data
simulation_data = os.path.join(data_dir, 'P_validation_10_steps.h5')
p_v = load_data(simulation_data, ['pressure'])
p_v = np.array(p_v)
p_v = p_v[0,...]

print('p_v shape is ', p_v.shape)

print('p_t max is ', np.max(p_t))
print('p_t min is ', np.min(p_t))

print('p_v max is ', np.max(p_v))
print('p_v min is ', np.min(p_v))

depth = 10
nr = k.shape[0]
train_nr = 2000
test_nr  = 100

initial_data = os.path.join(data_dir, 'P_initial.h5')
p_initial = load_data(initial_data, ['pressure'])
p_initial = np.array(p_initial)
p_initial = p_initial[0,...]

p_t = p_t - p_initial
print('max p is ', np.max(p_t), ', min p is ', np.min(p_t))

max_p = np.max(p_t, axis=(0,2,3,4), keepdims = True)
min_p = np.min(p_t, axis=(0,2,3,4), keepdims = True)

print('max_p shape is ', max_p.shape)
print('min_p shape is ', min_p.shape)

epsilon = 1e-6
p_t = (p_t - min_p)/(max_p - min_p + 1e-6)

p_v = p_v - p_initial
p_v = (p_v - min_p)/(max_p - min_p + 1e-6)

print('max p train is ', np.max(p_t), ', min p train is ', np.min(p_t))
print('max p validation is ', np.max(p_v), ', min p validation is ', np.min(p_v))

step_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
k     = k[:, :, :, :, None]
phi   = phi[:, :, :, :, None]
k_v   = k_v[:, :, :, :, None]
phi_v = phi_v[:, :, :, :, None]

train_x = np.concatenate([k, kvkh, phi], axis = -1)
train_y = p_t[:, step_index, :, :, :]

test_x  = np.concatenate([k_v, kvkh_v, phi_v], axis = -1)
test_y  = p_v[:, step_index, :, :, :]

train_y = train_y[:, :, :, :, :, None]
test_y  = test_y[:, :, :, :, :, None]

print('train_x shape is ', train_x.shape)
print('train_y shape is ', train_y.shape)
print('test_x shape is ', test_x.shape)
print('test_y shape is ', test_y.shape)

input_shape = (20, 80, 80, 3)
vae_model,_ = vae_util.create_vae(input_shape, depth)
vae_model.summary(line_length = 150)

output_dir = 'saved_models/'
epochs = 300
train_nr = train_x.shape[0]
test_nr  = 10
batch_size = 4
num_batch  = int(train_nr/batch_size) 
    
def vae_loss(x, t_decoded):
    '''Total loss for the plain UAE'''
    return K.mean(reconstruction_loss(x, t_decoded))

def reconstruction_loss(x, t_decoded):
    '''Reconstruction loss for the plain UAE'''
    return K.sum((K.batch_flatten(x) - K.batch_flatten(t_decoded)) ** 2, axis=-1)
    
def true_value(x):
    '''Reconstruction loss for the plain UAE'''
    return K.sum((K.batch_flatten(x)) ** 2, axis=-1)

def relative_error(x, t_decoded):
    return K.mean(reconstruction_loss(x, t_decoded) / (true_value(x)))

opt = Adam(lr = 3e-4)
vae_model.compile(loss = relative_error, optimizer = opt, metrics = [vae_loss, relative_error])

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
lrScheduler = ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 10, cooldown = 1, verbose = 1, min_lr = 1e-7)
filePath = 'saved_models/saved-model-10steps-lr3e-4-pressure-detrend-hd-0-filter_16_32_32_64-mse-{epoch:03d}-{val_loss:.2f}.h5'
checkPoint = ModelCheckpoint(filePath, monitor = 'val_loss', verbose = 1, save_best_only = False, \
                             save_weights_only = True, mode = 'auto', period = 20)

callbacks_list = [lrScheduler, checkPoint]

history = vae_model.fit(train_x, train_y, batch_size = batch_size, epochs = epochs, \
                        verbose = 2, validation_data = (test_x, test_y), callbacks = callbacks_list)
    
with open('HISTORY-pressure-mse-hd500-filter_8_16_32_32.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
