from keras import backend as K
from keras.layers import Layer, Add, InputSpec
from keras.layers.core import Activation
from keras.layers.convolutional import Conv3D, UpSampling3D
from keras.layers import BatchNormalization, TimeDistributed, Reshape, RepeatVector
from keras import regularizers
from tensorflow.keras.layers import ZeroPadding3D
import tensorflow as tf
reg_weights = 0.00001

def conv_bn_relu(nb_filter, depth, height, width, stride = (1, 1, 1)):
    def conv_func(x):
        x = Conv3D(nb_filter, (depth, height, width), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    return conv_func

def time_conv_bn_relu(nb_filter, depth, height, width, stride = (1, 1, 1)):
    def conv_func(x):
        x = TimeDistributed(Conv3D(nb_filter, (depth, height, width), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation("relu"))(x)
        return x

    return conv_func

def res_conv(nb_filter, depth, height, width, stride=(1, 1, 1)):
    def _res_func(x):
        identity = x

        a = Conv3D(nb_filter, (depth, height, width), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights))(x)
        a = BatchNormalization()(a)
        a = Activation("relu")(a)
        a = Conv3D(nb_filter, (depth, height, width), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights))(a)
        y = BatchNormalization()(a)

        return Add()([identity, y])

    return _res_func

def time_res_conv(nb_filter, depth, height, width, stride=(1, 1, 1)):
    def _res_func(x):
        identity = x

        a = TimeDistributed(Conv3D(nb_filter, (depth, height, width), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights)))(x)
        a = TimeDistributed(BatchNormalization())(a)
        a = TimeDistributed(Activation("relu"))(a)
        a = TimeDistributed(Conv3D(nb_filter, (depth, height, width), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights)))(a)
        y =TimeDistributed(BatchNormalization())(a)

        return Add()([identity, y])

    return _res_func

def dconv_bn_nolinear(nb_filter, depth, height, width, stride=(2, 2, 2), activation="relu"):
    def _dconv_bn(x):
        print('size is ', stride)      
        x = UnSampling3D(size=stride)(x)
        x = ReflectionPadding3D((int(depth/2), int(height/2), int(width/2)))(x)      
        x = Conv3D(nb_filter, (depth, height, width), padding='valid', kernel_regularizer=regularizers.l2(reg_weights))(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    return _dconv_bn

def time_dconv_bn_nolinear(nb_filter, depth, height, width, stride=(2, 2, 2), activation="relu"):
    def _dconv_bn(x):
        x = TimeDistributed(UpSampling3D(size=stride))(x)
        x = TimeDistributed(ReflectionPadding3D((int(depth/2), int(height/2), int(width/2))))(x)
        x = TimeDistributed(Conv3D(nb_filter, (depth, height, width), padding='valid', kernel_regularizer=regularizers.l2(reg_weights)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation(activation))(x)
        return x

    return _dconv_bn

class UnPooling3D(UpSampling3D):
    def __init__(self, size=(2, 2, 2)):
        super(UnPooling3D, self).__init__(size)

    def call(self, x, mask=None):
        shapes = x.get_shape().as_list()
        d = self.size[0] * shapes[1]
        w = self.size[1] * shapes[2]
        h = self.size[2] * shapes[3]
        return tf.compat.v1.image.resize_nearest_neighbor(x, (d, w, h))
   
class ReflectionPadding3D(ZeroPadding3D):
    """Reflection-padding layer for 3D data (spatial or spatio-temporal).
    Args:
        padding (int, tuple): The pad-width to add in each dimension.
            If an int, the same symmetric padding is applied to height and
            width.
            If a tuple of 3 ints, interpreted as two different symmetric
            padding values for height and width:
            ``(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)``.
            If tuple of 3 tuples of 2 ints, interpreted as
            ``((left_dim1_pad, right_dim1_pad),
            (left_dim2_pad, right_dim2_pad),
            (left_dim3_pad, right_dim3_pad))``
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """
    def call(self, inputs):
        d_pad, w_pad, h_pad = self.padding
        if self.data_format == 'channels_first':
            pattern = [[0, 0], [0, 0], [d_pad[0], d_pad[1]],
                       [w_pad[0], w_pad[1]], [h_pad[0], h_pad[1]]]
        else:
            pattern = [[0, 0], [d_pad[0], d_pad[1]],
                       [w_pad[0], w_pad[1]], [h_pad[0], h_pad[1]], [0, 0]]
        return tf.pad(inputs, pattern, mode='REFLECT')
       
class RepeatConv(Layer):
    """Repeats the input n times.
    # Example
    ```python
        model = Sequential()
        model.add(Dense(32, input_dim=32))
        # now: model.output_shape == (None, 32)
        # note: `None` is the batch dimension
        model.add(RepeatVector(3))
        # now: model.output_shape == (None, 3, 32)
    ```
    # Arguments
        n: integer, repetition factor.
    # Input shape
        4D tensor of shape `(num_samples, w, h, c)`.
    # Output shape
        5D tensor of shape `(num_samples, n, w, h, c)`.
    """

    def __init__(self, n, **kwargs):
        super(RepeatConv, self).__init__(**kwargs)
        self.n = n
        self.input_spec = InputSpec(ndim=5)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n, input_shape[1], input_shape[2], input_shape[3], input_shape[4])

    def call(self, inputs):
           
        x = K.expand_dims(inputs, 1)
        pattern = tf.stack([1, self.n, 1, 1, 1, 1])
        return K.tile(x, pattern)

    def get_config(self):
        config = {'n': self.n}
        base_config = super(RepeatConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))