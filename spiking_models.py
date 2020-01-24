import datetime
from collections import OrderedDict
from collections import namedtuple

import numpy as np
import numpy.random as rd
import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell, DropoutRNNCellMixin
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.framework import function
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops.variables import Variable
from tensorflow.python.keras.utils import tf_utils
# from rewiring_tools import weight_sampler

Cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell


def einsum_bi_ijk_to_bjk(a,b):
    batch_size = tf.shape(a)[0]
    shp_a = a.get_shape()
    shp_b = b.get_shape()

    b_ = tf.reshape(b,(int(shp_b[0]), int(shp_b[1]) * int(shp_b[2])))
    ab_ = tf.matmul(a,b_)
    ab = tf.reshape(ab_,(batch_size,int(shp_b[1]),int(shp_b[2])))
    return ab


def tf_roll(buffer, new_last_element=None, axis=0):
    with tf.name_scope('roll'):
        shp = buffer.get_shape()
        l_shp = len(shp)

        # Permute the index to roll over the right index
        perm = np.concatenate([[axis],np.arange(axis),np.arange(start=axis+1,stop=l_shp)])
        buffer = tf.transpose(buffer, perm=perm)

        # Add an element at the end of the buffer if requested, otherwise, add zero
        if new_last_element is None:
            shp = tf.shape(buffer)
            new_last_element = tf.zeros(shape=shp[1:], dtype=buffer.dtype)
        new_last_element = tf.expand_dims(new_last_element, axis=0)
        new_buffer = tf.concat([buffer[1:], new_last_element], axis=0, name='rolled')

        # Revert the index permutation
        inv_perm = np.argsort(perm)
        new_buffer = tf.transpose(new_buffer,perm=inv_perm)

        new_buffer = tf.identity(new_buffer,name='Roll')
        #new_buffer.set_shape(shp)
    return new_buffer


def map_to_named_tuple(S, f):
    state_dict = S._asdict()
    new_state_dict = OrderedDict({})
    for k, v in state_dict.items():
        new_state_dict[k] = f(v)

    new_named_tuple = S.__class__(**new_state_dict)
    return new_named_tuple

def placeholder_container_for_rnn_state(cell_state_size, dtype, batch_size, name='TupleStateHolder'):
    with tf.name_scope(name):
        default_dict = cell_state_size._asdict()
        placeholder_dict = OrderedDict({})
        for k, v in default_dict.items():
            if np.shape(v) == ():
                v = [v]
            shape = np.concatenate([[batch_size], v])
            placeholder_dict[k] = tf.placeholder(shape=shape, dtype=dtype, name=k)

        placeholder_tuple = cell_state_size.__class__(**placeholder_dict)
        return placeholder_tuple


def placeholder_container_from_example(state_example, name='TupleStateHolder'):
    with tf.name_scope(name):
        default_dict = state_example._asdict()
        placeholder_dict = OrderedDict({})
        for k, v in default_dict.items():
            placeholder_dict[k] = tf.placeholder(shape=v.shape, dtype=v.dtype, name=k)

        placeholder_tuple = state_example.__class__(**placeholder_dict)
        return placeholder_tuple

def feed_dict_with_placeholder_container(dict_to_update, state_holder, state_value, batch_selection=None):
    if state_value is None:
        return dict_to_update

    assert state_holder.__class__ == state_value.__class__, 'Should have the same class, got {} and {}'.format(
        state_holder.__class__, state_value.__class__)

    for k, v in state_value._asdict().items():
        if batch_selection is None:
            dict_to_update.update({state_holder._asdict()[k]: v})
        else:
            dict_to_update.update({state_holder._asdict()[k]: v[batch_selection]})

    return dict_to_update


#################################
# Rewirite the Spike function without hack
#################################

@function.Defun()
def SpikeFunctionGrad(v_scaled, dampening_factor, grad):
    dE_dz = grad
    dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)
    dz_dv_scaled *= dampening_factor

    dE_dv_scaled = dE_dz * dz_dv_scaled

    return [dE_dv_scaled,
            tf.zeros_like(dampening_factor)]


@function.Defun(grad_func=SpikeFunctionGrad)
def SpikeFunction(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)
    return tf.identity(z_, name="SpikeFunction")

def weight_matrix_with_delay_dimension(w, d, n_delay):
    """
    Generate the tensor of shape n_in x n_out x n_delay that represents the synaptic weights with the right delays.

    :param w: synaptic weight value, float tensor of shape (n_in x n_out)
    :param d: delay number, int tensor of shape (n_in x n_out)
    :param n_delay: number of possible delays
    :return:
    """
    with tf.name_scope('WeightDelayer'):
        w_d_list = []
        for kd in range(n_delay):
            mask = tf.equal(d,kd)
            w_d = tf.where(condition=mask, x=w, y=tf.zeros_like(w))
            w_d_list.append(w_d)

        delay_axis = len(d.shape)
        WD = tf.stack(w_d_list, axis=delay_axis)

    return WD


# PSP on output layer
def exp_convolve(tensor, decay, init=None):  # tensor shape (trial, time, neuron)
    with tf.name_scope('ExpConvolve'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float64]

        tensor_time_major = tf.transpose(tensor, perm=[1, 0, 2])
        if init is not None:
            assert str(init.get_shape()) == str(tensor_time_major[0].get_shape())  # must be batch x neurons
            initializer = init
        else:
            initializer = tf.zeros_like(tensor_time_major[0])

        filtered_tensor = tf.scan(lambda a, x: a * decay + (1 - decay) * x, tensor_time_major, initializer=initializer)
        filtered_tensor = tf.transpose(filtered_tensor, perm=[1, 0, 2])
    return filtered_tensor


LIFStateTuple = namedtuple('LIFStateTuple', ('v', 'z', 'i_future_buffer', 'z_buffer'))


def tf_cell_to_savable_dict(cell, sess, supplement={}):
    """
    Usefull function to return a python/numpy object from of of the tensorflow cell object defined here.
    The idea is simply that varaibles and Tensors given as attributes of the object with be replaced by there numpy value evaluated on the current tensorflow session.

    :param cell: tensorflow cell object
    :param sess: tensorflow session
    :param supplement: some possible
    :return:
    """

    dict_to_save = {}
    dict_to_save['cell_type'] = str(cell.__class__)
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dict_to_save['time_stamp'] = time_stamp

    dict_to_save.update(supplement)

    for k, v in cell.__dict__.items():
        if k == 'self':
            pass
        elif type(v) in [Variable, Tensor]:
            dict_to_save[k] = sess.run(v)
        elif type(v) in [bool, int, float, np.int64, np.ndarray]:
            dict_to_save[k] = v
        else:
            print('WARNING: attribute of key {} and value {} has type {}, recoding it as string.'.format(k, v, type(v)))
            dict_to_save[k] = str(v)

    return dict_to_save
#
# class LIF(Cell):
#     def __init__(self, n_in, n_rec, tau=20., thr=0.03,
#                  dt=1., n_refractory=0, dtype=tf.float32, n_delay=1, rewiring_connectivity=-1,
#                  in_neuron_sign=None, rec_neuron_sign=None,
#                  dampening_factor=0.3,
#                  injected_noise_current=0.,
#                  V0=1.):
#         """
#         Tensorflow cell object that simulates a LIF neuron with an approximation of the spike derivatives.
#
#         :param n_in: number of input neurons
#         :param n_rec: number of recurrent neurons
#         :param tau: membrane time constant
#         :param thr: threshold voltage
#         :param dt: time step of the simulation
#         :param n_refractory: number of refractory time steps
#         :param dtype: data type of the cell tensors
#         :param n_delay: number of synaptic delay, the delay range goes from 1 to n_delay time steps
#         :param reset: method of resetting membrane potential after spike thr-> by fixed threshold amount, zero-> to zero
#         """
#
#         if np.isscalar(tau): tau = tf.ones(n_rec, dtype=dtype) * np.mean(tau)
#         if np.isscalar(thr): thr = tf.ones(n_rec, dtype=dtype) * np.mean(thr)
#         tau = tf.cast(tau,dtype=dtype)
#         dt = tf.cast(dt,dtype=dtype)
#
#         self.dampening_factor = dampening_factor
#
#         # Parameters
#         self.n_delay = n_delay
#         self.n_refractory = n_refractory
#
#         self.dt = dt
#         self.n_in = n_in
#         self.n_rec = n_rec
#         self.data_type = dtype
#
#         self._num_units = self.n_rec
#
#         self.tau = tau
#         self._decay = tf.exp(-dt / tau)
#         self.thr = thr
#
#         self.V0 = V0
#         self.injected_noise_current = injected_noise_current
#
#         self.rewiring_connectivity = rewiring_connectivity
#         self.in_neuron_sign = in_neuron_sign
#         self.rec_neuron_sign = rec_neuron_sign
#
#         with tf.variable_scope('InputWeights'):
#
#             # Input weights
#             if 0 < rewiring_connectivity < 1:
#                 self.w_in_val, self.w_in_sign, self.w_in_var, _ = weight_sampler(n_in, n_rec, rewiring_connectivity, neuron_sign=in_neuron_sign, w_scale=self.V0)
#             else:
#                 self.w_in_init = rd.randn(n_in, n_rec) / np.sqrt(n_in)
#                 self.w_in_var = tf.Variable(self.w_in_init * self.V0, dtype=dtype, name="InputWeight")
#                 self.w_in_val = self.w_in_var
#
#             self.w_in_delay = tf.Variable(rd.randint(self.n_delay, size=n_in * n_rec).reshape(n_in, n_rec),dtype=tf.int32,name="InDelays",trainable=False)
#             self.W_in = weight_matrix_with_delay_dimension(self.w_in_val, self.w_in_delay, self.n_delay)
#
#         with tf.variable_scope('RecWeights'):
#             if 0 < rewiring_connectivity < 1:
#                 self.w_rec_val, self.w_rec_sign, self.w_rec_var, _ = weight_sampler(n_rec, n_rec, rewiring_connectivity, neuron_sign=rec_neuron_sign, w_scale=self.V0)
#             else:
#                 if rec_neuron_sign is not None or in_neuron_sign is not None:
#                     raise NotImplementedError('Neuron sign requested but this is only implemented with rewiring')
#                 self.w_rec_var = tf.Variable(rd.randn(n_rec, n_rec) / np.sqrt(n_rec) * self.V0, dtype=dtype,
#                                              name='RecurrentWeight')
#                 self.w_rec_val = self.w_rec_var
#
#             recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
#
#             self.w_rec_val = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val),self.w_rec_val)  # Disconnect autotapse
#             self.w_rec_delay = tf.Variable(rd.randint(self.n_delay, size=n_rec * n_rec).reshape(n_rec, n_rec),dtype=tf.int32,name="RecDelays",trainable=False)
#             self.W_rec = weight_matrix_with_delay_dimension(self.w_rec_val, self.w_rec_delay, self.n_delay)
#
#     @property
#     def state_size(self):
#         return LIFStateTuple(v=self.n_rec,
#                              z=self.n_rec,
#                              i_future_buffer=(self.n_rec, self.n_delay),
#                              z_buffer=(self.n_rec, self.n_refractory))
#
#     @property
#     def output_size(self):
#         return self.n_rec
#
#     def zero_state(self, batch_size, dtype, n_rec=None):
#         if n_rec is None: n_rec = self.n_rec
#
#         v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#         z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#
#         i_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_delay), dtype=dtype)
#         z_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_refractory), dtype=dtype)
#
#         return LIFStateTuple(
#             v=v0,
#             z=z0,
#             i_future_buffer=i_buff0,
#             z_buffer=z_buff0
#         )
#
#     def __call__(self, inputs, state, scope=None, dtype=tf.float32):
#
#         i_future_buffer = state.i_future_buffer + einsum_bi_ijk_to_bjk(inputs, self.W_in) + einsum_bi_ijk_to_bjk(
#             state.z, self.W_rec)
#
#         new_v, new_z = self.LIF_dynamic(
#             v=state.v,
#             z=state.z,
#             z_buffer=state.z_buffer,
#             i_future_buffer=i_future_buffer)
#
#         new_z_buffer = tf_roll(state.z_buffer, new_z, axis=2)
#         new_i_future_buffer = tf_roll(i_future_buffer, axis=2)
#
#         new_state = LIFStateTuple(v=new_v,
#                                   z=new_z,
#                                   i_future_buffer=new_i_future_buffer,
#                                   z_buffer=new_z_buffer)
#         return new_z, new_state
#
#     def LIF_dynamic(self, v, z, z_buffer, i_future_buffer, thr=None, decay=None, n_refractory=None, add_current=0.):
#         """
#         Function that generate the next spike and voltage tensor for given cell state.
# .expand_dims(thr,axis=1)
#             v_buffer_scaled = (v_buffer - thr_expanded) / thr_expanded
#             new_z = different
#         :param v
#         :param z
#         :param z_buffer:
#         :param i_future_buffer:
#         :param thr:
#         :param decay:
#         :param n_refractory:
#         :param add_current:
#         :return:
#         """
#
#         if self.injected_noise_current > 0:
#             add_current = tf.random_normal(shape=z.shape, stddev=self.injected_noise_current)
#
#         with tf.name_scope('LIFdynamic'):
#             if thr is None: thr = self.thr
#             if decay is None: decay = self._decay
#             if n_refractory is None: n_refractory = self.n_refractory
#
#             i_t = i_future_buffer[:, :, 0] + add_current
#
#             I_reset = z * thr * self.dt
#
#             new_v = decay * v + (1 - decay) * i_t - I_reset
#
#             # Spike generation
#             v_scaled = (v - thr) / thr
#
#             # new_z = differentiable_spikes(v_scaled=v_scaled)
#             new_z = SpikeFunction(v_scaled, self.dampening_factor)
#             new_z.set_shape([None, v.get_shape()[1]])
#
#             if n_refractory > 0:
#                 is_ref = tf.greater(tf.reduce_max(z_buffer[:, :, -n_refractory:], axis=2), 0)
#                 new_z = tf.where(is_ref, tf.zeros_like(new_z), new_z)
#
#             new_z = new_z * 1/ self.dt
#
#             return new_v, new_z
#
# ALIFStateTuple = namedtuple('ALIFState', (
#     'z',
#     'v',
#     'b',
#
#     'i_future_buffer',
#     'z_buffer'))
#
#
# class ALIF(LIF):
#     def __init__(self, n_in, n_rec, tau=20, thr=0.01,
#                  dt=1., n_refractory=0, dtype=tf.float32, n_delay=1,
#                  tau_adaptation=200., beta=1.6,
#                  rewiring_connectivity=-1, dampening_factor=0.3,
#                  in_neuron_sign=None, rec_neuron_sign=None, injected_noise_current=0.,
#                  V0=1., add_current=0., thr_min=0.005):
#         """
#         Tensorflow cell object that simulates a LIF neuron with an approximation of the spike derivatives.
#
#         :param n_in: number of input neurons
#         :param n_rec: number of recurrent neurons
#         :param tau: membrane time constant
#         :param thr: threshold voltage
#         :param dt: time step of the simulation
#         :param n_refractory: number of refractory time steps
#         :param dtype: data type of the cell tensors
#         :param n_delay: number of synaptic delay, the delay range goes from 1 to n_delay time steps
#         :param tau_adaptation: adaptation time constant for the threshold voltage
#         :param beta: amplitude of adpatation
#         :param rewiring_connectivity: number of non-zero synapses in weight matrices (at initialization)
#         :param in_neuron_sign: vector of +1, -1 to specify input neuron signs
#         :param rec_neuron_sign: same of recurrent neurons
#         :param injected_noise_current: amplitude of current noise
#         :param V0: to choose voltage unit, specify the value of V0=1 Volt in the desired unit (example V0=1000 to set voltage in millivolts)
#         """
#
#         super(ALIF, self).__init__(n_in=n_in, n_rec=n_rec, tau=tau, thr=thr, dt=dt, n_refractory=n_refractory,
#                                    dtype=dtype, n_delay=n_delay,
#                                    rewiring_connectivity=rewiring_connectivity,
#                                    dampening_factor=dampening_factor, in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
#                                    injected_noise_current=injected_noise_current,
#                                    V0=V0)
#
#         if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
#         if beta is None: raise ValueError("beta parameter for adaptive bias must be set")
#
#         self.tau_adaptation = tau_adaptation
#         self.beta = beta
#         self.min_beta = np.min(beta)
#         self.elifs = beta < 0
#         self.decay_b = np.exp(-dt / tau_adaptation)
#         self.add_current = add_current
#         self.thr_min = thr_min
#         b_max = (thr_min - thr) / beta
#         b_max[~np.isfinite(b_max)] = np.finfo(b_max.dtype).max
#         self.b_max = b_max
#
#     @property
#     def output_size(self):
#         return [self.n_rec,self.n_rec]
#
#     @property
#     def state_size(self):
#         return ALIFStateTuple(v=self.n_rec,
#                               z=self.n_rec,
#                               b=self.n_rec,
#                               i_future_buffer=(self.n_rec, self.n_delay),
#                               z_buffer=(self.n_rec, self.n_refractory))
#
#     def zero_state(self, batch_size, dtype, n_rec=None):
#         if n_rec is None: n_rec = self.n_rec
#
#         v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#         z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#         b0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#
#         i_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_delay), dtype=dtype)
#         z_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_refractory), dtype=dtype)
#
#         return ALIFStateTuple(
#             v=v0,
#             z=z0,
#             b=b0,
#             i_future_buffer=i_buff0,
#             z_buffer=z_buff0
#         )
#
#     def __call__(self, inputs, state, scope=None, dtype=tf.float32):
#
#         i_future_buffer = state.i_future_buffer + einsum_bi_ijk_to_bjk(inputs, self.W_in) + einsum_bi_ijk_to_bjk(
#             state.z, self.W_rec)
#
#         new_b = self.decay_b * state.b + (np.ones(self.n_rec) - self.decay_b) * state.z
#         # in case of negatively adapting threshold (transient increase in excitability of ELIF neurons):
#         # clip adaptive threshold component (new_b) to prevent the threshold (thr) getting too small or negative
#         clipped_new_b = tf.minimum(new_b, tf.ones_like(new_b, dtype=dtype) * tf.cast(self.b_max, dtype=dtype))
#
#         thr = self.thr + new_b * self.beta * self.V0
#         clipped_thr = self.thr + clipped_new_b * self.beta * self.V0
#
#         thr = tf.where(tf.cast(tf.ones([tf.shape(inputs)[0], 1]) * self.elifs, dtype=tf.bool), clipped_thr, thr)
#
#         new_v, new_z = self.LIF_dynamic(
#             v=state.v,
#             z=state.z,
#             z_buffer=state.z_buffer,
#             i_future_buffer=i_future_buffer,
#             decay=self._decay,
#             thr=thr,
#             add_current=self.add_current,
#         )
#
#         new_z_buffer = tf_roll(state.z_buffer, new_z, axis=2)
#         new_i_future_buffer = tf_roll(i_future_buffer, axis=2)
#
#         new_state = ALIFStateTuple(v=new_v,
#                                    z=new_z,
#                                    b=new_b,
#                                    i_future_buffer=new_i_future_buffer,
#                                    z_buffer=new_z_buffer)
#         return [new_z, thr], new_state

#
FastALIFStateTuple = namedtuple('ALIFState', (
    'z',
    'v',
    'b',
    'r',
))
#
#
# class FastALIFold(Cell):
#     def __init__(self, n_in, n_rec, tau=20, thr=0.01,
#                  dt=1., n_refractory=0, dtype=tf.float32, n_delay=1,
#                  tau_adaptation=200., beta=1.6,
#                  rewiring_connectivity=-1, dampening_factor=0.3,
#                  in_neuron_sign=None, rec_neuron_sign=None, injected_noise_current=0.,
#                  add_current=0., thr_min=0.005):
#         """
#         Tensorflow cell object that simulates a LIF neuron with an approximation of the spike derivatives.
#
#         :param n_in: number of input neurons
#         :param n_rec: number of recurrent neurons
#         :param tau: membrane time constant
#         :param thr: threshold voltage
#         :param dt: time step of the simulation
#         :param n_refractory: number of refractory time steps
#         :param dtype: data type of the cell tensors
#         :param n_delay: number of synaptic delay, the delay range goes from 1 to n_delay time steps
#         :param tau_adaptation: adaptation time constant for the threshold voltage
#         :param beta: amplitude of adpatation
#         :param rewiring_connectivity: number of non-zero synapses in weight matrices (at initialization)
#         :param in_neuron_sign: vector of +1, -1 to specify input neuron signs
#         :param rec_neuron_sign: same of recurrent neurons
#         :param injected_noise_current: amplitude of current noise
#         """
#         if np.isscalar(tau): tau = tf.ones(n_rec, dtype=dtype) * np.mean(tau)
#         if np.isscalar(thr): thr = tf.ones(n_rec, dtype=dtype) * np.mean(thr)
#         tau = tf.cast(tau,dtype=dtype)
#         dt = tf.cast(dt,dtype=dtype)
#
#         self.dampening_factor = dampening_factor
#
#         # Parameters
#         self.n_delay = n_delay
#         self.n_refractory = n_refractory
#
#         self.dt = dt
#         self.n_in = n_in
#         self.n_rec = n_rec
#         self.data_type = dtype
#
#         self._num_units = self.n_rec
#
#         self.tau = tau
#         self._decay = tf.exp(-dt / tau)
#         self.thr = thr
#
#         self.injected_noise_current = injected_noise_current
#
#         self.rewiring_connectivity = rewiring_connectivity
#         self.in_neuron_sign = in_neuron_sign
#         self.rec_neuron_sign = rec_neuron_sign
#
#         if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
#         if beta is None: raise ValueError("beta parameter for adaptive bias must be set")
#
#         self.tau_adaptation = tau_adaptation
#         self.beta = beta
#         self.min_beta = np.min(beta)
#         self.elifs = beta < 0
#         self.decay_b = tf.exp(-dt / tau_adaptation)
#         self.add_current = add_current
#         self.thr_min = thr_min
#         # b_max = (thr_min - thr) / beta
#         # b_max[~np.isfinite(b_max)] = np.finfo(b_max.dtype).max
#         # self.b_max = b_max
#         self.built = False
#         self._keras_style = False
#
#     @tf_utils.shape_type_conversion
#     def build(self, inputs_shape):
#         if inputs_shape[-1] is None:
#             raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
#                              str(inputs_shape))
#         # _check_supported_dtypes(self.dtype)
#         n_in = inputs_shape[-1]
#         n_rec = self.n_rec
#
#         # Input weights
#         self.w_in_init = rd.randn(n_in, n_rec) / np.sqrt(n_in)
#         # self.w_in_var = tf.Variable(self.w_in_init, dtype=dtype, name="InputWeight")
#         self.w_in_var = self.add_variable("InputWeight", shape=[n_in, n_rec])
#         self.w_in_val = self.w_in_var
#         self.W_in = self.w_in_var
#
#         # self.w_rec_var = tf.Variable(rd.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype, name='RecurrentWeight')
#         self.w_rec_var = self.add_variable("RecurrentWeight", shape=[n_rec, n_rec])
#         self.w_rec_val = self.w_rec_var
#         # Disconnect autotapse
#         recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
#         self.w_rec_val = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val), self.w_rec_val)
#         self.W_rec = self.w_rec_val
#
#         self.built = True
#
#     @property
#     def output_size(self):
#         return [self.n_rec, self.n_rec]
#
#     @property
#     def state_size(self):
#         return FastALIFStateTuple(v=self.n_rec, z=self.n_rec, b=self.n_rec, r=self.n_rec)
#
#     def zero_state(self, batch_size, dtype, n_rec=None):
#         if n_rec is None: n_rec = self.n_rec
#
#         v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#         z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#         b0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#         r0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#
#         return FastALIFStateTuple(v=v0, z=z0, b=b0, r=r0)
#
#     def compute_z(self, v, adaptive_thr):
#         v_scaled = (v - adaptive_thr) / adaptive_thr
#         z = SpikeFunction(v_scaled, self.dampening_factor)
#         z = z * 1 / self.dt
#         return z
#
#     def __call__(self, inputs, state, scope=None, dtype=tf.float32):
#
#         i_in = tf.matmul(inputs, self.w_in_val)
#         i_rec = tf.matmul(state.z, self.w_rec_val)
#         i_t = i_in + i_rec + self.add_current
#
#         new_b = self.decay_b * state.b + (np.ones(self.n_rec) - self.decay_b) * state.z
#         # # in case of negatively adapting threshold (transient increase in excitability of ELIF neurons):
#         # # clip adaptive threshold component (new_b) to prevent the threshold (thr) getting too small or negative
#         # clipped_new_b = tf.minimum(new_b, tf.ones_like(new_b, dtype=dtype) * tf.cast(self.b_max, dtype=dtype))
#         thr = self.thr + new_b * self.beta
#         # clipped_thr = self.thr + clipped_new_b * self.beta
#         # thr = tf.where(tf.cast(tf.ones([tf.shape(inputs)[0], 1]) * self.elifs, dtype=tf.bool), clipped_thr, thr)
#
#         I_reset = state.z * thr * self.dt
#
#         new_v = self._decay * state.v + (1 - self._decay) * i_t - I_reset
#
#         # Spike generation
#         is_refractory = tf.greater(state.r, .1)
#         zeros_like_spikes = tf.zeros_like(state.z)
#         new_z = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, thr))
#         new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
#                                  0., float(self.n_refractory))
#
#         return [new_z, thr], FastALIFStateTuple(v=new_v, z=new_z, b=new_b, r=new_r)
#

class KerasALIF(DropoutRNNCellMixin, Layer):
  def __init__(self,
               n_in, units, tau=20, thr=0.01,
               dt=1., n_refractory=0, dtype=tf.float32, n_delay=1,
               tau_adaptation=200., beta=1.6,
               rewiring_connectivity=-1, dampening_factor=0.3,
               in_neuron_sign=None, rec_neuron_sign=None, injected_noise_current=0.,
               add_current=0., thr_min=0.005,
               input_initializer='glorot_normal',  # FIXME: try glorot_uniform
               recurrent_initializer='glorot_normal',
               input_regularizer=None,
               recurrent_regularizer=None,
               input_constraint=None,
               recurrent_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               eprop_sym=False,
               **kwargs):
    super(KerasALIF, self).__init__(**kwargs)
    self.units = units

    if np.isscalar(tau): tau = tf.ones(units, dtype=dtype) * np.mean(tau)
    if np.isscalar(thr): thr = tf.ones(units, dtype=dtype) * np.mean(thr)
    tau = tf.cast(tau, dtype=dtype)
    dt = tf.cast(dt, dtype=dtype)

    self.dampening_factor = dampening_factor
    self.eprop_sym = eprop_sym

    # Parameters
    self.n_delay = n_delay
    self.n_refractory = n_refractory

    self.dt = dt
    self.n_in = n_in
    self.data_type = dtype

    # self._num_units = self.n_rec

    self.tau = tau
    self._decay = tf.exp(-dt / tau)
    self.thr = thr

    self.injected_noise_current = injected_noise_current

    self.rewiring_connectivity = rewiring_connectivity
    self.in_neuron_sign = in_neuron_sign
    self.rec_neuron_sign = rec_neuron_sign

    if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
    if beta is None: raise ValueError("beta parameter for adaptive bias must be set")

    self.tau_adaptation = tau_adaptation
    self.beta = beta
    self.min_beta = np.min(beta)
    self.elifs = beta < 0
    self.decay_b = tf.exp(-dt / tau_adaptation)
    self.add_current = add_current
    self.thr_min = thr_min

    self.input_initializer = initializers.get(input_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)

    self.input_regularizer = regularizers.get(input_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)

    self.input_constraint = constraints.get(input_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    # self.state_size = FastALIFStateTuple(v=self.units, z=self.units, b=self.units, r=self.units)
    self.state_size = data_structures.NoDependency([self.units, self.units, self.units, self.units])
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if input_shape[-1] is None:
        raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                         str(input_shape))
    # _check_supported_dtypes(self.dtype)
    n_in = input_shape[-1]
    n_rec = self.units
    self.W_in = self.add_weight(
      name="InputWeight", shape=[n_in, n_rec],
      initializer=self.input_initializer,
      regularizer=self.input_regularizer,
      constraint=self.input_constraint,
    )
    self.W_rec = self.add_weight(
      name="RecurrentWeight", shape=[n_rec, n_rec],
      initializer=self.recurrent_initializer,
      regularizer=self.recurrent_regularizer,
      constraint=self.recurrent_constraint,
    )

    # Disconnect autotapse
    recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
    self.W_rec = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.W_rec), self.W_rec)

    self.built = True

  def compute_z(self, v, adaptive_thr):
    v_scaled = (v - adaptive_thr) / adaptive_thr
    z = SpikeFunction(v_scaled, self.dampening_factor)
    z = z * 1 / self.dt
    return z

  def call(self, inputs, states, training=None):

    dp_mask = self.get_dropout_mask_for_cell(inputs, training)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(states[1], training)
    if 0 < self.dropout < 1.:
        inputs = inputs * dp_mask
    if 0 < self.recurrent_dropout < 1.:
        state = FastALIFStateTuple(v=states[0], z=states[1] * rec_dp_mask, b=states[2], r=states[3])
    else:
        state = FastALIFStateTuple(v=states[0], z=states[1], b=states[2], r=states[3])

    new_b = self.decay_b * state.b + (np.ones(self.units) - self.decay_b) * state.z
    thr = self.thr + new_b * self.beta

    if self.eprop_sym:
      z = tf.stop_gradient(state.z)
    else:
      z = state.z

    i_in = tf.matmul(inputs, self.W_in)
    i_rec = tf.matmul(z, self.W_rec)
    i_t = i_in + i_rec + self.add_current

    I_reset = z * thr * self.dt

    new_v = self._decay * state.v + (1 - self._decay) * i_t - I_reset

    # Spike generation
    is_refractory = tf.greater(state.r, .1)
    zeros_like_spikes = tf.zeros_like(z)
    new_z = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, thr))
    new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
                             0., float(self.n_refractory))

    return new_z, [new_v, new_z, new_b, new_r]

  def get_config(self):
    config = {
        'units':
            self.units,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
    }
    base_config = super(KerasALIF, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(
        self, inputs, batch_size, dtype))


DelayALIFStateTuple = namedtuple('DelayALIFStateTuple', (
    'z',
    'v',
    'b',
    'r',
    'i_future_buffer',
))


class KerasDelayALIF(DropoutRNNCellMixin, Layer):
  def __init__(self,
               n_in, units, tau=20, thr=0.01,
               dt=1., n_refractory=0, dtype=tf.float32, n_delay=1,
               tau_adaptation=200., beta=1.6,
               rewiring_connectivity=-1, dampening_factor=0.3,
               in_neuron_sign=None, rec_neuron_sign=None, injected_noise_current=0.,
               add_current=0., thr_min=0.005,
               input_initializer='glorot_normal',  # FIXME: try glorot_uniform
               recurrent_initializer='glorot_normal',
               input_regularizer=None,
               recurrent_regularizer=None,
               input_constraint=None,
               recurrent_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               eprop_sym=False,
               **kwargs):
    super(KerasDelayALIF, self).__init__(**kwargs)
    self.units = units

    if np.isscalar(tau): tau = tf.ones(units, dtype=dtype) * np.mean(tau)
    if np.isscalar(thr): thr = tf.ones(units, dtype=dtype) * np.mean(thr)
    tau = tf.cast(tau, dtype=dtype)
    dt = tf.cast(dt, dtype=dtype)

    self.dampening_factor = dampening_factor
    self.eprop_sym = eprop_sym

    # Parameters
    self.n_delay = n_delay
    self.n_refractory = n_refractory

    self.dt = dt
    self.n_in = n_in
    self.data_type = dtype

    # self._num_units = self.n_rec

    self.tau = tau
    self._decay = tf.exp(-dt / tau)
    self.thr = thr

    self.injected_noise_current = injected_noise_current

    self.rewiring_connectivity = rewiring_connectivity
    self.in_neuron_sign = in_neuron_sign
    self.rec_neuron_sign = rec_neuron_sign

    if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
    if beta is None: raise ValueError("beta parameter for adaptive bias must be set")

    self.tau_adaptation = tau_adaptation
    self.beta = beta
    self.min_beta = np.min(beta)
    self.elifs = beta < 0
    self.decay_b = tf.exp(-dt / tau_adaptation)
    self.add_current = add_current
    self.thr_min = thr_min

    self.input_initializer = initializers.get(input_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)

    self.input_regularizer = regularizers.get(input_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)

    self.input_constraint = constraints.get(input_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    # self.state_size = DelayALIFStateTuple(z=self.units, v=self.units, b=self.units, r=self.units,
    #                                       i_future_buffer=(self.units, self.n_delay))
    self.state_size = data_structures.NoDependency(
      [self.units, self.units, self.units, self.units, self.units * self.n_delay])
    self.output_size = self.units

  def weight_matrix_with_delay_dimension(self, w, d, n_delay):
    """
    Generate the tensor of shape n_in x n_out x n_delay that represents the synaptic weights with the right delays.

    :param w: synaptic weight value, float tensor of shape (n_in x n_out)
    :param d: delay number, int tensor of shape (n_in x n_out)
    :param n_delay: number of possible delays
    :return:
    """
    with tf.name_scope('WeightDelayer'):
      w_d_list = []
      for kd in range(n_delay):
        mask = tf.equal(d, kd)
        w_d = tf.where(condition=mask, x=w, y=tf.zeros_like(w))
        w_d_list.append(w_d)

      delay_axis = len(d.shape)
      WD = tf.stack(w_d_list, axis=delay_axis)

    return WD

  def tf_roll(self, buffer, new_last_element=None, axis=0):
    with tf.name_scope('roll'):
      shp = buffer.get_shape()
      l_shp = len(shp)

      # Permute the index to roll over the right index
      perm = np.concatenate([[axis], np.arange(axis), np.arange(start=axis + 1, stop=l_shp)])
      buffer = tf.transpose(buffer, perm=perm)

      # Add an element at the end of the buffer if requested, otherwise, add zero
      if new_last_element is None:
        shp = tf.shape(buffer)
        new_last_element = tf.zeros(shape=shp[1:], dtype=buffer.dtype)
      new_last_element = tf.expand_dims(new_last_element, axis=0)
      new_buffer = tf.concat([buffer[1:], new_last_element], axis=0, name='rolled')

      # Revert the index permutation
      inv_perm = np.argsort(perm)
      new_buffer = tf.transpose(new_buffer, perm=inv_perm)

      new_buffer = tf.identity(new_buffer, name='Roll')
      # new_buffer.set_shape(shp)
    return new_buffer

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if input_shape[-1] is None:
        raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                         str(input_shape))
    # _check_supported_dtypes(self.dtype)
    n_in = input_shape[-1]
    n_rec = self.units

    # self.w_in_init = rd.randn(n_in, n_rec) / np.sqrt(n_in)
    # self.w_in_var = tf.Variable(self.w_in_init, dtype=self.data_type, name="InputWeight")
    self.w_in_var = self.add_weight(
      name="InputWeight", shape=[n_in, n_rec],
      initializer=self.input_initializer,
      regularizer=self.input_regularizer,
      constraint=self.input_constraint,
    )
    self.w_in_val = self.w_in_var
    self.w_in_delay = tf.Variable(rd.randint(self.n_delay, size=n_in * n_rec).reshape(n_in, n_rec),
                                  dtype=tf.int32, name="InDelays", trainable=False)
    self.W_in = self.weight_matrix_with_delay_dimension(self.w_in_val, self.w_in_delay, self.n_delay)

    # self.w_rec_var = tf.Variable(rd.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=self.data_type, name='RecurrentWeight')
    self.w_rec_var = self.add_weight(
      name="RecurrentWeight", shape=[n_rec, n_rec],
      initializer=self.recurrent_initializer,
      regularizer=self.recurrent_regularizer,
      constraint=self.recurrent_constraint,
    )
    self.w_rec_val = self.w_rec_var
    recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
    self.w_rec_val = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val), self.w_rec_val)
    self.w_rec_delay = tf.Variable(rd.randint(self.n_delay, size=n_rec * n_rec).reshape(n_rec, n_rec), dtype=tf.int32,
                                   name="RecDelays", trainable=False)
    self.W_rec = self.weight_matrix_with_delay_dimension(self.w_rec_val, self.w_rec_delay, self.n_delay)

    self.built = True

  def compute_z(self, v, adaptive_thr):
    v_scaled = (v - adaptive_thr) / adaptive_thr
    z = SpikeFunction(v_scaled, self.dampening_factor)
    z = z * 1 / self.dt
    return z

  def call(self, inputs, states, training=None):

    dp_mask = self.get_dropout_mask_for_cell(inputs, training)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(states[1], training)
    if 0 < self.dropout < 1.:
        inputs = inputs * dp_mask
    if 0 < self.recurrent_dropout < 1.:
        state = DelayALIFStateTuple(v=states[0], z=states[1] * rec_dp_mask, b=states[2], r=states[3],
                                    i_future_buffer=states[4])
    else:
        state = DelayALIFStateTuple(v=states[0], z=states[1], b=states[2], r=states[3], i_future_buffer=states[4])

    new_b = self.decay_b * state.b + (np.ones(self.units) - self.decay_b) * state.z
    thr = self.thr + new_b * self.beta

    if self.eprop_sym:
      z = tf.stop_gradient(state.z)
    else:
      z = state.z

    i_in = tf.einsum('bi,ijk->bjk', inputs, self.W_in)
    i_rec = tf.einsum('bi,ijk->bjk', z, self.W_rec)
    old_i_future_buffer = tf.reshape(state.i_future_buffer, shape=[-1, self.units, self.n_delay])
    i_future_buffer = old_i_future_buffer + i_in + i_rec
    i_t = i_future_buffer[:, :, 0] + self.add_current

    I_reset = z * thr * self.dt

    new_v = self._decay * state.v + (1 - self._decay) * i_t - I_reset

    # Spike generation
    is_refractory = tf.greater(state.r, .1)
    zeros_like_spikes = tf.zeros_like(z)
    new_z = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, thr))
    new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
                             0., float(self.n_refractory))
    new_i_future_buffer = self.tf_roll(i_future_buffer, axis=2)

    return new_z, [new_v, new_z, new_b, new_r, tf.reshape(new_i_future_buffer, shape=[-1, self.units * self.n_delay])]

  def get_config(self):
    config = {
        'units':
            self.units,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
    }
    base_config = super(KerasALIF, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    n_rec = self.units

    v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
    z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
    b0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
    r0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

    i_buff0 = tf.zeros(shape=(batch_size, n_rec * self.n_delay), dtype=dtype)

    # return DelayALIFStateTuple(
    #   z=z0,
    #   v=v0,
    #   b=b0,
    #   r=r0,
    #   i_future_buffer=i_buff0
    # )
    return [v0, z0, b0, r0, i_buff0]

#
# STPStateTuple = namedtuple('STPState', (
#     'z',
#     'v',
#     'r',
#     'c',
#     'u',
#     'x',
# ))
#
#
# class STP(Cell):
#     def __init__(self, n_in, n_rec, tau=20, thr=0.01,
#                  dt=1., n_refractory=0, dtype=tf.float32,
#                  tau_D=200., tau_F=1500., U=0.2,
#                  rewiring_connectivity=-1, dampening_factor=0.3,
#                  in_neuron_sign=None, rec_neuron_sign=None, add_current=0.,
#                  w_in_init=None, w_rec_init=None,
#                  ):
#         """
#         Tensorflow cell object that simulates a LIF neuron with short term plasticity dynamic on synapses.
#
#         :param n_in: number of input neurons
#         :param n_rec: number of recurrent neurons
#         :param tau: membrane time constant
#         :param thr: threshold voltage
#         :param dt: time step of the simulation
#         :param n_refractory: number of refractory time steps
#         :param dtype: data type of the cell tensors
#         :param n_delay: number of synaptic delay, the delay range goes from 1 to n_delay time steps
#         :param tau_adaptation: adaptation time constant for the threshold voltage
#         :param beta: amplitude of adpatation
#         :param rewiring_connectivity: number of non-zero synapses in weight matrices (at initialization)
#         :param in_neuron_sign: vector of +1, -1 to specify input neuron signs
#         :param rec_neuron_sign: same of recurrent neurons
#         :param injected_noise_current: amplitude of current noise
#         :param V0: to choose voltage unit, specify the value of V0=1 Volt in the desired unit (example V0=1000 to set voltage in millivolts)
#         """
#         self.n_refractory = n_refractory
#         self.dampening_factor = dampening_factor
#         self.dt = dt
#         self.n_in = n_in
#         self.n_rec = n_rec
#         self.data_type = dtype
#         self._num_units = self.n_rec
#         self.tau = tau
#         self._decay = tf.exp(-dt / tau)
#         self.thr = thr
#
#         with tf.variable_scope('InputWeights'):
#             # Input weights
#             if 0 < rewiring_connectivity < 1:
#                 self.w_in_val, self.w_in_sign, self.w_in_var, _ = \
#                     weight_sampler(n_in, n_rec, rewiring_connectivity, neuron_sign=in_neuron_sign)
#             else:
#                 init_w_in_var = w_in_init if w_in_init is not None else \
#                     (rd.randn(n_in, n_rec) / np.sqrt(n_in)).astype(np.float32)
#                 self.w_in_var = tf.get_variable("InputWeight", initializer=init_w_in_var, dtype=dtype)
#                 self.w_in_val = self.w_in_var
#
#         with tf.variable_scope('RecWeights'):
#             if 0 < rewiring_connectivity < 1:
#                 self.w_rec_val, self.w_rec_sign, self.w_rec_var, _ = \
#                     weight_sampler(n_rec, n_rec, rewiring_connectivity, neuron_sign=rec_neuron_sign)
#             else:
#                 init_w_rec_var = w_rec_init if w_rec_init is not None else \
#                     (rd.randn(n_rec, n_rec) / np.sqrt(n_rec)).astype(np.float32)
#                 self.w_rec_var = tf.get_variable('RecurrentWeight', initializer=init_w_rec_var, dtype=dtype)
#
#             self.w_rec_val = self.w_rec_var
#             self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
#             # Disconnect autotapse
#             self.w_rec_val = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val), self.w_rec_val)
#
#         self.tau_D = tau_D
#         self.tau_F = tau_F
#         self.U = U
#         self.add_current = add_current
#
#     @property
#     def output_size(self):
#         return [self.n_rec, self.n_rec, self.n_rec]
#
#     @property
#     def state_size(self):
#         return STPStateTuple(
#             v=self.n_rec,
#             z=self.n_rec,
#             r=self.n_rec,
#             c=self.n_rec,
#             u=self.n_rec,
#             x=self.n_rec,
#         )
#
#     def zero_state(self, batch_size, dtype, n_rec=None):
#         if n_rec is None: n_rec = self.n_rec
#
#         v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#         z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#         r0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#         c0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#         u0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#         x0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
#
#         return STPStateTuple(
#             v=v0,
#             z=z0,
#             r=r0,
#             c=c0,
#             u=u0,
#             x=x0,
#         )
#
#     def compute_z(self, v):
#         v_scaled = (v - self.thr) / self.thr
#         z = SpikeFunction(v_scaled, self.dampening_factor)
#         z = z * 1 / self.dt
#         return z
#
#     def __call__(self, inputs, state, scope=None, dtype=tf.float32):
#         # u = np.exp(-dt / tauF) * u + U * (1. - (u + U)) * z
#         new_u = tf.exp(-self.dt / self.tau_F) * state.u + self.U * (1. - (state.u + self.U)) * state.z
#
#         # x = np.exp(-dt / tauD) * x + u * (1. - x) * z
#         new_x = tf.exp(-self.dt / self.tau_D) * state.x + (new_u + self.U) * (1. - state.x) * state.z
#
#         # u_trc[t] = u + U
#         # x_trc[t] = 1. - x
#         ux = tf.multiply(state.u + self.U, tf.ones_like(state.x) - state.x)  # (batch, neuron)
#         # w_rec_stp = tf.einsum('bi,ij->bij', ux, self.w_rec_val)  # (batch, neuron, neuron)
#
#         i_in = tf.matmul(inputs, self.w_in_val)
#         # i_rec = tf.einsum('bi,bij->bj', state.z, w_rec_stp)
#         i_rec = tf.matmul(state.z * ux, self.w_rec_val)
#         i_t = i_in + i_rec
#
#         i_reset = state.z * self.thr * self.dt
#
#         new_v = self._decay * state.v + (1. - self._decay) * i_t - i_reset
#
#         # Spike generation
#         is_refractory = tf.greater(state.r, .1)
#         zeros_like_spikes = tf.zeros_like(state.z)
#         new_z = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v))
#         new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1.,
#                                  0., float(self.n_refractory))
#         new_c = state.c + (tf.ones_like(new_z) - new_z) - new_z * state.c
#
#         new_state = STPStateTuple(
#             v=new_v,
#             z=new_z,
#             r=new_r,
#             c=new_c,
#             u=new_u,
#             x=new_x,
#         )
#         return [new_z, new_u, new_x], new_state
