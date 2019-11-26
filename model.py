# -*- coding: UTF-8 -*-
from __future__ import division
from __future__ import print_function
import random
import numpy as np
import tensorflow as tf
import rnn
import time

from tensorflow.python.layers.core import Dense
import basic_decoder
import decoder
import attention_wrapper
from utils import *

def copy_hparams(hparams):
    """Return a copy of an HParams instance."""
    return tf.contrib.training.HParams(**hparams.values())

def idx(c):
    for i,v in enumerate(c):
        if v>=0.7:
            return int(i)
        elif v>=0.4:
            return str(int(i))
        elif v>0:
            return int(np.floor(i / 7) * 7)

def samp(p,c):
    I = idx(c[0])
    if type(I) == type('s'):
        i = np.random.randint(20, size=1)
        #i = int(int(time.time())%20)
        return p['a'+I][p['s'+I][i]]
    else:
        i = np.random.randint(300, size=1)
        # if i in p['s'+str(I)]:
        #   i = np.random.randint(500, size=1)
        return p['a'+str(I)][200+i]
        

def refine(s):
    l=[]
    for i in s:
      if np.random.rand(1)>0.9999:
        i+=[0.2*np.random.rand(1)[0]-0.1,0.2*np.random.rand(1)[0]-0.1,0]
      l.append(i)

    s = np.array(l)
    s /= [25,25,1]
    return s

def get_default_hparams():
    """Return default HParams for sketch-rnn."""
    hparams = tf.contrib.training.HParams(
      data_set=['aaron_sheep.npz'],  # Our dataset.
      num_steps=10000000,  # Total number of steps of training. Keep large.
      save_every=500,  # Number of batches per checkpoint creation.
      max_seq_len=250,  # Not used. Will be changed by model. [Eliminate?]
      dec_rnn_size=512,  # Size of decoder.
      dec_model='lstm',  # Decoder: lstm, layer_norm or hyper.
      enc_rnn_size=256,  # Size of encoder.
      enc_model='lstm',  # Encoder: lstm, layer_norm or hyper.
      z_size=128,  # Size of latent vector z. Recommend 32, 64 or 128.
      kl_weight=0.5,  # KL weight of loss equation. Recommend 0.5 or 1.0.
      kl_weight_start=0.01,  # KL start weight when annealing.
      kl_tolerance=0.2,  # Level of KL loss at which to stop optimizing for KL.
      batch_size=100,  # Minibatch size. Recommend leaving at 100.
      grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
      num_mixture=20,  # Number of mixtures in Gaussian mixture model.
      learning_rate=0.001,  # Learning rate.
      decay_rate=0.9999,  # Learning rate decay per minibatch.
      kl_decay_rate=0.99995,  # KL annealing decay rate per minibatch.
      min_learning_rate=0.00001,  # Minimum learning rate.
      use_recurrent_dropout=True,  # Dropout with memory loss. Recomended
      recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep.
      use_input_dropout=False,  # Input dropout. Recommend leaving False.
      input_dropout_prob=0.90,  # Probability of input dropout keep.
      use_output_dropout=False,  # Output droput. Recommend leaving False.
      output_dropout_prob=0.90,  # Probability of output dropout keep.
      random_scale_factor=0.15,  # Random scaling data augmention proportion.
      augment_stroke_prob=0.10,  # Point dropping augmentation proportion.
      conditional=True,  # When False, use unconditional decoder-only model.
      is_training=True,  # Is model training? Recommend keeping true.
      gamma_coeff=0.01
    )
    return hparams

class Model(object):

    def __init__(self, hps, expr_num, attention_temperature=0.1, use_hmean=False, gpu_mode=True, reuse=False):
        """Initializer for the SketchRNN model.
        Args:
           hps: a HParams object containing model hyperparameters.
           expr_num: an integer that is the number of expressions.
           attention_temperature: a float controling the attention mechanism weight.
           use_hmean: a boolean that when true, set mu of attention wrapper as mean of h.
           gpu_mode: a boolean that when True, uses GPU mode.
           reuse: a boolean that when true, attemps to reuse variables.
        """
        self.hps = hps
        self.expr_num = expr_num
        self.attention_temperature = attention_temperature
        self.use_hmean = use_hmean
        with tf.variable_scope('model', reuse=reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu.')
            else:
                with tf.device('/gpu:0'):
                    tf.logging.info('Model using gpu.')
            self.build_model(reuse)

    def build_model(self,reuse):
      """Define model architecture."""
      self.init_cell()
      self.init_placeholder()
      self.encoder(reuse)
      if self.hps.conditional:
        self.latent_space(reuse)
      self.decoder(reuse)
      self.loss(reuse)
      self.optimize(reuse)

        
    def init_cell(self):
        if self.hps.is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if self.hps.dec_model == 'lstm':
            cell_fn = rnn.LSTMCell
        elif self.hps.dec_model == 'layer_norm':
            cell_fn = rnn.LayerNormLSTMCell
        elif self.hps.dec_model == 'hyper':
            cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        if self.hps.enc_model == 'lstm':
            enc_cell_fn = rnn.LSTMCell
        elif self.hps.enc_model == 'layer_norm':
            enc_cell_fn = rnn.LayerNormLSTMCell
        elif self.hps.enc_model == 'hyper':
            enc_cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        use_recurrent_dropout = self.hps.use_recurrent_dropout
        use_input_dropout = self.hps.use_input_dropout
        use_output_dropout = self.hps.use_output_dropout

        cell = cell_fn(
            self.hps.dec_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)

        if self.hps.conditional:
            self.enc_cell_fw = enc_cell_fn(
                self.hps.enc_rnn_size,
                use_recurrent_dropout=use_recurrent_dropout,
                dropout_keep_prob=self.hps.recurrent_dropout_prob)
            self.enc_cell_bw = enc_cell_fn(
                self.hps.enc_rnn_size,
                use_recurrent_dropout=use_recurrent_dropout,
                dropout_keep_prob=self.hps.recurrent_dropout_prob)

        tf.logging.info('Input dropout mode = %s.', use_input_dropout)
        tf.logging.info('Output dropout mode = %s.', use_output_dropout)
        tf.logging.info('Recurrent dropout mode = %s.', use_recurrent_dropout)
        if use_input_dropout:
            tf.logging.info('Dropout to input w/ keep_prob = %4.4f.',
                          self.hps.input_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(
              cell, input_keep_prob=self.hps.input_dropout_prob)
        if use_output_dropout:
            tf.logging.info('Dropout to output w/ keep_prob = %4.4f.',
                          self.hps.output_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(
              cell, output_keep_prob=self.hps.output_dropout_prob)
        self.cell = cell

    def init_placeholder(self):
      # Create palceholders for inputs to the model
        self.sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[self.hps.batch_size])        
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len + 1, 5])
        self.c_expr = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.expr_num])

    def encoder(self,reuse):
        with tf.variable_scope("encode", reuse=reuse):
            self.output_x = self.input_data[:, 1:self.hps.max_seq_len + 1, :]
            self.input_x = self.input_data[:, :self.hps.max_seq_len, :]

            if self.hps.conditional:
                overlay_c = tf.tile(self.c_expr, [self.hps.batch_size, self.hps.max_seq_len, self.expr_num])
                actual_output = tf.concat([self.output_x, overlay_c], 2)

                self.enc_output, last_states = tf.nn.bidirectional_dynamic_rnn(
                        self.enc_cell_fw,
                        self.enc_cell_bw,
                        self.output_x,
                        sequence_length=self.sequence_lengths,
                        dtype=tf.float32,
                        scope='ENC_RNN')

                last_state_fw, last_state_bw = last_states
                
                last_h_fw = self.enc_cell_fw.get_output(last_state_fw)
                last_h_bw = self.enc_cell_bw.get_output(last_state_bw)
                
                self.last_h = tf.concat([last_h_fw, last_h_bw, self.c_expr], 1)
                self.enc_outputs = tf.concat([self.enc_output[0], self.enc_output[1]], -1, name='encoder_outputs')

    def latent_space(self,reuse):
        with tf.variable_scope("latent_space", reuse=reuse): 
            mu = rnn.super_linear(
                    self.last_h,
                    self.hps.z_size,
                    input_size=self.hps.enc_rnn_size * 2 + self.expr_num,  # bi-dir, so x2
                    scope='ENC_RNN_mu',
                    init_w='gaussian',
                    weight_start=0.001)
                
            presig = rnn.super_linear(
                self.last_h,
                self.hps.z_size,
                input_size=self.hps.enc_rnn_size * 2 + self.expr_num,  # bi-dir, so x2
                scope='ENC_RNN_sigma',
                init_w='gaussian',
                weight_start=0.001)          
            self.mean, self.presig = mu, presig
            self.sigma = tf.exp(self.presig / 2.0)  # sigma > 0. div 2.0 -> sqrt.     
            eps = tf.random_normal((self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)            
            self.batch_z = self.mean + tf.multiply(self.sigma, eps)

    def tf_2d_normal(self, x1, x2, mu1, mu2, s1, s2, rho):
        """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""

        norm1 = tf.subtract(x1, mu1)

        norm2 = tf.subtract(x2, mu2)
        s1s2 = tf.multiply(s1, s2)

        # eq 25
        z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
           2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
        neg_rho = 1 - tf.square(rho)
        result = tf.exp(tf.div(-z, 2 * neg_rho))
        denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
        result = tf.div(result, denom)
        return result
  
    def get_mixture_coef(self,output):
        """Returns the tf slices containing mdn dist params."""
        # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
        z = output
        z_pen_logits = z[:, 0:3]  # pen states

        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

        # process output z's into MDN paramters

        # softmax all the pi's and pen states:
        z_pi = tf.nn.softmax(z_pi)
        z_pen = tf.nn.softmax(z_pen_logits)

        # exponentiate the sigmas and also make corr between -1 and 1.
        z_sigma1 = tf.exp(z_sigma1)
        z_sigma2 = tf.exp(z_sigma2)
        z_corr = tf.tanh(z_corr)

        r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
        return r

    def decoder(self,reuse):
        with tf.variable_scope("decode", reuse=reuse):

            if self.hps.conditional:
                self.batch_z = tf.concat([self.batch_z, self.c_expr],1)     
                pre_tile_y = tf.reshape(self.batch_z,
                                      [self.hps.batch_size, 1, self.hps.z_size+self.expr_num])
                overlay_x = tf.tile(pre_tile_y, [1, self.hps.max_seq_len, 1])  #replicating input multiples times
                actual_input_x = tf.concat([self.input_x, overlay_x], 2)

            else:
                self.batch_z = tf.zeros((self.hps.batch_size, self.hps.z_size), dtype=tf.float32)
                self.batch_z = tf.concat([self.batch_z, self.c_expr],1)
                actual_input_x = self.input_x

            self.num_mixture = self.hps.num_mixture # Number of mixtures in Gaussian mixture model.

            n_out = (3 + self.num_mixture * 6)

            output_w = tf.get_variable('output_w',[self.hps.dec_rnn_size, n_out])
            output_b = tf.get_variable('output_b',[n_out])


            if self.hps.conditional:
                self.initial_state = tf.nn.tanh(
                  rnn.super_linear(
                      self.batch_z,
                      self.cell.state_size,
                      init_w='gaussian',
                      weight_start=0.001,
                      input_size=self.hps.z_size+self.expr_num))

            else:
                self.initial_state = self.cell.zero_state(batch_size=self.hps.batch_size, dtype=tf.float32)

            output, last_state = tf.nn.dynamic_rnn(
              self.cell,
              actual_input_x,
              initial_state=self.initial_state,
              time_major=False,
              swap_memory=True,
              dtype=tf.float32,
              scope='RNN')
            
            self.c_kl_batch_train = tf.zeros([], dtype=tf.float32)
            self.training_logits = output
            output = tf.reshape(self.training_logits, [-1, self.hps.dec_rnn_size])
            output = tf.nn.xw_plus_b(output, output_w, output_b)
            self.final_state = last_state

            out = self.get_mixture_coef(output)

            [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out

            self.pi = o_pi
            self.mu1 = o_mu1
            self.mu2 = o_mu2
            self.sigma1 = o_sigma1
            self.sigma2 = o_sigma2
            self.corr = o_corr
            self.pen_logits = o_pen_logits
            self.pen = o_pen


    def recon_loss(self, z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,z_pen_logits, x1_data, x2_data, pen_data):
            
        """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
        # This represents the L_R only (i.e. does not include the KL loss term).


        result0 = self.tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
        epsilon = 1e-6
        # result1 is the loss wrt pen offset (L_s in equation 9 of
        # https://arxiv.org/pdf/1704.03477.pdf)
        result1 = tf.multiply(result0, z_pi)
        result1 = tf.reduce_sum(result1, 1, keep_dims=True)
        result1 = -tf.log(result1 + epsilon)  # avoid log(0)

        # loc_weight = tf.reshape(pen_data[:, 0]*2, [-1, 1])
        # result1 = tf.multiply(result1, loc_weight)

        #这里的fs是什么？？
        fs = 1.0 - pen_data[:, 2]  # use training data for this
        fs = tf.reshape(fs, [-1, 1])
        # Zero out loss terms beyond N_s, the last actual stroke
        result1 = tf.multiply(result1, fs)

        # result2: loss wrt pen state, (L_p in equation 9)
        result2 = tf.nn.softmax_cross_entropy_with_logits(
          labels=pen_data, logits=z_pen_logits)
        result2 = tf.reshape(result2, [-1, 1])
        if not self.hps.is_training:  # eval mode, mask eos columns
            result2 = tf.multiply(result2, fs)

        result = result1 + result2
        return result

    def loss(self,reuse):
        with tf.variable_scope('losses', reuse=reuse):
            if self.hps.conditional:
                  self.kl_cost = -0.5 * tf.reduce_mean((1 + self.presig - tf.square(self.mean) - tf.exp(self.presig)))
                  self.kl_cost = tf.maximum(self.kl_cost, self.hps.kl_tolerance) #kl_tolerance Level of KL loss at which to stop optimizing for KL.
            else:
                self.kl_cost = tf.zeros([], dtype=tf.float32)

            #self.context_kl_cost = tf.zeros([], dtype=tf.float32)
            self.context_kl_cost = tf.scalar_mul(self.hps.gamma_coeff, self.c_kl_batch_train)

            # Create the weights for sequence_loss
            # masks = tf.sequence_mask(self.target_sentence_length, self.decoder_num_tokens, dtype=tf.float32, name='masks')

            target = tf.reshape(self.output_x, [-1, 5])
            [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(target, 5, 1)
            pen_data = tf.concat([eos_data, eoc_data, cont_data], 1)

            self.r_cost = self.recon_loss(self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.corr, self.pen_logits, x1_data, x2_data, pen_data)            
            self.r_cost = tf.reduce_mean(self.r_cost)
            self.kl_weight = tf.Variable(self.hps.kl_weight_start, trainable=False)
            self.cost = self.r_cost + self.kl_cost * self.kl_weight


    def optimize(self,reuse):
        with tf.variable_scope('optimization', reuse=reuse):
            if self.hps.is_training:
                  self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
                  optimizer = tf.train.AdamOptimizer(self.lr)           
                  gvs = optimizer.compute_gradients(self.cost)
                  g = self.hps.grad_clip
                  capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs if grad is not None]
                  self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')



def sample(sess, pre, model, seq_len=250, temperature=1.0, greedy_mode=False,z=None,c=None):
    """Samples a sequence from a pre-trained model."""

    def adjust_temp(pi_pdf, temp):
        pi_pdf = np.log(pi_pdf) / temp
        pi_pdf -= pi_pdf.max()
        pi_pdf = np.exp(pi_pdf)
        pi_pdf /= pi_pdf.sum()
        return pi_pdf

    def get_pi_idx(x, pdf, temp=1.0, greedy=False):
        """Samples from a pdf, optionally greedily."""
        if greedy:
            return np.argmax(pdf)
        pdf = adjust_temp(np.copy(pdf), temp)
        accumulate = 0
        for i in range(0, pdf.size):
            accumulate += pdf[i]
            if accumulate >= x:
                return i
        tf.logging.info('Error with sampling ensemble.')
        return -1

    def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
        if greedy:
            return mu1, mu2
        mean = [mu1, mu2]
        s1 *= temp * temp
        s2 *= temp * temp
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
    print("c=",c)
    if z is None:
        c_ = sess.run(model.c_expr, feed_dict={model.c_expr: c})
        z_ = np.random.randn(1, model.hps.z_size)  # not used if unconditional
        # c = np.array([[0.2,0.8]])
        z = np.append(z_,c_,axis=1)

    if not model.hps.conditional:
        #prev_state = sess.run(model.initial_state)
        prev_state = sess.run(model.initial_state)
    else:
        prev_state = sess.run(model.initial_state, feed_dict={model.batch_z: z})

    strokes = np.zeros((seq_len, 5), dtype=np.float32)
    seq_len = 0
    mixture_params = []
    greedy = False
    temp = 1.0 

    stroke = samp(pre,c)
    strokes = refine(stroke)

    for i in range(seq_len):
        if not model.hps.conditional:
            feed = {
              model.input_x: prev_x,
              model.sequence_lengths: [1],
              model.initial_state: prev_state
            }
        else:
            feed = {
              model.input_x: prev_x,
              model.sequence_lengths: [1],
              model.initial_state: prev_state,
              model.batch_z: z
            }

        params = sess.run([
            model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr,
            model.pen, model.final_state
        ], feed)

        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = params

        if i < 0:
            greedy = False
            temp = 1.0
        else:
            greedy = greedy_mode
            temp = temperature

        idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)

        idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)
        eos = [0, 0, 0]
        eos[idx_eos] = 1

        next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                              o_sigma1[0][idx], o_sigma2[0][idx],
                                              o_corr[0][idx], np.sqrt(temp), greedy)

        strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

        params = [
            o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0],
            o_pen[0]
        ]

        mixture_params.append(params)

        prev_x = np.zeros((1, 1, 5), dtype=np.float32)
        prev_x[0][0] = np.array(
            [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
        prev_state = next_state

        if idx_eos == 2:
            break

    return strokes, mixture_params


