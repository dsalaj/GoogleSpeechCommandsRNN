# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Simple speech recognition to spot a limited number of keywords.

This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio_recognition.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run tensorflow/examples/speech_commands:train

This will write out checkpoints to .results/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down

Best LSNN training runs:
python3 train.py --model_architecture=lsnn --n_hidden=2048 --window_stride_ms=1. --avg_spikes=True

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import json
from datetime import datetime

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile

FLAGS = None


def main(_):
  # Set the verbosity based on flags (default is INFO, so we see all messages)
  tf.compat.v1.logging.set_verbosity(FLAGS.verbosity)

  # Start a new TensorFlow session.
  sess = tf.compat.v1.InteractiveSession()

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess,
      FLAGS.in_repeat
  )
  model_settings['n_hidden'] = FLAGS.n_hidden
  model_settings['n_layer'] = FLAGS.n_layer
  model_settings['dropout_prob'] = FLAGS.dropout_prob
  model_settings['n_lif_frac'] = FLAGS.n_lif_frac
  model_settings['tau'] = FLAGS.tau
  model_settings['refr'] = FLAGS.refr
  model_settings['beta'] = FLAGS.beta
  model_settings['n_thr_spikes'] = FLAGS.n_thr_spikes
  model_settings['n_delay'] = FLAGS.n_delay
  model_settings['eprop'] = FLAGS.eprop
  model_settings['random_eprop'] = FLAGS.random_eprop
  model_settings['avg_spikes'] = FLAGS.avg_spikes
  # model_settings['in_repeat'] = FLAGS.in_repeat
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir,
      FLAGS.silence_percentage, FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings, FLAGS.summaries_dir,
      FLAGS.n_thr_spikes, FLAGS.in_repeat
  )
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))
  n_thr_spikes = max(1, FLAGS.n_thr_spikes)
  training_placeholder = tf.compat.v1.placeholder(tf.bool, name='is_training')
  input_placeholder = tf.compat.v1.placeholder(
    tf.float32, [None, model_settings['fingerprint_size'] * (2 * n_thr_spikes - 1) * model_settings['in_repeat']],
    name='fingerprint_input')
  if FLAGS.quantize:
    fingerprint_min, fingerprint_max = input_data.get_features_range(
        model_settings)
    fingerprint_input = tf.quantization.fake_quant_with_min_max_args(
        input_placeholder, fingerprint_min, fingerprint_max)
  else:
    fingerprint_input = input_placeholder

  model_out = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      is_training=training_placeholder)

  if FLAGS.model_architecture == 'lsnn':
    logits, spikes, dropout_prob = model_out
    av = tf.reduce_mean(spikes, axis=(0, 1))
  else:
    logits, dropout_prob = model_out

  # Define loss and optimizer
  ground_truth_input = tf.compat.v1.placeholder(
      tf.int64, [None], name='groundtruth_input')

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.compat.v1.add_check_numerics_ops()
    control_dependencies = [checks]

  # Create the back propagation and training evaluation machinery in the graph.
  with tf.compat.v1.name_scope('cross_entropy'):
    cross_entropy_mean = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=ground_truth_input, logits=logits)

  if FLAGS.model_architecture == 'lsnn':
    regularization_f0 = 10 / 1000  # 10Hz
    loss_reg = tf.reduce_sum(tf.square(av - regularization_f0) * FLAGS.reg)
    cross_entropy_mean += loss_reg

  if FLAGS.quantize:
    try:
      tf.contrib.quantize.create_training_graph(quant_delay=0)
    except ImportError as e:
      msg = e.args[0]
      msg += ('\n\n The --quantize option still requires contrib, which is not '
              'part of TensorFlow 2.0. Please install a previous version:'
              '\n    `pip install tensorflow<=1.15`')
      e.args = (msg,)
      raise e

  with tf.compat.v1.name_scope('train'), tf.control_dependencies(
      control_dependencies):
    learning_rate_input = tf.compat.v1.placeholder(
        tf.float32, [], name='learning_rate_input')
    if FLAGS.optimizer == 'gradient_descent':
      train_step = tf.compat.v1.train.GradientDescentOptimizer(
          learning_rate_input).minimize(cross_entropy_mean)
    elif FLAGS.optimizer == 'momentum':
      train_step = tf.compat.v1.train.MomentumOptimizer(
          learning_rate_input, .9,
          use_nesterov=True).minimize(cross_entropy_mean)
    elif FLAGS.optimizer == 'adam':
      train_step = tf.compat.v1.train.AdamOptimizer(
          learning_rate_input).minimize(cross_entropy_mean)
    else:
      raise Exception('Invalid Optimizer')
  predicted_indices = tf.argmax(input=logits, axis=1)
  correct_prediction = tf.equal(predicted_indices, ground_truth_input)
  confusion_matrix = tf.math.confusion_matrix(labels=ground_truth_input,
                                              predictions=predicted_indices,
                                              num_classes=model_settings['label_count'])
  evaluation_step = tf.reduce_mean(input_tensor=tf.cast(correct_prediction,
                                                        tf.float32))
  with tf.compat.v1.get_default_graph().name_scope('eval'):
    tf.compat.v1.summary.scalar('cross_entropy', cross_entropy_mean)
    tf.compat.v1.summary.scalar('accuracy', evaluation_step)

  global_step = tf.compat.v1.train.get_or_create_global_step()
  increment_global_step = tf.compat.v1.assign(global_step, global_step + 1)

  saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

  # Merge all the summaries and write them out to results/retrain_logs (by default)
  merged_summaries = tf.compat.v1.summary.merge_all(scope='eval')
  train_writer = tf.compat.v1.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                                 sess.graph)
  validation_writer = tf.compat.v1.summary.FileWriter(
      FLAGS.summaries_dir + '/validation')

  tf.compat.v1.global_variables_initializer().run()

  start_step = 1

  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)

  tf.compat.v1.logging.info('Training from step: %d ', start_step)

  # Save graph.pbtxt.
  tf.io.write_graph(sess.graph_def, FLAGS.train_dir,
                    stored_name + '.pbtxt')

  # Save list of words.
  with gfile.GFile(
      os.path.join(FLAGS.train_dir, stored_name + '_labels.txt'),
      'w') as f:
    f.write('\n'.join(audio_processor.words_list))

  # Training loop.
  performance_metrics = {'val': [], 'test': [], 'firing_rates': []}
  training_steps_max = np.sum(training_steps_list)
  for training_step in xrange(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is.
    training_steps_sum = 0
    for i in range(len(training_steps_list)):
      training_steps_sum += training_steps_list[i]
      if training_step <= training_steps_sum:
        learning_rate_value = learning_rates_list[i]
        break
    # Pull the audio samples we'll use for training.
    train_fingerprints, train_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
        FLAGS.background_volume, time_shift_samples, 'training', sess)
    # Run the graph with this batch of training data.
    train_nodes = [
            merged_summaries,
            evaluation_step,
            cross_entropy_mean,
            train_step,
            increment_global_step,
        ]

    train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
        train_nodes,
        feed_dict={
            fingerprint_input: train_fingerprints,
            ground_truth_input: train_ground_truth,
            learning_rate_input: learning_rate_value,
            dropout_prob: FLAGS.dropout_prob,
            training_placeholder: True,
        })
    train_writer.add_summary(train_summary, training_step)
    if training_step % FLAGS.print_every == 0:
        if FLAGS.model_architecture != 'lsnn':
            tf.compat.v1.logging.info(
                'Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                (training_step, learning_rate_value, train_accuracy * 100,
                 cross_entropy_value))
        else:

            tf.compat.v1.logging.info(
                'Step #%d: rate %.4f, accuracy %.1f%%, cross entropy %.3f' %
                (training_step, learning_rate_value, train_accuracy * 100, cross_entropy_value))
    is_last_step = (training_step == training_steps_max)
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      set_size = audio_processor.set_size('validation')
      total_accuracy = 0
      total_conf_matrix = None
      for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                     0.0, 0, 'validation', sess))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        val_nodes = [merged_summaries, evaluation_step, confusion_matrix]
        if FLAGS.model_architecture == 'lsnn':
            val_nodes.append(spikes)

        val_nodes_results = sess.run(
            val_nodes,
            feed_dict={
                fingerprint_input: validation_fingerprints,
                ground_truth_input: validation_ground_truth,
                dropout_prob: 1.0,
                training_placeholder: False,
            })
        if FLAGS.model_architecture == 'lsnn':
            validation_summary, validation_accuracy, conf_matrix, val_spikes = val_nodes_results
        else:
            validation_summary, validation_accuracy, conf_matrix = val_nodes_results
        validation_writer.add_summary(validation_summary, training_step)
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
          total_conf_matrix = conf_matrix
        else:
          total_conf_matrix += conf_matrix
        if FLAGS.model_architecture == 'lsnn':
            neuron_rates = np.mean(val_spikes, axis=(0, 1)) * 1000
            firing_stats = [np.mean(neuron_rates), np.min(neuron_rates), np.max(neuron_rates)]
      performance_metrics['val'].append(total_accuracy)
      tf.compat.v1.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
      tf.compat.v1.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                                (training_step, total_accuracy * 100, set_size))
      if FLAGS.model_architecture == 'lsnn':
          tf.compat.v1.logging.info('Firing rates: avg %.1f min %.1f max %.1f' %
                                    (firing_stats[0], firing_stats[1], firing_stats[2]))
          performance_metrics['firing_rates'].append('avg %.1f min %.1f max %.1f' %
                                                     (firing_stats[0], firing_stats[1], firing_stats[2]))
      with open(os.path.join(FLAGS.summaries_dir, 'performance.json'), 'w') as f:
          json.dump({**performance_metrics, 'flags': {**vars(FLAGS)}}, f, indent=4, sort_keys=True)

    # Save the model checkpoint periodically.
    if (training_step % FLAGS.save_step_interval == 0 or
        training_step == training_steps_max):
      checkpoint_path = os.path.join(FLAGS.train_dir,
                                     stored_name + '.ckpt')
      tf.compat.v1.logging.info('Saving to "%s-%d"', checkpoint_path,
                                training_step)
      saver.save(sess, checkpoint_path, global_step=training_step)

  set_size = audio_processor.set_size('testing')
  tf.compat.v1.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in xrange(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth,
            dropout_prob: 1.0,
            training_placeholder: False,
        })
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.compat.v1.logging.warn('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.compat.v1.logging.warn('Final test accuracy = %.1f%% (N=%d)' %
                            (total_accuracy * 100, set_size))
  performance_metrics['test'].append(total_accuracy)

  with open(os.path.join(FLAGS.summaries_dir, 'performance.json'), 'w') as f:
    json.dump({**performance_metrics, 'flags': {**vars(FLAGS)}}, f, indent=4, sort_keys=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is.',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How far to move in time between spectogram timeslices.',)
  parser.add_argument(
      '--feature_bin_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC / FBANK fingerprint',
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='15000,3000',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='results/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='results/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=1000,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
  parser.add_argument(
      '--quantize',
      type=bool,
      default=False,
      help='Whether to train the model for eight-bit deployment')
  parser.add_argument(
      '--preprocess',
      type=str,
      default='mfcc',
      help='Spectrogram processing mode. Can be "mfcc", "average", or "micro"')
  parser.add_argument(
      '--n_hidden',
      type=int,
      default=2048,
      help='Number of hidden units in recurrent models.')
  parser.add_argument(
      '--n_layer',
      type=int,
      default=1,
      help='Number of stacked layers in recurrent models.')
  parser.add_argument(
      '--dropout_prob',
      type=float,
      default=0.0,
      help='Dropoout probability for recurrent models.',)
  parser.add_argument(
      '--print_every',
      type=int,
      default=10,
      help='How often to print the training step results.',)
  parser.add_argument(
      '--reg',
      type=float,
      default=0.001,
      help='Firing rate regularization coefficient.',)
  parser.add_argument(
      '--n_lif_frac',
      type=float,
      default=0.0,
      help='Fraction of non-adaptive LIF neurons in LSNN.',)
  parser.add_argument(
      '--beta',
      type=float,
      default=2.,
      help='Adaptation coefficient of ALIF neurons in LSNN.',)
  parser.add_argument(
      '--comment',
      type=str,
      default='',
      help='String to append to output dir.')
  parser.add_argument(
      '--n_thr_spikes',
      type=int,
      default=-1,
      help='Number of thresholds in thr-crossing analog to spike encoding.',)
  parser.add_argument(
      '--tau',
      type=float,
      default=20.,
      help='Membrane time constant of ALIF neurons in LSNN.',)
  parser.add_argument(
      '--refr',
      type=int,
      default=2,
      help='Number of refractory time steps of ALIF neurons in LSNN.',)
  parser.add_argument(
      '--in_repeat',
      type=int,
      default=1,
      help='Number of time steps to repeat every input feature.',)
  parser.add_argument(
      '--n_delay',
      type=int,
      default=0,
      help='Maximum number of timesteps for synapse delay in LSNN.',)
  parser.add_argument(
      '--random_eprop',
      type=bool,
      default=False,
      help='Use random eprop for LSNN training')
  parser.add_argument(
      '--eprop',
      type=bool,
      default=False,
      help='Use symmetric eprop for LSNN training')
  parser.add_argument(
      '--avg_spikes',
      type=bool,
      default=True,
      help='Average spikes over time for readout')

  # Function used to parse --verbosity argument
  def verbosity_arg(value):
    """Parses verbosity argument.

    Args:
      value: A member of tf.logging.
    Raises:
      ArgumentTypeError: Not an expected value.
    """
    value = value.upper()
    if value == 'INFO':
      return tf.compat.v1.logging.INFO
    elif value == 'DEBUG':
      return tf.compat.v1.logging.DEBUG
    elif value == 'ERROR':
      return tf.compat.v1.logging.ERROR
    elif value == 'FATAL':
      return tf.compat.v1.logging.FATAL
    elif value == 'WARN':
      return tf.compat.v1.logging.WARN
    else:
      raise argparse.ArgumentTypeError('Not an expected value')
  parser.add_argument(
      '--verbosity',
      type=verbosity_arg,
      default=tf.compat.v1.logging.INFO,
      help='Log verbosity. Can be "INFO", "DEBUG", "ERROR", "FATAL", or "WARN"')
  parser.add_argument(
      '--optimizer',
      type=str,
      default='adam',
      help='Optimizer (gradient_descent, momentum, adam)')

  FLAGS, unparsed = parser.parse_known_args()
  if FLAGS.random_eprop:
      FLAGS.eprop = True
  print(json.dumps(vars(FLAGS), indent=4))
  stored_name = '{}_{}_l{}_h{}_w{}str{}_do{}'.format(
      datetime.now().strftime("%Y%m%d-%H%M%S"),
      FLAGS.model_architecture, FLAGS.n_layer, FLAGS.n_hidden, FLAGS.window_size_ms, FLAGS.window_stride_ms,
      FLAGS.dropout_prob)
  if FLAGS.model_architecture == 'lsnn':
      stored_name += '_b{}_lif{}_reg{}'.format(FLAGS.beta, FLAGS.n_lif_frac, FLAGS.reg)
  stored_name += '_{}'.format(FLAGS.comment)
  FLAGS.summaries_dir = os.path.join(FLAGS.summaries_dir, stored_name)
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
