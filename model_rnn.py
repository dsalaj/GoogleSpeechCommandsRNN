import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys

storage_path = './results'
os.makedirs(storage_path, exist_ok=True)

filelabel = 'features_rnn2'
test_X = np.load('test_'+filelabel+'.npy')
print("Test Features: ",test_X.shape)
test_Y = np.load('test_labels_rnn.npy')
print("Test Labels: ",test_Y.shape)
train_X = np.load('train_'+filelabel+'.npy')
print("Train Features: ",train_X.shape)
train_Y = np.load('train_labels_rnn.npy')
print("Train Labels: ",train_Y.shape)
#sys.exit()

learning_rate = 0.00025
training_iters = 35000
batch_size = 150
display_step = 50

n_input = train_X.shape[2]
n_steps = train_X.shape[1]
n_hidden = 100
n_classes = train_Y.shape[1]
dropout = 0.5

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

weight_h = tf.Variable(tf.random_normal([n_input, n_hidden]))
bias_h = tf.Variable(tf.random_normal([n_hidden]))

def RNN(_x, weight, bias):

    _x = tf.transpose(_x, [1, 0, 2])
    _x = tf.reshape(_x, [-1, n_input])

    _x = tf.nn.relu(tf.matmul(_x, weight_h) + bias_h)
    _x = tf.split(_x, n_steps, 0)

    cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple = True)
    cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple = True)
    #cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = dropout)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1, cell_2], state_is_tuple = True)
    output, state = tf.nn.static_rnn(cell, _x, dtype = tf.float32)

    #output = tf.transpose(output, [1, 0, 2])
    #last = tf.gather(output, (int)(output.get_shape()[0]) - 1)
    last = output[-1]
    return tf.matmul(last, weight) + bias

prediction = RNN(x, weight, bias)

# Define loss and optimizer
loss_f = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_f)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    saver = tf.train.Saver()

    for itr in range(training_iters):
        offset = (itr * batch_size) % (train_Y.shape[0] - batch_size)
        batch_x = train_X[offset:(offset + batch_size), :, :]
        batch_y = train_Y[offset:(offset + batch_size), :]
        _, c = session.run([optimizer, loss_f],feed_dict={x: batch_x, y : batch_y})

        if itr % display_step == 0:
            # Calculate batch accuracy
            acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
            print ("Iter " + str(itr) + ", Minibatch Loss= " + \
                  "{}".format(loss) + ", Training Accuracy= " + \
                  "{}".format(acc))
        if itr % 1000 == 0:
            saver.save(session,os.path.join(storage_path, 'model_rnn'+ str(itr)))

    print('Test accuracy: ',session.run(accuracy, feed_dict={x: test_X, y: test_Y}))
    print('Train accuracy: ',session.run(accuracy, feed_dict={x: train_X, y: train_Y}))
    saver.save(session,os.path.join(storage_path, 'final_model_rnn'))
