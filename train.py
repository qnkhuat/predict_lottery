import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import math
import time
import os
import random
import data_process as dp

_keep_rate = 0.5
_iter = 1000
_lr = 0.0001


def create_placeholders(n_H, n_W, n_C, n_y):
    X = tf.placeholder(shape=[None, n_H, n_W, n_C], dtype=tf.float32)
    Y = tf.placeholder(shape=[None, n_y], dtype=tf.float32)

    return X, Y


def prepare_params(X):
    m_train, n_H_train, n_W_train, n_C_train = X.shape
    X, Y = create_placeholders(n_H_train, n_W_train, n_C_train, 10)

    return X, Y


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost


def var_summary(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def conv(name, X_train, in_c, out_c, is_max_pool=False):
    name_scope = 'conv_max' if is_max_pool else 'conv'

    with tf.name_scope(name_scope):
        with tf.name_scope('weights'):
            W = tf.get_variable('W' + name, [4, 4, in_c, out_c], initializer=tf.contrib.keras.initializers.he_normal())
            var_summary(W)
        with tf.name_scope('biases'):
            b = tf.get_variable('b' + name, [out_c], initializer=tf.constant_initializer(0.1))
            var_summary(b)

        Z = tf.nn.conv2d(X_train, W, strides=[1, 2, 2, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(Z, b)
        batch_norm = tf.contrib.layers.batch_norm(pre_activation, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                                  updates_collections=None)
        out = tf.nn.relu(batch_norm)

        if is_max_pool:
            out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return out


def fc(input,dropout,is_dropout=True):
    output = tf.contrib.layers.flatten(input)
    if is_dropout:
        output = tf.nn.dropout(output,dropout)
    output = tf.contrib.layers.fully_connected(output, 1024, activation_fn=None)
    output = tf.contrib.layers.fully_connected(output, 10, activation_fn=None)
    return output


def fc_vgg(activation,out_c, dropout, is_dropout=True,is_activate=True):
    if is_activate:
        activate=tf.nn.relu
    else:
        activate=None

    with tf.name_scope('fc'):
        output = tf.contrib.layers.flatten(activation)
        if is_dropout:
            output = tf.nn.dropout(output, dropout)
        output = tf.contrib.layers.fully_connected(output, out_c, activation_fn=activate)
    return output

def forward_prop(X_train, dropout):
    conv1 = conv('W1', X_train, 3, 8)
    conv2 = conv('W2', conv1, 8, 16, is_max_pool=True)

    conv3 = conv('W3', conv2, 16, 32)
    conv4 = conv('W4', conv3, 32, 64, is_max_pool=True)

    conv5 = conv('W5', conv4, 64, 128)
    conv6 = conv('W6', conv5, 128, 256, is_max_pool=True)

    output = fc(conv6, dropout)

    # convs = conv('7', convs, 256, 512)
    # convs = conv('8', convs, 512, 512, is_max_pool=True)
    #
    # convs = conv('9', convs, 512, 512)
    # convs = conv('10', convs, 512, 512, is_max_pool=True)


    # output = fc(conv6, dropout)
    #
    # fc1 = fc_vgg(convs,4096,dropout)
    # fc2 = fc_vgg(fc1, 4096, dropout)

    # fc3 = fc(convs, dropout)
    # output=tf.nn.softmax(fc3)
    return output


def predict(Y, output):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return accuracy


def main():

    X_train_origin, Y_train_origin, X_test_origin, Y_test_origin = dp.loadData()
    X_train_origin,X_test_origin=dp.data_preprocessing(X_train_origin,X_test_origin)

    ops.reset_default_graph()

    X, Y = prepare_params(X_train_origin)

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    output = forward_prop(X, keep_prob)
    cost = compute_cost(output, Y)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.name_scope('accuracy'):
        accuracy = predict(Y, output)

    tf.summary.scalar('accuracy', accuracy)

    m = X_train_origin.shape[0]
    minibatch_size = 32

    # load data from txt
    cache_files_name = ['data/costs.txt', 'data/trains.txt', 'data/tests.txt']
    dp.ensure_dir(cache_files_name)
    costs, trains, tests = dp.load_txt(cache_files_name)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(init)
        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter('graph/2' + '/train',
                                             sess.graph)
        test_writer = tf.summary.FileWriter('graph/2' + '/test')

        try:
            ckpt = tf.train.get_checkpoint_state('./checkpoint/')
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Haha I found a checkpoint.")
        except:
            print("No checkpoint found.")

        for i in range(_iter):
            global _lr
            if i>=10:
                lr=_lr/2
            elif i>=100:
                lr=_lr=10
            else :
                lr = _lr

            start_time = time.time()

            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            minibatches = dp.random_mini_batches(X_train_origin, Y_train_origin, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                minibatch_X = dp.data_augmentation(minibatch_X)
                _, temp_cost = sess.run([optimizer, cost],feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob: _keep_rate,learning_rate:lr})

                minibatch_cost += temp_cost / num_minibatches

            costs = np.append(costs, minibatch_cost)

            end_time = time.time()
            total_time = end_time - start_time

            if i % 10 == 0:
                summary, train_accuracy = sess.run([merged, accuracy], feed_dict={X: X_train_origin, Y: Y_train_origin,
                                                                                  keep_prob: _keep_rate})
                train_writer.add_summary(summary, i)

                summary, test_accuracy = sess.run([merged, accuracy],
                                                  feed_dict={X: X_test_origin, Y: Y_test_origin, keep_prob: 1.0})
                test_writer.add_summary(summary, i)

                trains, tests = dp.append_data(trains, tests, train_accuracy, test_accuracy)

                # save data to txt
                dp.save_txt(costs, trains, tests, cache_files_name)
                print("cost after {} iters : {} in {} each with train accuracy = {} and test accuracy = {} ".format(i,
                                                                                                                    minibatch_cost,
                                                                                                                    total_time,
                                                                                                                    train_accuracy,
                                                                                                                    test_accuracy))

                saver.save(sess, "./checkpoint/model.ckpt", global_step=1)

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)


if __name__ == "__main__":
    main()
