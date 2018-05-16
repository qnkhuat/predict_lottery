import numpy as np
import math
import time
import os
import random
import tensorflow as tf
from data.utlis import *
from tensorflow.python.framework import ops
from data.convert import *
_keep_rate = 0.5
_iter = 1000
_lr = 0.0001

def create_placeholders(number_of_ref_days,all_numbers,n_C):
    X = tf.placeholder(shape=[None,number_of_ref_days,all_numbers,n_C]
                       ,dtype=tf.float32)
    Y = tf.placeholder(shape=[None,all_numbers],dtype=tf.float32)
    return X,Y


def prepare_params(X):
    m,number_of_ref_days,all_numbers,n_C = X.shape#n_C=1
    assert all_numbers == 100 ,'seems like input order is wrong'
    X,Y=create_placeholders(number_of_ref_days,all_numbers,n_C)
    return X,Y

def compute_cost(Z,Y):
        # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z,labels=Y))
    cost =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y
                                                                 ,logits =Z))
    var_summary(cost)
    return cost  # NOTE: use sigmoid instead of softmax

def conv_layer(name, X_train,in_c,out_c,filter=3,stride=2,is_max_pool=False):
    name_scope = 'conv_max' if is_max_pool else 'conv'
    with tf.name_scope(name_scope):
        with tf.name_scope('weights'):
            W = tf.get_variable('W'+name,[filter,filter,in_c,out_c]
                    ,initializer = tf.contrib.keras.initializers.he_normal()) # NOTE: Didn't provide the shape yet
            var_summary(W)
        with tf.name_scope('biases'):
            b=tf.get_variable('b'+name,[out_c]
                              ,initializer = tf.constant_initializer(0.1)) # NOTE: need to fine_tune this
            var_summary(b)

        Z = tf.nn.conv2d(X_train,W,strides=[1,stride,stride,1],padding='SAME')
        pre_activation = tf.nn.bias_add(Z,b)
        batch_norm = tf.contrib.layers.batch_norm(pre_activation, decay= 0.9
                  ,center=True,scale =True,epsilon=1e-3,updates_collections=None)
        out = tf.nn.sigmoid(batch_norm) # NOTE: use sigmoid instead of relu


        if is_max_pool:
            out = tf.nn.max_pool(out,ksize = [1,2,2,1] ,strides = [1,2,2,1]
                                 ,padding='SAME')

        return out


def fc(input , dropout ,is_dropout=True):
    output = tf.contrib.layers.flatten(input)
    if is_dropout:
        output = tf.nn.dropout(output,dropout)
    output = tf.contrib.layers.fully_connected(output,1024,activation_fn= None)
    output = tf.contrib.layers.fully_connected(output,100,activation_fn= None)
    return output

def fc_vgg(input , dropout,is_dropout=True,is_activate = True):
    with tf.name_scope('fc'):
        output = tf.contrib.layers.flatten(input)
        if is_dropout:
            output = tf.nn.dropout(output,dropout)
        if is_activate:
            output = tf.contrib.layers.fully_connected(output, out_c
                                                       , activation_fn=activate)
        else:
            output = tf.contrib.layers.fully_connected(output, out_c)

    return output



def forward_prop(X_train,dropout):
    conv = conv_layer('1',X_train,in_c=1,out_c=8)
    conv = conv_layer('2',conv,in_c=8,out_c=16,is_max_pool=True)

    conv = conv_layer('3',conv,in_c=16,out_c=32)
    conv = conv_layer('4',conv,in_c=32,out_c=64,is_max_pool=True)

    conv = conv_layer('5',conv,in_c=64,out_c=128)
    conv = conv_layer('6',conv,in_c=128,out_c=256,is_max_pool=True)

    conv = conv_layer('7',conv,in_c=256,out_c=256)
    conv = conv_layer('8',conv,in_c=256,out_c=256,is_max_pool=True)

    output = fc(conv,dropout)

    return output




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


def predict(Y,output):
    # correct_prediction = tf.equal(tf.argmax(output,1), tf.nn.k_top(Y,27))
    correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(output)), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def main():
    update('data/predict.xlsx')
    X_train= convert_dat_to_predict('data/predict.xlsx','data/predict.txt',10)

    ops.reset_default_graph()

    X,Y = prepare_params(X_train)

    keep_prob= tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)
    output = forward_prop(X,keep_prob)

    with tf.name_scope('accuracy'): # NOTE: need to chagne predict
        accuracy = predict(Y, output)
    tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
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

        for i in range(1):
            global _lr
            start_time = time.time()
            out = sess.run([output],feed_dict= {X:X_train,keep_prob:_keep_rate
                                                ,lr:_lr})
            order = np.argsort(out[0])
            print(np.sort(order[0][73:]))
            end_time =time.time()
            total_time=end_time -start_time


if __name__ == "__main__":
    main()
