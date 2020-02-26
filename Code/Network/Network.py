"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def HomographyModel(Img, ImageSize, MiniBatchSize, isTraining=True):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    # Define layers
    ClassNum = 8
    # Network 1
    Conv1 = tf.layers.Conv2D(filters=64, kernel_size=[3, 3], activation=tf.nn.relu)
    Conv2 = tf.layers.Conv2D(filters=64, kernel_size=[3, 3], activation=tf.nn.relu)
    Conv3 = tf.layers.Conv2D(filters=128, kernel_size=[3, 3], activation=tf.nn.relu)
    Conv4 = tf.layers.Conv2D(filters=128, kernel_size=[3, 3], activation=tf.nn.relu)
    Conv5 = tf.layers.Conv2D(filters=32, kernel_size=[3, 3], activation=tf.nn.relu)
    Conv6 = tf.layers.Conv2D(filters=32, kernel_size=[3, 3], activation=tf.nn.relu)
   
    Dense1 = tf.layers.Dense(units=1024)
    Dense2 = tf.layers.Dense(units=ClassNum)

    # Construct the graph
    conv1 = Conv1.apply(Img)
    bn1 = tf.layers.batch_normalization(conv1, training=isTraining)
    conv2 = Conv2.apply(bn1)
    bn2 = tf.layers.batch_normalization(conv2, training=isTraining)
    bn2 = tf.layers.max_pooling2d(inputs=bn2, pool_size=2, strides=2) # for network 2
    conv3 = Conv3.apply(bn2)
    bn3 = tf.layers.batch_normalization(conv3, training=isTraining)
    conv4 = Conv4.apply(bn3)
    bn4 = tf.layers.batch_normalization(conv4, training=isTraining)
    bn4 = tf.layers.max_pooling2d(inputs=bn4, pool_size=2, strides=2)
    conv5 = Conv5.apply(bn4)
    bn5 = tf.layers.batch_normalization(conv5, training=isTraining)
    conv6 = Conv6.apply(bn5)
    bn6 = tf.layers.batch_normalization(conv6, training=isTraining)
    bn6 = tf.contrib.layers.flatten(bn6)
    # output = tf.reshape(bn4, [-1, bn4.get_shape()[1]*bn4.get_shape()[2]*bn4.get_shape()[3]])
    output = Dense1.apply(bn6)
    H4Pt = Dense2.apply(output)
    #prSoftMax = tf.nn.softmax(prLogits)
   
    return H4Pt, H4Pt

def HomographyModelResNet(Img, ImageSize, MiniBatchSize, isTraining=True):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    # Define layers
    ClassNum = 8
    Conv0 = tf.layers.Conv2D(filters=32, kernel_size=[7, 7], padding='same')
    Conv1 = tf.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same')
    Conv2 = tf.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same')
    Conv3 = tf.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same')
    Conv4 = tf.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same')
    Conv5 = tf.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same')
    Conv6 = tf.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same')
    Conv7 = tf.layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same')
    Conv8 = tf.layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same')
    Dense1 = tf.layers.Dense(units=ClassNum)

    # Construct the graph
    conv0 = Conv0.apply(Img)
    bn0 = tf.layers.batch_normalization(conv0, training=isTraining)
    a0 = tf.nn.relu(bn0)

    
    pool = tf.nn.max_pool2d(a0, ksize=[3, 3], strides=2, padding='SAME')

    conv1 = Conv1.apply(pool)
    bn1 = tf.layers.batch_normalization(conv1, training=isTraining)
    a1 = tf.nn.relu(bn1)

    conv2 = Conv2.apply(a1) + conv1
    bn2 = tf.layers.batch_normalization(conv2, training=isTraining)
    a2 =  tf.nn.relu(bn2)

    conv3 = Conv3.apply(a2)
    bn3 = tf.layers.batch_normalization(conv3, training=isTraining)
    a3 =  tf.nn.relu(bn3)
    
    conv4 = Conv4.apply(a3) + conv3
    bn4 = tf.layers.batch_normalization(conv4, training=isTraining)
    a4 =  tf.nn.relu(bn4)

    conv5 = Conv5.apply(a4)
    bn5 = tf.layers.batch_normalization(conv5, training=isTraining)
    a5 =  tf.nn.relu(bn5)
    
    conv6 = Conv6.apply(a5) + conv5
    bn6 = tf.layers.batch_normalization(conv6, training=isTraining)
    a6 =  tf.nn.relu(bn6)

    conv7 = Conv7.apply(a6)
    bn7 = tf.layers.batch_normalization(conv7, training=isTraining)
    a7 =  tf.nn.relu(bn7)

    conv8 = Conv8.apply(a7) + conv7
    bn8 = tf.layers.batch_normalization(conv8, training=isTraining)
    avg = tf.reduce_mean(bn8, axis=[1,2])

    H4Pt = Dense1.apply(avg)
    #prSoftMax = tf.nn.softmax(prLogits)
   
    return H4Pt, H4Pt


def TensorDLT(Corners, Pred):

    print('Corners, ', Corners)
    print('Prediction, ', Pred)

    zero = tf.constant(0,  dtype=tf.float32, shape=[Corners.shape[0]])
    one = tf.constant(1,  dtype=tf.float32, shape=[Corners.shape[0]])
    #one = tf.ones((Corners.shape[0]))

    A_row = []
    b_row = []
    for i in range(0, 8, 2):
        u1 = Corners[:, i]
        v1 = Corners[:, i+1]
        u2 = Corners[:, i] + Pred[:, i]
        v2 = Corners[:, i+1] + Pred[:, i+1]
        v2u1 = tf.multiply(v2, u1)
        v2v1 = tf.multiply(v2, v1)
        u2u1 = tf.multiply(u2, u1)
        u2v1 = tf.multiply(u2, v1)

        At = tf.stack([zero, zero, zero, -u1, -v1, -one, v2u1, v2v1], axis=-1)
        Ad = tf.stack([u1, v1, one, zero, zero, zero, -u2u1, -u2v1], axis=-1)
        A_row.append(At)
        A_row.append(Ad)
        b_row.append(-v2)
        b_row.append(u2)

    A = tf.stack(A_row, axis=1)
    b = tf.stack(b_row, axis=1)

    # Either way solve the equation AH = b
    # inv = tf.linalg.pinv(A)
    # H = tf.matmul(inv, tf.expand_dims(b, axis=-1))
    H = tf.reshape(tf.matrix_solve(A, tf.expand_dims(b, axis=-1)), (-1, 8))
    H = tf.concat([H, tf.expand_dims(one, axis=-1)], axis=-1)

    return H
