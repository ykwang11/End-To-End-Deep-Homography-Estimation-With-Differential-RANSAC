#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import tensorflow as tf
import cv2
import sys
import os
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel
from Network.Network import TensorDLT
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *


# Don't generate pyc codes
sys.dont_write_bytecode = True

    
def GenerateBatch(BasePath, DirNamesTrain, ImageSize, LargeImgSize, MiniBatchSize):
    """
    Inputs: 
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    Tau = min(ImageSize[0], ImageSize[1])/3

    # Share the random crop across the batch
    x, y = iu.randomCrop((LargeImgSize[0], LargeImgSize[1]), ImageSize)
    Corners = np.array([(x, y),
                        (x + ImageSize[1], y),
                        (x + ImageSize[1], y + ImageSize[0]),
                        (x, y + ImageSize[0])], dtype=np.float32)

    # Get indices for the patch
    Indices = np.arange(0, LargeImgSize[0] * LargeImgSize[1])
    Indices = Indices.reshape(LargeImgSize[0], LargeImgSize[1])
    Indices = Indices[y : y+ImageSize[0], x : x+ImageSize[1]]

    I1Batch = []
    ImgOrgBatch = []
    LabelBatch = []
    CornerBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain)-1)
        
        RandImageName = BasePath + "/Data" + os.sep + DirNamesTrain[RandIdx] + '.jpg'   
        ImageNum += 1
        Image = cv2.imread(RandImageName, 0)
        Image = cv2.resize(Image, (LargeImgSize[1], LargeImgSize[0]))

        Label = np.random.randint(2*Tau, size=8) - Tau

        # distination (mapped) corners with (x, y) format
        Warped = np.array([(Corners[0, 0] + Label[0], Corners[0, 1] + Label[1]),
                           (Corners[1, 0] + Label[2], Corners[1, 1] + Label[3]),
                           (Corners[2, 0] + Label[4], Corners[2, 1] + Label[5]),
                           (Corners[3, 0] + Label[6], Corners[3, 1] + Label[7])], dtype=np.float32)

        CroppedPatch, WarpedPatch = iu.getWarpingPair(Image, ImageSize, Corners, Warped)

        #print('c', CroppedPatch.shape)
        #print('w', WarpedPatch.shape)

        #cv2.imshow("Image", Image)
        #cv2.imshow("Warped Patch", WarpedPatch)
        #cv2.imshow("Cropped Patch", CroppedPatch)

        I1 = np.float32(np.stack([CroppedPatch, WarpedPatch], -1))

        # Append All Images and Mask
        I1Batch.append(I1)
        ImgOrgBatch.append(Image[..., None])
        LabelBatch.append(Label)
        CornerBatch.append(Corners.reshape(8))

        #cv2.waitKey()
        
    return I1Batch, ImgOrgBatch, LabelBatch, CornerBatch, Indices


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    
def TrainOperation(ImgPH, ImgOrgPH, LabelPH, CornerPH, IndicesPH,
                   DirNamesTrain, NumTrainSamples, ImageSize, LargeImgSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    LabelPH is the one-hot encoded label placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
    ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """      
    # Predict output with forward pass
    with tf.name_scope('HomographyNet'):
        H4Pt, _ = HomographyModel(ImgPH, ImageSize, MiniBatchSize)

    with tf.name_scope('Supervised'):
        with tf.name_scope('Loss'):
            # lossSup = tf.norm(LabelPH - H4Pt, ord=2)
            lossSup = tf.nn.l2_loss(LabelPH - H4Pt) / MiniBatchSize
        with tf.name_scope('Adam'):
            OptimizerSup = tf.train.AdamOptimizer(learning_rate=1e-2)
            gradsSup = OptimizerSup.compute_gradients(lossSup)
            cappedGradsSup = [(tf.clip_by_value(grad, -20., 20.), var) for grad, var in gradsSup]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                TrainStepSup = OptimizerSup.apply_gradients(cappedGradsSup)


    with tf.name_scope('Unsupervised'):
        with tf.name_scope('TensorDLT'):
            #H = tf.reshape(TensorDLT(CornerPH, LabelPH), (-1, 3, 3))
            H = tf.reshape(TensorDLT(CornerPH, H4Pt), (-1, 3, 3))

        with tf.name_scope('SpatialTransformer'):
            # Normalized inverse computation Hinv
            M = np.array([[LargeImgSize[1]/2.0, 0., LargeImgSize[1]/2.0],
                          [0., LargeImgSize[0]/2.0, LargeImgSize[0]/2.0],
                          [0., 0., 1.]]).astype(np.float32)
            M = tf.expand_dims(tf.constant(M, tf.float32), axis=0) 
            M_inv = tf.linalg.pinv(M)
            H = tf.matmul(tf.matmul(M_inv, H), M)
            # Transform image 1 (large image) to image 2
            outSize = (LargeImgSize[0], LargeImgSize[1])
            predImg, _ = transformer(ImgOrgPH, H, outSize)
            #checkout = predImg # TODO


        with tf.name_scope('Cropping'):
            y_t = tf.range(0, MiniBatchSize * LargeImgSize[0] * LargeImgSize[1], LargeImgSize[0] * LargeImgSize[1])
            z = tf.tile(tf.expand_dims(y_t,[1]), [1, ImageSize[0] * ImageSize[1]])
            z = tf.reshape(z, (-1, ImageSize[0], ImageSize[1]))
            cropIndices = z + IndicesPH
            cropIndices_flat = tf.reshape(cropIndices, [-1])
            predImg = tf.reshape(predImg, [-1])
            predPatch = tf.gather(predImg, cropIndices_flat)
            predPatch = tf.reshape(predPatch, [-1, ImageSize[0], ImageSize[1]])
            #checkout2 = predPatch

            warpedPatch = ImgPH[..., 1]
            #checkout3 = warpedPatch


        with tf.name_scope('Loss'):
            lossUnsup = tf.reduce_mean(tf.abs(warpedPatch - predPatch))

        with tf.name_scope('Adam'):
            OptimizerUnsup = tf.train.AdamOptimizer(learning_rate=1e-5)
            gradsUnsup = OptimizerUnsup.compute_gradients(lossUnsup)
            cappedGradsUnsup = [(tf.clip_by_value(grad, -20., 20.), var) for grad, var in gradsUnsup]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                TrainStepUnsup = OptimizerUnsup.apply_gradients(cappedGradsUnsup)
        
    # Tensorboard
    # Create a summary to monitor loss tensor
    tf.summary.scalar('SupervisedLossEveryIter', lossSup)
    tf.summary.scalar('UnsupervisedLossEveryIter', lossUnsup)
    # tf.summary.image('Anything you want', AnyImg)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()

    # Setup Saver
    Saver = tf.train.Saver()
    
    np.random.seed(seed=0)  # Deterministic analysis
    with tf.Session() as sess:       
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())

        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                I1Batch, ImgOrgBatch, LabelBatch, CornerBatch, Indices = GenerateBatch(BasePath, DirNamesTrain, ImageSize, LargeImgSize, MiniBatchSize)
                FeedDict = {ImgPH: I1Batch, ImgOrgPH: ImgOrgBatch, LabelPH: LabelBatch, CornerPH: CornerBatch, IndicesPH: Indices}

                if ModelType is 'Sup':
                    _, LossThisBatch, Summary = sess.run([TrainStepSup, lossSup, MergedSummaryOP], feed_dict=FeedDict)
                else:
                    _, LossThisBatch, Summary = sess.run([TrainStepUnsup, lossUnsup, MergedSummaryOP], feed_dict=FeedDict)
                    '''
                    Pred, predpatch, patch, LossThisBatch, Summary = sess.run([checkout, checkout2, checkout3, lossUnsup, MergedSummaryOP], feed_dict=FeedDict)
                    cv2.imshow('Image', ImgOrgBatch[0])
                    cv2.imshow('PA', (I1Batch[0]).astype(np.uint8)[..., 0])
                    cv2.imshow('PB', (I1Batch[0]).astype(np.uint8)[..., 1])
                    cv2.imshow("warped image", np.array(Pred).astype(np.uint8)[0])
                    cv2.imshow("Predict PB", np.array(predpatch).astype(np.uint8)[0])
                    cv2.imshow("Place Holder PB", np.array(patch).astype(np.uint8)[0])
                    cv2.waitKey()
                    '''

                # Save checkpoint every some SaveCheckPoint's iterations
                if PerEpochCounter % SaveCheckPoint == 0:
                    # Save the Model learnt in this epoch
                    #SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                    #Saver.save(sess,  save_path=SaveName)
                    #print('\n' + SaveName + ' Model Saved...')
                    print('\n' + 'Loss This Batch is %f' %LossThisBatch)

                # Tensorboard
                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()

            # Save model every epoch
            #SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            SaveName = CheckPointPath + str(50) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print('\n' + SaveName + ' Model Saved...')
            

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='ykwang11_p1/Phase2', help='Base path of images, Default: ykwang11_p1/Phase2')
    Parser.add_argument('--CheckPointPath', default='ykwang11_p1/Phase2/Checkpoints/', help='Path to save Checkpoints, Default: ykwang11_p1/Phase2/Checkpoints/')
    Parser.add_argument('--ModelType', default='Sup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=128, help='Size of the MiniBatch to use, Default:128')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='ykwang11_p1/Phase2/Logs/', help='Path to save Logs for Tensorboard, Default: ykwang11_p1/Phase2/Logs/')

    Args = Parser.parse_args()
    ModelType = Args.ModelType
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath + ModelType + '/'
    LogsPath = Args.LogsPath + ModelType + '/'


    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, NumClasses = SetupAll(BasePath, CheckPointPath)
    LargeImgSize = [120, 180, 1]


    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
    CornerPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, NumClasses))
    LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, NumClasses))  # Four courners
    ImgOrgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, LargeImgSize[0], LargeImgSize[1], LargeImgSize[2]))
    IndicesPH = tf.placeholder(tf.int32, shape=(ImageSize[0], ImageSize[1]))

    TrainOperation(ImgPH, ImgOrgPH, LabelPH, CornerPH, IndicesPH, 
                   DirNamesTrain, NumTrainSamples, ImageSize, LargeImgSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType)
        
    
if __name__ == '__main__':
    main()
 
