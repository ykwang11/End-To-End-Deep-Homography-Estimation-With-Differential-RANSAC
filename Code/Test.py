#!/usr/bin/env python

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
import cv2
import os
import sys
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *

# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs: 
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """   
    # Image Input Shape
    ImageSize = [64, 64, 2]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.jpg'))
    SkipFactor = 1
    for count in range(1, NumImages+1, SkipFactor):
        DataPath.append(BasePath + str(count) + '.jpg')

    return ImageSize, DataPath
    
def ReadImages(ImageSize, DataPath):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    
    ImageName = DataPath
    Image = cv2.imread(ImageName)
    Gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    
    if (Image is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image cannot be read')
        sys.exit()
        
    Tau = min(ImageSize[0], ImageSize[1])/3
    # generate label
    Label = np.random.randint(2*Tau, size=8) - Tau

    # get corner correspondences
    Corners, Warped = iu.getCorrespondence(Image.shape, Label, ImageSize)

    # get forward and backward homographies
    CroppedPatch, WarpedPatch = iu.getWarpingPair(Gray, ImageSize, Corners, Warped)
    I1 = np.float32(np.stack([CroppedPatch, WarpedPatch], -1))

    #cv2.imshow("Image", Image)
    #cv2.imshow("Warped Patch", WarpedPatch)
    #cv2.imshow("Cropped Patch", CroppedPatch)

    I1S = iu.StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, Image, Label, Corners
                

def BoundingBox(Img, Corners, TrueLabel, PredLabel, count):

    TrueWarped = np.array([(Corners[0, 0] + TrueLabel[0], Corners[0, 1] + TrueLabel[1]),
                           (Corners[1, 0] + TrueLabel[2], Corners[1, 1] + TrueLabel[3]),
                           (Corners[2, 0] + TrueLabel[4], Corners[2, 1] + TrueLabel[5]),
                           (Corners[3, 0] + TrueLabel[6], Corners[3, 1] + TrueLabel[7])], dtype=np.float32)

    PredWarped = np.array([(Corners[0, 0] + PredLabel[0], Corners[0, 1] + PredLabel[1]),
                           (Corners[1, 0] + PredLabel[2], Corners[1, 1] + PredLabel[3]),
                           (Corners[2, 0] + PredLabel[4], Corners[2, 1] + PredLabel[5]),
                           (Corners[3, 0] + PredLabel[6], Corners[3, 1] + PredLabel[7])], dtype=np.float32)

    for i in range(4):
        cv2.circle(Img, (Corners[i, 0], Corners[i, 1]), 2, (255,0,0), 3)
        cv2.circle(Img, (TrueWarped[i, 0], TrueWarped[i, 1]), 2, (0,0,255), 3)
        cv2.circle(Img, (PredWarped[i, 0], PredWarped[i, 1]), 2, (0,255,255), 3)
    cv2.polylines(Img, np.int32([Corners]), True, (255,0,0))
    cv2.polylines(Img, np.int32([TrueWarped]), True, (0,0,255))
    cv2.polylines(Img, np.int32([PredWarped]), True, (0,255,255))
    cv2.imwrite("ykwang11_p1/Phase2/Output/%d.png" %count, Img)
    #cv2.imshow("result", Img)
    #cv2.waitKey()



def TestOperation(ImgPH, ImageSize, ModelPath, DataPath):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    Length = ImageSize[0]
    # Predict output with forward pass, MiniBatchSize for Test is 1
    H4Pt, _ = HomographyModel(ImgPH, ImageSize, 1)

    # Setup Saver
    Saver = tf.train.Saver()

    errorSum = 0
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        for count in tqdm(range(np.size(DataPath))):            
            DataPathNow = DataPath[count]
            Img, ImgOrg, Label, Corners = ReadImages(ImageSize, DataPathNow)

            FeedDict = {ImgPH: Img}
            Pred = sess.run(H4Pt, FeedDict)
            #print ("pred ", Pred)
            #print ("label ", Label)
            #print ("---------> ", Corners)
            errorSum += np.aqrt(np.mean((Pred - Label)*(Pred - Label)))
            #print ("---------> ", Pred)
            BoundingBox(ImgOrg, Corners, Label, Pred.reshape(-1), count)



    print('The average 2-norm error is %f ' % (errorSum / np.size(DataPath)))

def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--CheckPointPath', default='ykwang11_p1/Phase2/Checkpoints/', help='Path to save Checkpoints, Default: ykwang11_p1/Phase2/Checkpoints/')
    Parser.add_argument('--BasePath', dest='BasePath', default='ykwang11_p1/Phase2/Data/Val/', help='Path to load images from, Default:BasePath')
    Args = Parser.parse_args()
    ModelType = Args.ModelType
    CheckPointPath = Args.CheckPointPath + ModelType + '/'
    BasePath = Args.BasePath

    # Load lastest checkpoint
    LatestFile = FindLatestModel(CheckPointPath)
    ModelPath =  CheckPointPath + LatestFile + '.ckpt'

    # Setup all needed parameters including file reading
    ImageSize, DataPath = SetupAll(BasePath)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], ImageSize[2]))

    print(ModelPath)
    print(DataPath)
    TestOperation(ImgPH, ImageSize, ModelPath, DataPath)
     
if __name__ == '__main__':
    main()
 
