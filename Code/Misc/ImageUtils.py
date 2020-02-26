"""
CMSC733 Spring 2020: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: Alohomora: Phase 2


Author(s):
Yu-Kai Wang (ykwang11@terpmail.umd.edu)
Master in Robotics,
University of Maryland, College Park
"""
import cv2
import numpy as np
from numpy.linalg import inv
import sys
import DataUtils as du


# Don't generate pyc codes
sys.dont_write_bytecode = True


def randomCrop(imageShape, patchSize):
    Tau = min(patchSize[0], patchSize[1])/3
    y = np.random.randint(imageShape[0] - (patchSize[0] + Tau)) + Tau
    x = np.random.randint(imageShape[1] - (patchSize[1] + Tau)) + Tau

    return x, y


def getCorrespondence(imageShape, label, patchSize):
    # source (random cropped) corners with (x, y) form
    x, y = randomCrop(imageShape, patchSize)
    src = np.array([(x, y),
                    (x + patchSize[1], y),
                    (x + patchSize[1], y + patchSize[0]),
                    (x, y + patchSize[0])], dtype=np.float32)

    # distination (mapped) corners with (x, y) format
    dist = np.array([(src[0, 0] + label[0], src[0, 1] + label[1]),
                     (src[1, 0] + label[2], src[1, 1] + label[3]),
                     (src[2, 0] + label[4], src[2, 1] + label[5]),
                     (src[3, 0] + label[6], src[3, 1] + label[7])], dtype=np.float32)

    return src, dist


def getWarpingPair(image, patchSize, src, dist):
    # randomly crop images
    y = int(src[0, 1])
    x = int(src[0, 0])
    crop = image[y : y + patchSize[0], x : x + patchSize[1]]
 
    # get forward and backward homographies
    Hab = cv2.getPerspectiveTransform(src, dist) # A to B
    Hba = np.linalg.inv(Hab) # B to A
    
    # warp back corners of the second image
    inv = cv2.perspectiveTransform(dist.reshape(-1, 1, 2), Hba).reshape(-1, 2) # corner is in (x, y)
    max_shape = np.max(inv, axis=0)

    # opencv's warpPerspective uses dsize uses (width, height)
    # that is fully tested
    max_shape = (image.shape[1], image.shape[0])

    # warp back the second image
    warped = cv2.warpPerspective(image, Hba, max_shape)
    patch = warped[y : y + patchSize[0], x : x + patchSize[1]]


    return crop, patch


def generateData(image, patchSize, tau, drawing=False):
    # generate label
    label = np.random.randint(2*tau, size=8) - tau

    # get corner correspondences
    src, dist = getCorrespondence(image.shape, label, patchSize)

    # get forward and backward homographies
    crop, patch = getWarpingPair(image, patchSize, src, dist)

    # NOTE: reoprt only, show image with bounding boxe
    if drawing:
        bound = image.copy()
        for i in range(4):
            cv2.circle(bound, (src[i, 0], src[i, 1]), 2, (255,0,0), 3)
            cv2.circle(bound, (dist[i, 0], dist[i, 1]), 2, (0,0,255), 3)
        cv2.polylines(bound, np.int32([src]), True, (255,0,0))
        cv2.polylines(bound, np.int32([dist]), True, (0,0,255))
        cv2.imshow("bounding boxes", bound) 

    return crop, patch, label

def StandardizeInputs(image):
    # Dummy function, should add more operations later
    return image


def testWarping():
    # Load testing data
    DataPath = "ykwang11_p1/Phase2/Data/Train/18.jpg"
    image = cv2.imread(DataPath)
    cv2.imshow("image", image)

    # cotrol the patch size
    np.random.seed(seed=100)  # for deterministic analysis
    patchSize = [32, 32, 3]
    tau = min(patchSize[0], patchSize[1])/2

    crop, patch, label = generateData(image, patchSize, tau, drawing=True)
    print("The ground truth homography is", label)

    cv2.imshow("crop", crop)
    cv2.imshow("patch", patch)
    print(crop.shape)
    print(patch.shape)
    dummy = np.concatenate([crop, patch], -1)
    print(dummy.shape)
    cv2.waitKey()

if __name__ == '__main__':
    testWarping()
