import numpy as np # linear algebra library
import pandas as pd # data processing library
from imread import imread, imsave
#from skimage.io import imread
import matplotlib.pyplot as pyplot

import os
print(os.listdir("../../Datasets/airbus-ship-detection/"))

train = os.listdir('../../Datasets/airbus-ship-detection/train_v2')
print(len(train))

test = os.listdir('../../Datasets/airbus-ship-detection/test_v2')
print(len(test))

submission = pd.read_csv('../../Datasets/airbus-ship-detection/sample_submission_v2.csv')
submission.head()

def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

masks = pd.read_csv('../../Datasets/airbus-ship-detection/train_ship_segmentations_v2.csv')
masks.head()

ImageId = '25e25d47f.jpg'

img = imread('../../Datasets/airbus-ship-detection/train_v2' + ImageId)
img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

#Take the individual ship masks and create a single mask array for all ships


#Simple explanation

#Basically encoded pixel is a list of pairs of starting pixels and amount of pixels to take in consideration

# Example:
# [1 10 2 15 3 20]

#Starting from pixel 1 take 10 pixels
#Starting from pixel 2 take 15 pixels