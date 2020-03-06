import numpy as np # linear algebra library
import pandas as pd # data processing library
from imread import imread, imsave
#from skimage.io import imread
from matplotlib import pyplot as plt


import os
print(os.listdir("../../../Datasets/airbus-ship-detection/"))

train = os.listdir('../../../Datasets/airbus-ship-detection/train_v2')
print(len(train))

test = os.listdir('../../../Datasets/airbus-ship-detection/test_v2')
print(len(test))

submission = pd.read_csv('../../../Datasets/airbus-ship-detection/sample_submission_v2.csv')
submission.head()

def rle_decode(mask_rle, shape=(768, 768)):
    if not isinstance(mask_rle, str):
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        return img.reshape(shape).T
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

masks = pd.read_csv('../../../Datasets/airbus-ship-detection/train_ship_segmentations_v2.csv')
masks.head()

ImageId = '7fd161e8b.jpg'

img = imread('../../../Datasets/airbus-ship-detection/train_v2/' + ImageId)
img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

#Take the individual ship masks and create a single mask array for all ships

all_masks = np.zeros((768, 768))
for mask in img_masks:
    if mask is not None:
        all_masks += rle_decode(mask)   

fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')
axarr[0].imshow(img)
axarr[1].imshow(all_masks)
axarr[2].imshow(img)
axarr[2].imshow(all_masks, alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()

#Simple explanation

#Basically encoded pixel is a list of pairs of starting pixels and amount of pixels to take in consideration

# Example:
# [1 10 2 15 3 20]

#Starting from pixel 1 take 10 pixels
#Starting from pixel 2 take 15 pixels