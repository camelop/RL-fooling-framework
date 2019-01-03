from skimage.measure import *
import numpy as np 
x = np.array([[1,2],[3,4]])
# return a punish between 0-1
def punish_ssim(im1, im2):
    result = compare_ssim(im1.astype(np.float64), im2.astype(np.float64))
    return result