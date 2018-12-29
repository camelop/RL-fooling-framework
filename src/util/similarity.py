from skimage.measure import *
import numpy as np 

# return a punish between 0-1
def punish_ssim(im1, im2):
    return compare_ssim(im1, im2)
