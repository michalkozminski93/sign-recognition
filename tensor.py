import os
import cv2
import numpy as np
import tensorflow as tf


class ROI:
    def __init__(self, x, y, w, h, mask):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.imgField = w*h
        self.mask = mask
        self.shape = "undefined"


#hello = tf.constant("halo halo")
#sess = tf.Session()
#print (sess.run(hello))

blabla = ROI(1,2,3,4,"red") 
print (blabla.shape)
blabla.shape = "circle"
print (blabla.shape)