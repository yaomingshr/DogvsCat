from PIL import Image
from numpy import *
from pylab import *
import os
import sift
import imtools

testim = 'test.jpg'
rspath = 'test.sift'

im = sift.process_image(testim,rspath,"--edge-thresh 10 --peak-thresh 8")
locs,decs = sift.read_features_from_file(rspath)
sift.plot_features(im,locs)
