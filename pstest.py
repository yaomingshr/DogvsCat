from PIL import Image
from numpy import *
from pylab import *
import os
import sift
import imtools
import pickle

#testim = 'test.jpg'
#rspath = 'test0.sift'

#im = sift.process_image(testim,rspath,"--edge-thresh 5 --peak-thresh 10")
#locs,decs = sift.read_features_from_file(rspath)
#sift.plot_features(im,locs)
#lcount = imtools.linecount(rspath)
#print lcount

testf = 'train2-test2.ares'
fobj = open(testf,'rb')
res = pickle.load(fobj)
print res.shape
