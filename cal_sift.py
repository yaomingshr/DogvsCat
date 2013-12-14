from PIL import Image
from numpy import *
from pylab import *
import os
import sift
import imtools

ntrain = 25000
ntest = 12500
train_path = '../train/'
test_path = '../test1/'
train_feat_path = '../trainfeature/'
test_feat_path = '../testfeature/'
kind = ['cat.','dog.']

for i in range(0,ntrain/2):
    for k in kind:
        imname = k + str(i) + '.jpg'
        impath = train_path + imname 
        fpath = train_feat_path + imname + '.sift' 
        sift.process_image(impath,fpath)
    print 'get feature(train):' + str(i+1) + '/12,500 finished'
    
for i in range(1,ntest+1):
    imname = str(i) + '.jpg'
    impath = test_path + imname
    fpath = test_feat_path + imname + '.sift'
    sift.process_image(impath,fpath)
    print 'get feature(test):' + str(i) + '/12,500 finished'

