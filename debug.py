import sift
from numpy import *
import os
from sklearn.cluster import KMeans

ffile = 'problem.sift'
des = sift.get_descriptor(ffile)
f = loadtxt(ffile)
print f

