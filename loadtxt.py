import os
from numpy import * 
import sift

fname = 'test0.sift'
tt = sift.iter_loadtxt(fname)
print tt
