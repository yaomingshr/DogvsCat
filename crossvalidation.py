import os
from sklearn import svm
import csv
import pickle
from numpy import *

clf = svm.SVC(C = 0.8)
for train_iter in range(0,3):
    for test_iter in range(0,3):
        fname = 'train' + str(train_iter) + '-test' + str(test_iter) + '.ares'
        ffile = open(fname,'rb')
        p_temp = pickle.load(ffile)
        #normalization
        #print p_temp.dtype float64
        #print p_temp
        p_temp = p_temp / 2
        ffile.close()
        boftt = 8000
        cvNum = 1200
        if train_iter == 2:
            boftt = 9000
            cvNum = 1500
        capacity = boftt - cvNum * 2
        X = append(p_temp[cvNum:boftt/2],p_temp[boftt/2 + cvNum:boftt],0)
        y = append(zeros(capacity / 2),ones(capacity /2),0)
#        print p_temp.shape,X.shape,y.shape
        clf.fit(X,y)
        cvX = append(p_temp[0:cvNum],p_temp[boftt / 2:boftt / 2 + cvNum],0)
        cvy = append(zeros(cvNum),ones(cvNum),0)
        gety = clf.predict(cvX)
        cvy = int8(cvy)
        gety = int8(gety)
        cvRes = cvy ^ gety
        dcount = 0
        for i in cvRes:
            if i == 0:
                dcount = dcount + 1
        rRate = dcount / (cvNum * 2.0)
        print 'svm' + str(train_iter) + str(test_iter) + ':' + str(rRate)
        exit(0)
