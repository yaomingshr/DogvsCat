import sift
from numpy import *
import pickle
import os
from sklearn.cluster import KMeans

train_feat_path = '../trainfeature/'
test_feat_path = '../testfeature/'
kind = ['cat.','dog.']
m = 12500
m_by_iter = 4000

nkinds = 80
myMeans = KMeans(n_clusters = nkinds)

for iter_i in range(1,9):  # should be (0,9),but the first we already have
    des = []
    a_des = []
    train_iter = iter_i / 3
    test_iter = iter_i % 3
    
    train_L = m_by_iter * train_iter
    test_L = m_by_iter * test_iter + 1

    if train_iter == 2:
        train_R = m
    else:
        train_R = m_by_iter * (train_iter + 1)

    if test_iter == 2:
        test_R = m + 1
    else:
        test_R = m_by_iter * (test_iter + 1) + 1

    for k in range(0,2):  
        for i in range(train_L,train_R):
            ffile = train_feat_path + kind[k] + str(i) + '.jpg.sift'
            des.extend(sift.iter_loadtxt(ffile))
            print 'got train' + str(train_iter)
            #print 'get des(train):' + str(k*m+i+1) + '/25000 finished'
            
    for i in range(test_L,test_R):
        ffile = test_feat_path + str(i) + '.jpg.sift'
        des.extend(sift.iter_loadtxt(ffile))
        print 'got test' + str(test_iter)
        #print 'get des(test):' + str(i) + '/12500 finished'


    a_des = array(des)
    res_arr = myMeans.fit_predict(a_des)
    res_fname = 'train'+str(train_iter)+'-test'+str(test_iter)+'.kres'
    f_res = open(res_fname,'wb')
    pickle.dump(res_arr,f_res)
    f_res.close()
