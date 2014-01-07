from numpy import *
import pickle
import os
import imtools

train_feat_path = '../trainfeature/'
test_feat_path = '../testfeature/'
kind = ['cat.','dog.']
m = 12500
m_by_iter = 4000

for train_iter in range(0,3):
    for test_iter in range(0,3):

        fk_name = 'train' + str(train_iter) + '-test' + str(test_iter) + '.kres'
        ares_fname = 'train' + str(train_iter) + '-test' + str(test_iter) + '.ares'
        fk = open(fk_name,'rb');
        n_kres = pickle.load(fk)
        suml_count = 0

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

        partRes = []
            
        for k in range(0,2):
            for i in range(train_L,train_R):
                ffile = train_feat_path + kind[k] + str(i) + '.jpg.sift'
                lcount = imtools.linecount(ffile)
                kres_temp = zeros((1,80))
                for tt in range(suml_count,suml_count+lcount):
                    kres_temp[0,n_kres[tt]] = kres_temp[0,n_kres[tt]] + 1
                suml_count = suml_count + lcount

                sf_kres = kres_temp.tolist()
                partRes.extend(sf_kres)

        for i in range(test_L,test_R):
            ffile = test_feat_path + str(i) + '.jpg.sift'
            lcount = imtools.linecount(ffile)
            kres_temp = zeros((1,80))
            for tt in range(suml_count,suml_count+lcount):
                kres_temp[0,n_kres[tt]] = kres_temp[0,n_kres[tt]] + 1
            suml_count = suml_count + lcount
            sf_kres = kres_temp.tolist()
            partRes.extend(sf_kres)
            
        partRes = array(partRes)
        ares_f = open(ares_fname,'wb')
        pickle.dump(partRes,ares_f)
        ares_f.close()
