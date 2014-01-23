import sift
from numpy import *
import pickle
import os
from sklearn.cluster import KMeans

train_feat_path = '../trainfeature/'
test_feat_path = '../testfeature/'
kind = ['cat.','dog.']

nkinds = 300
myMeans = KMeans(n_clusters = nkinds,n_jobs = 3)

tra_num = 4000
test_num = 2000
des = []
for k in kind:
    for i in range(0,(tra_num + test_num) / 2 - 1):
        ffile = train_feat_path + k + str(i) + '.jpg.sift'
        des.extend(sift.iter_loadtxt(ffile))

print 'got all features, now kmeans...'

a_des = array(des)
res_arr = myMeans.fit_predict(a_des)
print 'kmeans finished, now save to file'
res_fname = 'test-k300.kres'
f_res = open(res_fname,'wb')
pickle.dump(res_arr,f_res)
f_res.close()

print 'all done'


        
