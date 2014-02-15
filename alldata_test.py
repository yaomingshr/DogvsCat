import sift
from numpy import *
import pickle
import os
from sklearn.cluster import KMeans

train_feat_path = '../trainfeature/'
test_feat_path = '../testfeature/'
kind = ['cat.','dog.']
m = 12500

n_clusters = 500
myMeans = KMeans(n_clusters = n_clusters, n_jobs = 3)
des = []
#get train data
for k in kind:
    for i in range(0,m):
        ffile = train_feat_path + k + str(i) + '.jpg.sift'
        des.extend(sift.iter_loadtxt(ffile))

print 'got all train data'

for i in range(1,m+1):
    ffile = test_feat_path + str(i) + '.jpg.sift'
    des.extend(sift.iter_loadtxt(ffile))
    
print 'got all test data'

feat_des = array(des)
clu_res = myMeans.fit_predict(feat_des)
#clear, save to file
des = []
clu_res_fname = 'alldata-k' + str(n_clusters) + '.kres'
f_res = open(clu_res_fname,'wb')
pickle.dump(feat_des,f_res)
f_res.close()
