import sift
from numpy import *
import pickle
import os
from sklearn.cluster import KMeans

train_feat_path = '../trainfeature/'
test_feat_path = '../testfeature/'
kind = ['cat.','dog.']
m = 12500

des = []

for k in range(0,2):  
    for i in range(0,m):
        ffile = train_feat_path + kind[k] + str(i) + '.jpg.sift'
        if os.path.getsize(ffile) != 0:
            des.extend(sift.get_descriptor(ffile))
        print 'get des(train):' + str(k*m+i+1) + '/25,000 finished'

for i in range(1,m+1):
    ffile = test_feat_path + str(i) + '.jpg.sift'
    if os.path.getsize(ffile) != 0:
        des.extend(sift.get_descriptor(ffile))
    print 'get des(test):' + str(i) + '/12,500 finished'

#des = array(des)
#nKeypoints = des.shape[0]
#res_example = zeros(nKeypoints,)
nkinds = 100
myMeans = KMeans(n_clusters = nkinds)
res_example = myMeans.fit_predict(des)

f_res = open('kmeans.res','wb')
pickle.dump(res_example,f_res)
f_res.close()
