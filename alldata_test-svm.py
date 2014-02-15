from numpy import *
import pickle
import os
import imtools
from sklearn import svm
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

train_feat_path = '../trainfeature/'
test_feat_path = '../testfeature/'
kind = ['cat.','dog.']
m = 12500
n_clusters = 500
clu_res_fname = 'alldata-k' + str(n_clusters) + '.kres'

fk = open(clu_res_fname,'rb')
n_kres = pickle.load(fk)
fk.close()

suml_count = 0
aRes = []

#arrange train data
for k in kind:
    for i in range(0,m):
        ffile = train_feat_path + k + str(i) + '.jpg.sift'
        lcount = imtools.linecount(ffile)
        kres_temp = zeros((1,n_clusters))
        for tt in range(suml_count,suml_count+lcount):
            kres_temp[0,n_kres[tt]] = kres_temp[0,n_kres[tt]] + 1
        suml_count = suml_count + lcount

        sf_kres = kres_temp.tolist()
        aRes.extend(sf_kres)

#arrange test data
for i in range(1,m+1):
    ffile = test_feat_path + str(i) + '.jpg.sift'
    lcount = imtools.linecount(ffile)
    kres_temp = zeros((1,n_clusters))
    for tt in range(suml_count,suml_count+lcount):
        kres_temp[0,n_kres[tt]] = kres_temp[0,n_kres[tt]] + 1
    suml_count = suml_count + lcount
    sf_kres = kres_temp.tolist()
    aRes.extend(sf_kres)
    
aRes = array(aRes)        

clf = svm.SVC(C = 1.0)

X = aRes[0:25000]
y = append(zeros(12500),ones(12500),0)
clf.fit(X,y)
X_test = aRes[25000:37500]
np_res = clf.predict(X_test)

csvfile = file('final.csv','wb')
res_writer = csv.writer(csvfile)
res_writer.writerow(['id','label'])
csv_format = []
for i in range(1,12501):
    csv_format.append([i,np_res[i-1]]) 
for line in csv_format:
    res_writer.writerow(line)
csvfile.close()
