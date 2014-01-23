from numpy import *
import pickle
import os
import imtools
from sklearn import svm
import csv
from sklearn.ensemble import RandomForestClassifier

train_feat_path = '../trainfeature/'
kind = ['cat.','dog.']

fk_name = 'test-k300.kres'
ares_fname = 'test-k300.ares'

fk = open(fk_name,'rb')
n_kres = pickle.load(fk)
fk.close()

suml_count = 0
aRes = []
for k in kind:
    for i in range(0,2999):
        ffile = train_feat_path + k + str(i) + '.jpg.sift'
        lcount = imtools.linecount(ffile)
        kres_temp = zeros((1,300))
        for tt in range(suml_count,suml_count+lcount):
            kres_temp[0,n_kres[tt]] = kres_temp[0,n_kres[tt]] + 1
        suml_count = suml_count + lcount

        sf_kres = kres_temp.tolist()
        aRes.extend(sf_kres)
aRes = array(aRes)

clf = svm.SVC(C = 0.8)

#aRes = aRes / 2
y = append(zeros(2000),ones(2000),0)
X = append(aRes[0:2000],aRes[2999:4999],0)
clf.fit(X,y)
test_data = append(aRes[2000:2999],aRes[4999:5998],0)
test_y = append(zeros(999),ones(999),0)
test_got = clf.predict(test_data)

test_y = int8(test_y)
test_got = int8(test_got)
resCom = test_y ^ test_got
dcount = 0

for i in resCom:
    if i == 0:
        dcount = dcount + 1

cRate = dcount / 1998.0

print 'correct rate is ' + str(cRate)
