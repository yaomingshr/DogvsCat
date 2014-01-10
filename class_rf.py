import os
from sklearn import svm
import csv
import pickle
from numpy import *
from sklearn.ensemble import RandomForestClassifier

p_list = [zeros((4000,3)),zeros((4000,3)),zeros((4500,3))]
rfX_list = [zeros((8000,3)),zeros((8000,3)),zeros((9000,3))]
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
        test_data = p_temp[boftt:]
        
        p_list[test_iter][:,train_iter] = clf.predict(test_data)
        rfX_list[train_iter][:,test_iter] = clf.predict(p_temp[0:boftt])
        print 'finish' + str(train_iter) + str(test_iter)
        

rfX = []
enX = []
rfc = RandomForestClassifier(n_estimators=400)
for rlist in rfX_list:
    rfX.extend(rlist)
for plist in p_list:
    enX.extend(plist)
rfX = array(rfX)
enX = array(enX)

ytmp = append(zeros(4000),ones(4000))
ytmp = append(ytmp,zeros(4000))
ytmp = append(ytmp,ones(4000))
ytmp = append(ytmp,zeros(4500))
y = append(ytmp,ones(4500))
y = int8(y)

rfc.fit(rfX,y)
np_res = rfc.predict(enX)

    

csvfile = file('final.csv','wb')
res_writer = csv.writer(csvfile)
res_writer.writerow(['id','label'])
csv_format = []
for i in range(1,12501):
    csv_format.append([i,np_res[i-1]]) 
for line in csv_format:
    res_writer.writerow(line)
csvfile.close()
