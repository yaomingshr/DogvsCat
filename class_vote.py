import os
from sklearn import svm
import csv
import pickle
from numpy import *

p_list = [zeros((4000,3)),zeros((4000,3)),zeros((4500,3))]
clf = svm.SVC()
for train_iter in range(0,3):
    for test_iter in range(0,3):
        fname = 'train' + str(train_iter) + '-test' + str(test_iter) + '.ares'
        ffile = open(fname,'rb')
        p_temp = pickle.load(ffile)
        ffile.close()
        boftt = 8000
        if train_iter == 2:
            boftt = 9000

        X = p_temp[0:boftt]
        y = append(zeros(boftt / 2),ones(boftt /2))
        test_data = p_temp[boftt:]
        clf.fit(X,y)
        p_list[test_iter][:,train_iter] = clf.predict(test_data)

for i in range(0,3):
    p_list[i] = p_list[i][:,0] + p_list[i][:,1] + p_list[i][:,2]
    p_list[i] = int8(p_list[i])
    p_list[i] = p_list[i] / 2

np_tmp = append(p_list[0],p_list[1])
np_res = append(np_tmp,p_list[2])

csvfile = file('final.csv','wb')
res_writer = csv.writer(csvfile)
res_writer.writerow(['id','label'])
csv_format = []
for i in range(1,12501):
    csv_format.append([i,np_res[i-1]]) 
for line in csv_format:
    res_writer.writerow(line)
csvfile.close()
