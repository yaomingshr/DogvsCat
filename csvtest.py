import os
import csv
from numpy import *

cf = file('test.csv','wb')
cw = csv.writer(cf)
cw.writerow(['Column1', 'Column2', 'Column3'])
lines = [range(3) for i in range(5)]
for line in lines:
    cw.writerow(line)
