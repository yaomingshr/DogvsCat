import imtools
import os

train_feat_path = '../trainfeature/'
test_feat_path = '../testfeature/'

trainlist = imtools.get_imlist(train_feat_path)
testlist = imtools.get_imlist(test_feat_path)

zcount = 0

for fp in trainlist:
    if os.path.getsize(fp) == 0:
        zcount = zcount + 1

for fp in testlist:
    if os.path.getsize(fp) == 0:
        zcount = zcount + 1

print zcount
