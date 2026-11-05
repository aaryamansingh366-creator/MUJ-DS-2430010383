# Kidney CT classification | Author: Aaryaman Singh (Reg. No. 2430010383), Manipal University Jaipur
import numpy as np, json, time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import sys
from common import *
split=sys.argv[1]
X,y,tr,va,te=load_split(split, seed=344 if split=='group' else 42)
g=X.mean(-1)
t=time.time()
F=np.stack([hog(im,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2)) for im in g])
sc=StandardScaler().fit(F[tr]); Ft=sc.transform(F)
best=None
for C in [0.0005,0.002,0.01,0.05,0.2]:
    m=LinearSVC(C=C,max_iter=5000,class_weight='balanced').fit(Ft[tr],y[tr])
    a=accuracy_score(y[va],m.predict(Ft[va]))
    if best is None or a>best[0]: best=(a,C,m)
m=best[2]; p=m.predict(Ft[te])
pr,rc,f1,_=precision_recall_fscore_support(y[te],p,average='macro',zero_division=0)
res=dict(model='hog_svm',split=split,C=best[1],val_acc=best[0],acc=accuracy_score(y[te],p),precision=pr,recall=rc,f1=f1,cm=confusion_matrix(y[te],p).tolist(),feat_dim=F.shape[1],secs=time.time()-t)
print(res); json.dump(res,open(f'res_hog_svm_{split}.json','w'))
