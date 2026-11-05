# Kidney CT classification | Author: Aaryaman Singh (Reg. No. 2430010383), Manipal University Jaipur
import os
import glob, os, numpy as np
from PIL import Image
root=glob.glob(os.environ.get('DATA_DIR','data')+'/CT-KIDNEY*/CT-KIDNEY*')[0]
classes=['Cyst','Normal','Stone','Tumor']
S=128
X=[];y=[];meta=[]
for ci,c in enumerate(classes):
    fs=sorted(glob.glob(f'{root}/{c}/*.jpg'))
    for f in fs:
        im=Image.open(f).convert('RGB'); meta.append(im.size)
        X.append(np.asarray(im.resize((S,S),Image.BILINEAR),dtype=np.uint8)); y.append(ci)
X=np.stack(X); y=np.array(y)
np.savez_compressed('ct128.npz',X=X,y=y)
print(X.shape, np.bincount(y))
import collections
print(len(set(meta)), 'distinct original sizes; most common', collections.Counter(meta).most_common(3))
# montage
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rng=np.random.RandomState(1)
fig,ax=plt.subplots(2,4,figsize=(8,4.3))
for ci,c in enumerate(classes):
    idx=np.where(y==ci)[0]; sel=rng.choice(idx,2,replace=False)
    for r,i in enumerate(sel):
        ax[r,ci].imshow(X[i]); ax[r,ci].axis('off')
        if r==0: ax[r,ci].set_title(c,fontsize=11)
plt.tight_layout(); plt.savefig('fig_samples.png',dpi=200)
