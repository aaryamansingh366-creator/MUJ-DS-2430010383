# Kidney CT classification | Author: Aaryaman Singh (Reg. No. 2430010383), Manipal University Jaipur
import glob, re, numpy as np, collections
from PIL import Image
from common import *
root=glob.glob(os.environ.get('DATA_DIR','data')+'/CT-KIDNEY*/CT-KIDNEY*')[0]
sizes=[];ids=[]
for ci,c in enumerate(classes):
    fs=sorted(glob.glob(f'{root}/{c}/*.jpg'))
    for f in fs:
        sizes.append(Image.open(f).size); ids.append(int(re.search(r'\((\d+)\)',f).group(1)))
X,y,tr,va,te=load_split()
print("distinct sizes per class and top-3:")
for ci,c in enumerate(classes):
    cs=collections.Counter([sizes[i] for i in np.where(y==ci)[0]])
    print(c,len(cs),cs.most_common(3), 'id range',min(ids[i] for i in np.where(y==ci)[0]),max(ids[i] for i in np.where(y==ci)[0]))
# size-only classifier: majority class per size on train, test accuracy
sz=np.array([f"{w}x{h}" for w,h in sizes])
maj={}
for s in set(sz[tr]):
    maj[s]=np.bincount(y[tr][sz[tr]==s],minlength=4).argmax()
pred=np.array([maj.get(s,1) for s in sz[te]]); print("size-only majority-vote test acc:",(pred==y[te]).mean(), "unseen sizes in test:",sum(s not in maj for s in sz[te]))
# purity of sizes
pur=[ (s, np.bincount(y[sz==s],minlength=4).tolist()) for s in set(sz)]
mixed=[p for p in pur if sum(v>0 for v in p[1])>1]
print("sizes shared by >1 class:",len(mixed),"of",len(pur)); print(sorted(mixed,key=lambda p:-sum(p[1]))[:6])
# near duplicate check: nearest neighbour pixel distance test->train (downsampled gray 32x32)
g=np.stack([np.asarray(Image.fromarray(im).convert('L').resize((32,32)),dtype=np.float32) for im in X]).reshape(len(X),-1)
from sklearn.neighbors import NearestNeighbors
nn=NearestNeighbors(n_neighbors=1).fit(g[tr]); d,ix=nn.kneighbors(g[te])
print("test->train NN dist percentiles (32x32 gray L2):",np.percentile(d,[5,25,50,75,95]).round(1))
print("1-NN pixel classifier test acc:",(y[tr][ix[:,0]]==y[te]).mean())
# within-train adjacent slice distance for reference
nn2=NearestNeighbors(n_neighbors=2).fit(g[tr]); d2,_=nn2.kneighbors(g[tr][:400]); print("train self NN(excluding itself) percentiles:",np.percentile(d2[:,1],[5,25,50,75,95]).round(1))
