# Kidney CT classification | Author: Aaryaman Singh (Reg. No. 2430010383), Manipal University Jaipur
import glob, re, os, numpy as np
from PIL import Image
classes=['Cyst','Normal','Stone','Tumor']
def _meta():
    root=glob.glob(os.environ.get('DATA_DIR','data')+'/CT-KIDNEY*/CT-KIDNEY*')[0]
    ids=[];sizes=[]
    for c in classes:
        for f in sorted(glob.glob(f'{root}/{c}/*.jpg')):
            ids.append(int(re.search(r'\((\d+)\)',f).group(1))); sizes.append(Image.open(f).size)
    return np.array(ids), np.array([f"{w}x{h}" for w,h in sizes])
def load_split(kind='block', seed=42, block=10):
    d=np.load('ct128.npz'); X,y=d['X'],d['y']
    if os.path.exists('meta.npz'):
        m=np.load('meta.npz'); ids,sz=m['ids'],m['sz']
    else:
        ids,sz=_meta(); np.savez('meta.npz',ids=ids,sz=sz)
    rng=np.random.RandomState(seed)
    tr,va,te=[],[],[]
    for ci in range(4):
        idx=np.where(y==ci)[0]; N=len(idx)
        if kind=='block':
            idx=idx[np.argsort(ids[idx])]
            blocks=[idx[i:i+block] for i in range(0,N,block)]
            order=rng.permutation(len(blocks)); n=len(blocks)
            ntr=int(round(.70*n)); nva=int(round(.15*n))
            for k,b in enumerate(order):
                (tr if k<ntr else va if k<ntr+nva else te).extend(blocks[b])
        else:  # group: whole (class,size) source groups go to one side
            groups=[idx[sz[idx]==s] for s in sorted(set(sz[idx]))]
            order=rng.permutation(len(groups)); nt=nv=0
            for gi in order:
                g=groups[gi]; s=len(g)
                if nt<0.08*N and nt+s<=0.25*N: te.extend(g); nt+=s
                elif nv<0.08*N and nv+s<=0.25*N: va.extend(g); nv+=s
                else: tr.extend(g)
    return X,y,np.array(tr),np.array(va),np.array(te)
if __name__=='__main__':
    for k in ['block','group']:
        X,y,tr,va,te=load_split(k, seed=344 if k=='group' else 42); print(k)
        for n,s in [('train',tr),('val',va),('test',te)]:
            print(' ',n,len(s),np.bincount(y[s],minlength=4))
