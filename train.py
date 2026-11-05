# Kidney CT classification | Author: Aaryaman Singh (Reg. No. 2430010383), Manipal University Jaipur
import sys, time, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from common import load_split
model_name, split, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed = 344 if split=='group' else 42
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(1)
X,y,tr,va,te = load_split(split, seed=seed)
Xt = torch.from_numpy(X).permute(0,3,1,2).float()/255.
mean = Xt[tr].mean((0,2,3),keepdim=True); std = Xt[tr].std((0,2,3),keepdim=True)
Xt = F.avg_pool2d(Xt,2); mean = Xt[tr].mean((0,2,3),keepdim=True); std = Xt[tr].std((0,2,3),keepdim=True)
Xt = (Xt-mean)/std; yt = torch.from_numpy(y)

class SmallCNN(nn.Module):
    def __init__(s, nc=4):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True),nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(3,32),blk(32,64),blk(64,128),blk(128,128),blk(128,256))
        s.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.4),nn.Linear(256,nc))
    def forward(s,x): return s.head(s.f(x))
if model_name=='cnn': net=SmallCNN()
else: net=torchvision.models.resnet18(num_classes=4)   # random init (no pretrained weights available offline)
nparams=sum(p.numel() for p in net.parameters())

cnt=np.bincount(y[tr],minlength=4); w=torch.tensor((cnt.sum()/(4*cnt))**0.5,dtype=torch.float32)
crit=nn.CrossEntropyLoss(weight=w,label_smoothing=0.05)
opt=torch.optim.AdamW(net.parameters(),lr=2e-3 if model_name=='cnn' else 1e-3,weight_decay=1e-4)
bs=32; steps=epochs*math.ceil(len(tr)/bs)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=opt.param_groups[0]['lr'],total_steps=steps,pct_start=0.2)

def augment(x):
    B=x.size(0)
    ang=(torch.rand(B)*2-1)*math.radians(10); sc=1+(torch.rand(B)*2-1)*0.1
    tx=(torch.rand(B)*2-1)*0.06; ty=(torch.rand(B)*2-1)*0.06
    flip=(torch.rand(B)<0.5).float()*2-1
    cos,sin=torch.cos(ang)/sc,torch.sin(ang)/sc
    theta=torch.stack([torch.stack([cos*flip,-sin,tx],1),torch.stack([sin*flip,cos,ty],1)],1)
    g=F.affine_grid(theta,x.shape,align_corners=False)
    x=F.grid_sample(x,g,padding_mode='border',align_corners=False)
    c=1+(torch.rand(B,1,1,1)*2-1)*0.1; b=(torch.rand(B,1,1,1)*2-1)*0.1
    return x*c+b

@torch.no_grad()
def predict(idx):
    net.eval(); out=[]
    for i in range(0,len(idx),128): out.append(F.softmax(net(Xt[idx[i:i+128]]),1))
    return torch.cat(out).numpy()
def macro_f1(idx,p): return precision_recall_fscore_support(y[idx],p.argmax(1),average='macro',zero_division=0)[2]

import os
tag=f'{model_name}_{split}'; ck=f'ck_{tag}.pt'; start=0; hist=[]; best=(-1,None,None); prev=0.0
if os.path.exists(ck):
    c=torch.load(ck,weights_only=False); net.load_state_dict(c['net']); opt.load_state_dict(c['opt']); sched.load_state_dict(c['sched']); start=c['ep']; hist=c['hist']; best=c['best']; prev=c['secs']
t0=time.time()-prev; tstart=time.time()
for ep in range(start,epochs):
    if time.time()-tstart>200: break
    net.train(); perm=tr[np.random.permutation(len(tr))]; tl=0;n=0;corr=0
    for i in range(0,len(perm),bs):
        b=perm[i:i+bs]
        if len(b)<2: continue
        xb=augment(Xt[b]); yb=yt[b]
        opt.zero_grad(); out=net(xb); loss=crit(out,yb); loss.backward(); opt.step(); sched.step()
        tl+=loss.item()*len(b); n+=len(b); corr+=(out.argmax(1)==yb).sum().item()
    pv=predict(va); f1v=macro_f1(va,pv); accv=accuracy_score(y[va],pv.argmax(1))
    hist.append(dict(epoch=ep+1,train_loss=tl/n,train_acc=corr/n,val_acc=accv,val_f1=f1v,secs=time.time()-t0))
    print(f"[{tag}] ep{ep+1}/{epochs} loss {tl/n:.3f} tr_acc {corr/n:.3f} val_acc {accv:.3f} val_f1 {f1v:.3f} t={time.time()-t0:.0f}s",flush=True)
    if f1v>=best[0]:
        best=(f1v,ep+1,{k:v.clone() for k,v in net.state_dict().items()})
    torch.save(dict(net=net.state_dict(),opt=opt.state_dict(),sched=sched.state_dict(),ep=ep+1,hist=hist,best=best,secs=time.time()-t0),ck)
if len(hist)<epochs: print('PAUSED at',len(hist)); sys.exit(0)
net.load_state_dict(best[2])
t1=time.time(); pt=predict(te); inf_ms=(time.time()-t1)/len(te)*1000
p=pt.argmax(1); pr,rc,f1,_=precision_recall_fscore_support(y[te],p,average='macro',zero_division=0)
perc=precision_recall_fscore_support(y[te],p,average=None,zero_division=0)
try: auc=roc_auc_score(y[te],pt,multi_class='ovr',average='macro')
except Exception: auc=float('nan')
res=dict(model=model_name,split=split,params=nparams,epochs=epochs,best_epoch=best[1],val_f1=best[0],
         acc=accuracy_score(y[te],p),precision=pr,recall=rc,f1=f1,auc=auc,cm=confusion_matrix(y[te],p,labels=[0,1,2,3]).tolist(),
         per_class=dict(precision=perc[0].tolist(),recall=perc[1].tolist(),f1=perc[2].tolist(),support=perc[3].tolist()),
         train_secs=time.time()-t0,inf_ms_per_img=inf_ms,hist=hist,n=dict(train=len(tr),val=len(va),test=len(te)))
json.dump(res,open(f'res_{tag}.json','w')); np.save(f'probs_{tag}.npy',pt); torch.save(best[2],f'w_{tag}.pt')
print('FINAL',tag,{k:res[k] for k in ['acc','precision','recall','f1','auc','best_epoch']},flush=True)
