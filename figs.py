# Kidney CT classification | Author: Aaryaman Singh (Reg. No. 2430010383), Manipal University Jaipur
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':8.5})
C=['Cyst','Normal','Stone','Tumor']
# architecture
fig,ax=plt.subplots(figsize=(3.5,2.5)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,10)
def box(x,y,w,h,t,fc='#e8eef7'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.15',fc=fc,ec='#333',lw=0.7)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=7.6)
def arr(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=0.7))
r1=[('CT slice\n(JPEG)',0.1),('Resize 128\n+ 2x pool\n(64x64)',2.6),('Normalize\n(mean/std)',5.1),('Augment\n(flip, rot.,\nshift, scale)',7.6)]
for t,x in r1: box(x,7.2,2.3,2.4,t,'#fff4d6')
for i in range(3): arr(r1[i][1]+2.3,8.4,r1[i+1][1],8.4)
arr(8.75,7.2,8.75,6.3); ax.plot([8.75,1.2],[6.3,6.3],'k',lw=0.7); arr(1.2,6.3,1.2,5.6)
cb=[('Conv\n32\nBN-ReLU\nPool',0.1),('Conv\n64\nBN-ReLU\nPool',2.1),('Conv\n128\nBN-ReLU\nPool',4.1),('Conv\n128\nBN-ReLU\nPool',6.1),('Conv\n256\nBN-ReLU\nPool',8.1)]
for t,x in cb: box(x,3.2,1.6,2.5,t)
for i in range(4): arr(cb[i][1]+1.6,4.45,cb[i+1][1],4.45)
arr(9.0,3.2,9.0,2.7); ax.plot([9.0,1.5],[2.7,2.7],'k',lw=0.7); arr(1.5,2.7,1.5,2.1)
box(0.1,0.2,2.9,1.9,'GAP +\nDropout(0.4)','#e3f3e3'); box(3.6,0.2,2.6,1.9,'FC(256→4)\nSoftmax','#e3f3e3'); box(6.8,0.2,3.0,1.9,'Cyst | Normal\nStone | Tumor','#f9dede')
arr(3.0,1.15,3.6,1.15); arr(6.2,1.15,6.8,1.15)
plt.savefig('fig_arch.png',dpi=300,bbox_inches='tight'); plt.close()
# curves (strict split)
fig,ax=plt.subplots(1,2,figsize=(3.5,1.9),sharey=True)
for a,m,t in zip(ax,['cnn','resnet18'],['Lightweight CNN','ResNet-18']):
    h=json.load(open(f'res_{m}_group.json'))['hist']; e=[x['epoch'] for x in h]
    a.plot(e,[x['train_acc'] for x in h],label='Train'); a.plot(e,[x['val_acc'] for x in h],label='Val (held-out)'); a.set_title(t,fontsize=8.5); a.set_xlabel('Epoch'); a.grid(alpha=.3)
ax[0].set_ylabel('Accuracy'); ax[0].legend(fontsize=6.8,loc='lower right'); plt.tight_layout(); plt.savefig('fig_curves.png',dpi=300); plt.close()
# confusion matrices ResNet-18
fig,ax=plt.subplots(1,2,figsize=(3.5,1.9))
for a,s,t in zip(ax,['block','group'],['Block split','Source-held-out']):
    cm=np.array(json.load(open(f'res_resnet18_{s}.json'))['cm']); n=cm/cm.sum(1,keepdims=True)
    a.imshow(n,cmap='Blues',vmin=0,vmax=1); a.set_title(t,fontsize=8.5); a.set_xticks(range(4)); a.set_yticks(range(4)); a.set_xticklabels(C,rotation=45,fontsize=6.8); a.set_yticklabels(C if s=='block' else [],fontsize=6.8)
    for i in range(4):
        for j in range(4): a.text(j,i,cm[i,j],ha='center',va='center',fontsize=7.4,color='white' if n[i,j]>.5 else 'black')
    a.set_xlabel('Predicted',fontsize=7.4)
ax[0].set_ylabel('True',fontsize=7.4); plt.tight_layout(); plt.savefig('fig_cm.png',dpi=300); plt.close()
# class distribution
fig,ax=plt.subplots(figsize=(3.5,1.7)); n=[817,1425,181,512]; b=ax.bar(C,n,color=['#4c78a8','#59a14f','#f28e2b','#e15759']); ax.bar_label(b,fontsize=7.4); ax.set_ylabel('Images'); ax.set_ylim(0,1650); plt.tight_layout(); plt.savefig('fig_dist.png',dpi=300)
