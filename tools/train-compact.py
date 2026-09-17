"""Explicitly labelled distillation candidate, not the complete connectome.
Independent local observation -> recurrent memory -> twist/precision. The
unchanged Engine supplies actual distributed GBP and collision-checked physics.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=1800);parser.add_argument('--parent');parser.add_argument('--lr',type=float,default=5e-4);parser.add_argument('--snapshot-every',type=int,default=0);args=parser.parse_args()
torch.manual_seed(20260917);np.random.seed(20260917);torch.set_num_threads(4)
assert torch.cuda.is_available()
assert '4080' in torch.cuda.get_device_name(), 'Only use the local 4080 for this task'
device='cuda';folder=ROOT/'local-results/distillation'
files=sorted(folder.glob('seed-*.json'))+sorted(folder.glob('dagger-*.json'));assert len(files)>=32
episodes=[];identities=[]
for file in files:
 data=json.loads(file.read_text());identities.append(data['seed'])
 if not data['success'] and not data.get('modelInduced'):continue
 x=np.asarray([r['obs'] for r in data['rows']],dtype=np.float32)
 y=np.concatenate((np.asarray([r['twist'] for r in data['rows']],dtype=np.float32),np.log(np.asarray([r['precision'] for r in data['rows']],dtype=np.float32))),axis=-1)
 episodes.append((data['seed'],x,y))
train=[e for e in episodes if e[0]%6!=5];valid=[e for e in episodes if e[0]%6==5]
assert train and valid
flatx=np.concatenate([e[1].reshape(-1,27) for e in train]);flaty=np.concatenate([e[2].reshape(-1,6) for e in train])
xmean=flatx.mean(0);xstd=np.maximum(flatx.std(0),.05);ymean=flaty.mean(0);ystd=np.maximum(flaty.std(0),.03)
parent=None
if args.parent:
 parent=json.loads(Path(args.parent).read_text())
 xmean,xstd,ymean,ystd=[np.asarray(parent[k],dtype=np.float32) for k in ('xmean','xstd','ymean','ystd')]
def pack(items):
 result=[]
 for seed,x,y in items:
  x=(x-xmean)/xstd;y=(y-ymean)/ystd
  for a in range(8):result.append((torch.tensor(x[:,a],device=device),torch.tensor(y[:,a],device=device)))
 return result
trainseq=pack(train);valseq=pack(valid)
class Compact(nn.Module):
 def __init__(self):
  super().__init__();self.embed=nn.Sequential(nn.Linear(27,128),nn.SiLU());self.gru=nn.GRU(128,128,batch_first=True);self.head=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,6))
 def forward(self,x,h=None):
  z,h=self.gru(self.embed(x.clamp(-8,8)),h);return self.head(z),h
model=Compact().to(device)
if parent:model.load_state_dict({k:torch.tensor(v['data'],device=device).reshape(v['shape']) for k,v in parent['parameters'].items()})
optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
def batch(sequences,b=48,length=96):
 xs=[];ys=[]
 for _ in range(b):
  x,y=sequences[np.random.randint(len(sequences))];start=np.random.randint(max(1,len(x)-length+1));xs.append(x[start:start+length]);ys.append(y[start:start+length])
 return torch.stack(xs),torch.stack(ys)
def fullbatch(sequences,b=12):
 chosen=[sequences[np.random.randint(len(sequences))] for _ in range(b)];length=max(len(x) for x,y in chosen)
 xs=torch.zeros(b,length,27,device=device);ys=torch.zeros(b,length,6,device=device);mask=torch.zeros(b,length,1,device=device)
 for i,(x,y) in enumerate(chosen):xs[i,:len(x)]=x;ys[i,:len(y)]=y;mask[i,:len(x)]=1
 return xs,ys,mask
def sequence_loss(pred,target,mask):
 return (((pred[:,:,:3]-target[:,:,:3])**2)*mask).sum()/(mask.sum()*3)+.15*(((pred[:,:,3:]-target[:,:,3:])**2)*mask).sum()/(mask.sum()*3)
starttime=time.time();best=float('inf');outdir=ROOT/'local-results/compact';outdir.mkdir(exist_ok=True)
data_manifest=[dict(file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files]
def export_model(step,score,canonical=False):
 export=dict(kind='compact-local-GRU-distilled-v1',full_connectome=False,hidden=128,xmean=xmean.tolist(),xstd=xstd.tolist(),ymean=ymean.tolist(),ystd=ystd.tolist(),parameters={k:dict(shape=list(v.shape),data=v.detach().cpu().flatten().tolist()) for k,v in model.state_dict().items()},train_seeds=sorted(set(e[0] for e in train)),validation_seeds=sorted(set(e[0] for e in valid)),validation_loss=score,training_step=step,learning_rate=args.lr,data_manifest=data_manifest,trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_sha256=hashlib.sha256(Path(args.parent).read_bytes()).hexdigest() if args.parent else None)
 encoded=json.dumps(export);identity=hashlib.sha256(encoded.encode()).hexdigest();path=outdir/f'model-{identity[:12]}.json';path.write_text(encoded)
 # CPU FP32 reference excludes CUDA TF32 roundoff from the JS parity criterion.
 reference=Compact().cpu();reference.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()});reference.eval()
 with torch.no_grad():
  x,y=valseq[0];prediction,_=reference(x.cpu()[None]);parity=json.dumps(dict(reference='CPU float32 (not CUDA TF32)',normalized_obs=x.cpu().tolist(),normalized_output=prediction[0].tolist()))
 (outdir/f'parity-{identity[:12]}.json').write_text(parity)
 if canonical:(outdir/'model.json').write_text(encoded);(outdir/'parity.json').write_text(parity)
 print(json.dumps(dict(snapshot=str(path),sha256=identity,step=step,validation=score)),flush=True)
for step in range(args.steps):
 full=bool(parent) or step>=args.steps-1000
 if step==args.steps-1000:best=float('inf')
 model.train()
 if full:
  x,y,mask=fullbatch(trainseq);pred,_=model(x);loss=sequence_loss(pred,y,mask)
 else:
  x,y=batch(trainseq);pred,_=model(x);loss=((pred[:,32:,:3]-y[:,32:,:3])**2).mean()+.15*((pred[:,32:,3:]-y[:,32:,3:])**2).mean()
 optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
 if step%100==0 or step+1==args.steps:
  model.eval()
  with torch.no_grad():
   validation=[]
   for _ in range(4):
    if full:
     vx,vy,vm=fullbatch(valseq,b=8);vp,_=model(vx);validation.append(sequence_loss(vp,vy,vm).item())
    else:
     vx,vy=batch(valseq,b=24,length=128);vp,_=model(vx);validation.append((((vp[:,32:,:3]-vy[:,32:,:3])**2).mean()+.15*((vp[:,32:,3:]-vy[:,32:,3:])**2).mean()).item())
  score=float(np.mean(validation));print(json.dumps(dict(step=step+1,loss=loss.item(),validation=score,seconds=time.time()-starttime)),flush=True)
  if score<best:
   best=score;torch.save(model.state_dict(),outdir/'best.pt')
  if args.snapshot_every and (step+1)%args.snapshot_every==0:export_model(step+1,score)
 elif args.snapshot_every and (step+1)%args.snapshot_every==0:export_model(step+1,None)
model.load_state_dict(torch.load(outdir/'best.pt',weights_only=True));model.eval()
export_model(args.steps,best,canonical=True)
print(json.dumps(dict(done=True,parameters=sum(p.numel() for p in model.parameters()),best=best,episodes=len(episodes),path=str(outdir/'model.json'))),flush=True)
