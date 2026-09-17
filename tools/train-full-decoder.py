"""Fine-tune ONLY the original full-connectome feed-forward decoder.
Graph, sensory projection, normalization, local GRU and GRU residual stay frozen.
The GRU contribution on each recorded observation history is removed exactly
from the supervised target. No small-brain substitution or runtime teacher.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=2000);p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--parent');p.add_argument('--tag',default='r0');p.add_argument('--loss',choices=['raw','action','braking'],default='raw');p.add_argument('--save-every',type=int,default=250);args=p.parse_args()
torch.manual_seed(20260917);np.random.seed(20260917);torch.set_num_threads(4)
assert torch.cuda.is_available() and '4080' in torch.cuda.get_device_name()
torch.backends.cuda.matmul.allow_tf32=False
meta=json.loads((ROOT/'model/model.json').read_text());original=np.fromfile(ROOT/'model/weights.bin',dtype=np.float32)
folder=ROOT/'local-results/full-campaign-training';out=folder/'checkpoints';out.mkdir(exist_ok=True)
class Decoder(nn.Module):
 def __init__(self,flat):
  super().__init__();self.net=nn.Sequential(nn.Linear(512,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,6))
  state={}
  for name,tensor in self.state_dict().items():
   a=meta['arrays'][name];state[name]=torch.from_numpy(flat[a['offset']:a['offset']+a['length']].copy().reshape(a['shape']))
  self.load_state_dict(state)
 def forward(self,x):return self.net(x)
weights_by_hash={meta['weights_sha256']:original};parent_flat=original
for file in out.glob('*.json'):
 info=json.loads(file.read_text())
 if info.get('schema')!='full-connectome-decoder-finetune-v1':continue
 weights_by_hash[info['weightsSha256']]=np.fromfile(file.parent/info['weightsFile'],dtype=np.float32)
if args.parent:
 info=json.loads(Path(args.parent).read_text());parent_flat=weights_by_hash[info['weightsSha256']]
model=Decoder(parent_flat).cuda();model.eval()
def array(name):
 a=meta['arrays'][name];return torch.tensor(original[a['offset']:a['offset']+a['length']],device='cuda')
xmean,xstd,ymean,ystd=[array(n) for n in ('input_mean','input_std','target_mean','target_std')]
banks={'train':[],'validation':[]};manifest=[]
for path in sorted(folder.glob('r*-*.json')):
 r=json.loads(path.read_text())
 if r.get('schema')!='full-brain-decoder-dagger-v1':continue
 raw=np.fromfile(folder/r['file'],dtype=np.float32).reshape(-1,525)
 assert len(raw)==r['ticks']*8 and r['fullConnectome']
 assert hashlib.sha256((folder/r['file']).read_bytes()).hexdigest()==r['sha256']
 reference=Decoder(weights_by_hash[r['studentWeights']]).cuda().eval()
 x=torch.tensor(raw[:,:512].copy(),device='cuda');x=((x-xmean)/xstd).clamp(-10,10)
 target=torch.tensor(raw[:,512:518].copy(),device='cuda');baseline=torch.tensor(raw[:,518:524].copy(),device='cuda');flag=torch.tensor(raw[:,524].copy(),device='cuda')
 with torch.no_grad():ref=torch.cat([reference(part) for part in x.split(2048)]);supervised=ref+(target-baseline)/ystd
 frozen_residual=(baseline-ymean)/ystd-ref
 banks[r['split']].append((x,supervised,flag,frozen_residual,target,baseline));manifest.append({'file':path.name,'sha256':r['sha256'],'level':r['level'],'split':r['split'],'studentWeights':r['studentWeights']})
 del reference
data={split:tuple(torch.cat([row[k] for row in bank]) for k in range(6)) for split,bank in banks.items()}
assert not ({r['level'] for r in manifest if r['split']=='train'}&{r['level'] for r in manifest if r['split']=='validation'})
x,y,mask,res,target,baseline=data['train'];vx,vy,vm,vres,vtarget,vbaseline=data['validation'];changed=torch.where(mask>0)[0];unchanged=torch.where(mask==0)[0]
assert len(changed)>0 and len(unchanged)>0
optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
scale=ystd/torch.tensor([.0144,.0144,.032,1,1,1],device='cuda');weights=torch.tensor([1,1,1,.05,.05,.05],device='cuda')
def loss_fn(pred,target):return ((pred-target)*scale).square().mul(weights).mean(-1)
def decode(raw):
 v=raw[:,:3]/.08
 ratio=torch.maximum(torch.ones_like(v[:,0]),torch.maximum(torch.linalg.vector_norm(v[:,:2],dim=1)/.18,v[:,2].abs()/.4))
 return v/ratio[:,None]/torch.tensor([.18,.18,.4],device='cuda')
def action_loss(pred,residual,target,baseline):
 raw=(pred+residual)*ystd+ymean
 loss=(decode(raw)-decode(target)).square().mean(-1)+.02*(raw[:,3:]-target[:,3:]).square().mean(-1)+.00001*((raw-baseline)/ystd).square().mean(-1)
 if args.loss=='braking':
  limits=torch.tensor([.0144,.0144,.032],device='cuda');small=torch.maximum(torch.linalg.vector_norm(target[:,:2]/.0144,dim=1),target[:,2].abs()/.032)<.8
  # Saturation has zero radial gradient. Explicitly supervise pre-clamp
  # magnitude ONLY when the target requests braking, not throughout travel.
  loss=loss+.1*small*nn.functional.smooth_l1_loss(raw[:,:3]/limits,target[:,:3]/limits,reduction='none').mean(-1)
 return loss
start=time.time();print(json.dumps({'start':True,'examples':len(x),'validation':len(vx),'trainable':sum(t.numel() for t in model.parameters()),'frozen':'full graph, sensory encoder, GRU and residual','device':torch.cuda.get_device_name()}),flush=True)
def export(step,metrics):
 flat=parent_flat.copy()
 for name,t in model.state_dict().items():
  a=meta['arrays'][name];flat[a['offset']:a['offset']+a['length']]=t.detach().cpu().numpy().reshape(-1)
 # Everything outside the existing net.* arrays must remain exactly unchanged.
 frozen=np.ones(len(flat),dtype=bool)
 for name,a in meta['arrays'].items():
  if name.startswith('net.'):frozen[a['offset']:a['offset']+a['length']]=False
 assert np.array_equal(flat[frozen],original[frozen])
 stem=f'{args.tag}-step-{step:05d}';binary=flat.tobytes();(out/(stem+'.bin')).write_bytes(binary)
 torch.save({'decoder':model.state_dict(),'parentWeights':meta['weights_sha256'],'graphUnchanged':True,'data':manifest},out/(stem+'.pt'))
 info={'schema':'full-connectome-decoder-finetune-v1','fullConnectome':True,'weightsFile':stem+'.bin','weightsSha256':hashlib.sha256(binary).hexdigest(),'checkpointSha256':hashlib.sha256((out/(stem+'.pt')).read_bytes()).hexdigest(),'parentCheckpoint':meta['checkpoint_sha256'],'parentWeights':meta['weights_sha256'],'step':step,'metrics':metrics,'data':manifest,'frozenParametersBitExact':True,'trainable':'original net.* only','lossMode':args.loss,'trainerSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (out/(stem+'.json')).write_text(json.dumps(info));print(json.dumps({'checkpoint':str(out/(stem+'.json')),'step':step,**metrics}),flush=True)
for step in range(1,args.steps+1):
 model.train();idx=torch.cat([changed[torch.randint(len(changed),(256,),device='cuda')],unchanged[torch.randint(len(unchanged),(256,),device='cuda')]])
 pred=model(x[idx]);loss=action_loss(pred,res[idx],target[idx],baseline[idx]) if args.loss!='raw' else loss_fn(pred,y[idx]);loss=(loss*torch.where(mask[idx]>0,1.,10. if args.loss!='raw' else 3.)).mean()
 optim.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optim.step()
 if step%args.save_every==0 or step==args.steps:
  model.eval()
  with torch.no_grad():
   err=torch.cat([action_loss(model(a),r,t,b) if args.loss!='raw' else loss_fn(model(a),y) for a,y,r,t,b in zip(vx.split(4096),vy.split(4096),vres.split(4096),vtarget.split(4096),vbaseline.split(4096))]);metrics={'loss':loss.item(),'validationTeacher':err[vm>0].mean().item(),'validationPreserve':err[vm==0].mean().item(),'seconds':time.time()-start}
  export(step,metrics)
print(json.dumps({'done':True,'seconds':time.time()-start}),flush=True)
