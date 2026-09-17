// Native local GPU generation. Never controls a browser or changes GPU settings.
import fs from 'node:fs';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {FlyGPU} from '../gpu-model.mjs';
import {Engine,clearance} from '../engine.mjs';
import {campaignLayout,campaignMeta,playerTimeLowerBound} from '../campaign.mjs';
import {encodeReplay,decodeReplay} from '../replay-codec.mjs';
const require=createRequire(import.meta.url),R=require('../race.js'),C=require('../contracts.js');
const {create,globals}=await import('../local-deps/package/index.js');Object.assign(globalThis,globals);
Object.defineProperty(globalThis,'navigator',{value:{gpu:create(['backend=d3d12','adapter=NVIDIA'])},configurable:true});
const originalFetch=globalThis.fetch;globalThis.fetch=async(url,opts)=>String(url).startsWith('file:')?new Response(await fs.promises.readFile(new URL(url))):originalFetch(url,opts);
const args=Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('=')));
const levels=(args.levels||Array.from({length:100},(_,i)=>i+1).join(',')).split(',').map(Number),maxAttempts=Number(args.attempts||12),probe=Object.hasOwn(args,'probe');
const base=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url))),root=new URL('../local-results/campaign/',import.meta.url);
let weightsURL;
if(args.parent){
 const parent=JSON.parse(fs.readFileSync(args.parent));
 if(parent.schema!=='full-connectome-decoder-finetune-v1'||!parent.fullConnectome||!parent.frozenParametersBitExact)throw Error('Not an audited full-connectome decoder candidate');
 base.weights_sha256=parent.weightsSha256;base.checkpoint_sha256=parent.checkpointSha256;
 weightsURL=new URL(parent.weightsFile,pathToFileURL(args.parent)).href;
}
fs.mkdirSync(root,{recursive:true});const log=x=>{const row={...x,time:new Date().toISOString()};console.log(JSON.stringify(row));fs.appendFileSync(new URL('trials.jsonl',root),JSON.stringify(row)+'\n');};
const hash=x=>createHash('sha256').update(x).digest('hex');
const gpu=await FlyGPU.create(base,()=>{},{kernel:'tile16',precision:'f32',profile:false,weightsURL});
log({type:'start',adapter:gpu.adapterInfo,weights:base.weights_sha256,levels,probe,sourceHash:hash(fs.readFileSync(new URL('../engine.mjs',import.meta.url)))});
if(gpu.adapterInfo.vendor!=='nvidia')throw Error('Campaign generation requires the verified local NVIDIA adapter');
try{
 for(const level of levels){
  const acceptedPath=new URL(`level-${String(level).padStart(3,'0')}.json`,root);
  if(!probe&&fs.existsSync(acceptedPath)){log({type:'resume',level});continue;}
  let accepted=false;
  for(let attempt=0;attempt<maxAttempts;attempt++){
   const layout=campaignLayout(level,attempt);if(args.aperture){if(!probe)throw Error('Aperture overrides are probe-only');layout.apertures.fill(Number(args.aperture));}
   if(args.goalY){if(!probe)throw Error('Goal overrides are probe-only');layout.goals.forEach(g=>g[1]+=Number(args.goalY));}
   if(Object.hasOwn(args,'variedStart')){if(!probe)throw Error('Start override is probe-only');layout.initial[1]+=.32*Math.sin(level*2.399963+attempt);layout.initial[2]+=.15*Math.cos(level*1.618+attempt);}
   if(args.wideSlots){if(!probe)throw Error('Wide-slot override is probe-only');const slots=args.wideSlots.split(',').map(Number);layout.apertures=layout.apertures.map((_,i)=>slots.includes(i)?1.9:.915);layout.narrow_count=layout.apertures.length-slots.length;}
   if(args.wideRise){if(!probe)throw Error('Rise override is probe-only');for(let i=1;i<layout.wall_x.length;i++)if(layout.apertures[i]>1.2){layout.gap_y[i]=layout.gap_y[i-1]+Number(args.wideRise);layout.goals[i][1]=layout.gap_y[i];}}
   if(args.spacing){if(!probe)throw Error('Spacing override is probe-only');layout.wall_x=layout.wall_x.map((_,i)=>i*Number(args.spacing));layout.goals.forEach((g,i)=>g[0]=layout.wall_x[i]+1.05);}
   if(Object.hasOwn(args,'clearanceTransitions')){if(!probe)throw Error('Transition override is probe-only');layout.intermediate_transition='rear-clearance';}
   const m=campaignMeta(base,layout),engine=new Engine(m,gpu),d=engine.reset(layout.seed),frames=[engine.snapshot()],start=performance.now();
   let minClearance=Infinity,pathLength=0,rotation=0,peak=0,previous=engine.state;
   while(!engine.done){const f=await engine.step();minClearance=Math.min(minClearance,clearance(f.state,engine.walls,m));pathLength+=Math.hypot(f.state[0]-previous[0],f.state[1]-previous[1]);rotation+=Math.abs(Math.atan2(Math.sin(f.state[2]-previous[2]),Math.cos(f.state[2]-previous[2])));peak=Math.max(peak,Math.abs(f.state[2]));frames.push(f);previous=f.state;}
   const outcome={type:'trial',level,attempt,seed:layout.seed,weightsSha256:base.weights_sha256,trainingCheckpoint:args.parent||null,success:engine.success,ticks:engine.tick,stage:engine.stage,contacts:engine.contacts,minClearance,pathLength,rotationRad:rotation,peakRotationDegrees:peak*180/Math.PI,seconds:(performance.now()-start)/1000,finalState:engine.state,layout};log(outcome);
   if(!engine.success)continue;
   // Audit the renderer's geometry and every accepted interpolation interval.
   for(let k=1;k<frames.length;k++)if(C.collides(frames[k].state,d)||!R.safeMotion(frames[k-1].state,frames[k].state,d))throw Error(`Independent swept audit rejected level ${level} tick ${k}`);
   const sender=base.edges.flat(),receiver=sender.map((_,i)=>sender[i^1]),header={level,layout,world_x:m.config.world_x,player_dock_angle:d.player_dock_angle,dt:m.config.dt,brainCount:base.display_ids.length,sender,receiver,sweeps:base.gbp.steps,weightsSha256:base.weights_sha256,checkpointSha256:base.checkpoint_sha256,model:{neurons:base.neurons,agents:8,steps:base.neural_steps,precision:'f32',kernel:'tile16'},sourceSimulationSeconds:engine.tick*m.config.dt};
   const binary=encodeReplay(frames,header),roundtrip=decodeReplay(binary);let brainError=0;
   for(let k=1;k<frames.length;k++){if(frames[k].state.some((v,j)=>v!==roundtrip.frames[k].state[j]))throw Error('Replay changed a physics pose');for(let j=0;j<header.brainCount;j++)brainError=Math.max(brainError,Math.abs(frames[k].brain[j]-roundtrip.frames[k].brain[j]));}
   const packed=gzipSync(binary,{level:6}),stem=`level-${String(level).padStart(3,'0')}${probe?'-probe-'+hash(JSON.stringify(layout)).slice(0,10):''}`,file=stem+'.bin.gz';
   fs.writeFileSync(new URL(file,root),packed);
   const record={...outcome,type:'accepted',schema:'campaign-level-v1',file,sha256:hash(packed),bytes:packed.length,uncompressedBytes:binary.length,maxBrainDisplayError:brainError,playerTimeLowerBound:playerTimeLowerBound(d,R.SETTINGS),timingStatus:'pending executable human reference; not an optimality claim',header};
   fs.writeFileSync(new URL(stem+'.json',root),JSON.stringify(record,null,2));log({type:'accepted',level,attempt,bytes:packed.length,ticks:engine.tick,maxBrainDisplayError:brainError});accepted=true;break;
  }
  if(!accepted)log({type:'unfilled',level,maxAttempts});
 }
}finally{gpu.device.destroy();delete globalThis.navigator;}
process.exit(0);
