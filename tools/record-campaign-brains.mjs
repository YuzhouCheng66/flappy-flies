// Additional measured display traces only: no policy/trajectory replacement.
import fs from 'node:fs';
import {gzipSync,gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {FlyGPU} from '../gpu-model.mjs';
import {Engine} from '../engine.mjs';
import {campaignMeta} from '../campaign.mjs';
import {COURSE_IDS} from '../campaign-selection.mjs';
import {decodeReplay} from '../replay-codec.mjs';
const {create,globals}=await import('../local-deps/package/index.js');Object.assign(globalThis,globals);
Object.defineProperty(globalThis,'navigator',{value:{gpu:create(['backend=d3d12','adapter=NVIDIA'])},configurable:true});
const nativeFetch=globalThis.fetch;globalThis.fetch=async(u,o)=>String(u).startsWith('file:')?new Response(await fs.promises.readFile(new URL(u))):nativeFetch(u,o);
const root=new URL('../local-results/campaign/',import.meta.url),base=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url))),manifest=JSON.parse(fs.readFileSync(new URL('manifest.json',root)));
const hash=b=>createHash('sha256').update(b).digest('hex');
const output=new URL('brains.json',root),index=fs.existsSync(output)?JSON.parse(fs.readFileSync(output)):{schema:'flappy-brain-sidecars-v1',neuronsPerAgent:2048,encoding:'int16-le / 32767',levels:{}};
const save=()=>{fs.writeFileSync(new URL('brains.pending.json',root),JSON.stringify(index,null,2));fs.renameSync(new URL('brains.pending.json',root),output);};
let gpu,activeHash;
try{for(const id of COURSE_IDS){
 const entry=manifest.levels.find(e=>e.id===id);
 if(index.levels[id]?.recordingSha256===entry.sha256)continue;
 const packed=fs.readFileSync(new URL(entry.file,root));if(hash(packed)!==entry.sha256)throw Error('Recording hash mismatch');
 const record=decodeReplay(gunzipSync(packed));
 if(activeHash!==entry.weightsSha256){
  gpu?.device.destroy();let weightsURL;
  if(entry.weightsSha256!==base.weights_sha256){
   const checkpoint=new URL('../local-results/full-campaign-training/checkpoints/readouts1-step-30000.json',import.meta.url),parent=JSON.parse(fs.readFileSync(checkpoint));
   if(parent.weightsSha256!==entry.weightsSha256)throw Error('Wrong source checkpoint');
   weightsURL=new URL(parent.weightsFile,checkpoint).href;
  }
  gpu=await FlyGPU.create({...base,weights_sha256:entry.weightsSha256},()=>{},{kernel:'tile16',precision:'f32',profile:false,captureAllBrains:true,weightsURL});
  if(gpu.adapterInfo.vendor!=='nvidia')throw Error('Expected local NVIDIA GPU');activeHash=entry.weightsSha256;
 }
 const engine=new Engine(campaignMeta(base,record.header.layout),gpu);engine.reset(record.header.layout.seed);
 const count=record.frames.length,B=2048,arrays=Array.from({length:7},()=>Buffer.alloc(count*B*2));let poseError=0,brainError=0;
 for(let tick=1;tick<count;tick++){
  const f=await engine.step(),old=record.frames[tick];if(!f||f.stage!==old.stage||f.success!==old.success)throw Error(`Trajectory mismatch ${id}/${tick}`);
  poseError=Math.max(poseError,...f.state.map((v,k)=>Math.abs(v-old.state[k])));
  brainError=Math.max(brainError,...f.brain.map((v,k)=>Math.abs(v-old.brain[k])));
  if(poseError>1e-6||brainError>1/32767+1e-7)throw Error(`Parity failed ${id}/${tick}: ${poseError}, ${brainError}`);
  const brains=engine.neural.brain_all;
  for(let a=1;a<8;a++)for(let j=0;j<B;j++)arrays[a-1].writeInt16LE(Math.round(Math.max(-1,Math.min(1,brains[a][j]))*32767),(tick*B+j)*2);
 }
 if(!engine.success)throw Error('Missing successful terminal');
 const agents=[];
 for(let a=1;a<8;a++){const bytes=gzipSync(arrays[a-1],{level:6}),file=`brain-${String(id).padStart(3,'0')}-${String(a+1).padStart(2,'0')}.bin.gz`;fs.writeFileSync(new URL(file,root),bytes);agents.push({agent:a+1,file,bytes:bytes.length,sha256:hash(bytes)});}
 index.levels[id]={recordingSha256:entry.sha256,weightsSha256:entry.weightsSha256,frames:count,poseMaxError:poseError,agent01MaxDisplayError:brainError,agents};save();
 console.log(JSON.stringify({sourceLevel:id,complete:Object.keys(index.levels).length,poseError,brainError,bytes:agents.reduce((s,a)=>s+a.bytes,0)}));
}}finally{gpu?.device.destroy();delete globalThis.navigator;}
process.exit(0);
