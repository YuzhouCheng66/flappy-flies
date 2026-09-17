import {FlyGPU} from './gpu-model.mjs';
import {Engine} from './engine.mjs';
import {CompactModel} from './compact-model.mjs';
import {measureBackend} from './backend-selector.mjs';
import {observe} from './engine.mjs';
let engine,meta,generation=0,chain=Promise.resolve(),ready=false,cancellation=0;
const candidate=['localhost','127.0.0.1','[::1]'].includes(location.hostname)?new URLSearchParams(location.search).get('candidate'):null;
async function createModel(){
 if(!/^[a-f0-9]{12}$/.test(candidate||''))return FlyGPU.create(meta,p=>postMessage({event:'progress',...p}));
 const bytes=await(await fetch(`./local-results/compact/model-${candidate}.json`)).arrayBuffer();
 const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
 if(!hash.startsWith(candidate))throw Error('Research checkpoint identity mismatch');
 const model=new CompactModel(JSON.parse(new TextDecoder().decode(bytes))),probe=new Engine(meta,model);probe.reset(91000);
 const measured=await measureBackend(model,[observe(probe.state,probe.layout,probe.stage,probe.context,meta)],{warmup:3,samples:8});
 // Preview is explicitly unvalidated for release. Measure its speed without
 // falsely marking it as an independently accepted production backend.
 if(measured.medianMs>15||measured.p95Ms>20)throw Error('Research candidate misses the measured real-time deadline');
 model.adapterInfo={vendor:'CPU',device:'Compact local GRU · research candidate'};model.measurement=measured;model.modelHash=hash;return model;
}
async function request(type,body){
 if(type==='init'){
  meta=await(await fetch('./model/model.json')).json();engine=new Engine(meta,null);
  createModel().then(gpu=>{engine.gpu=gpu;gpu.reset();ready=true;postMessage({event:'ready',adapter:gpu.adapterInfo,candidates:gpu.adapterCandidates,kernel:gpu.kernel,modelHash:gpu.modelHash,measurement:gpu.measurement});}).catch(e=>postMessage({event:'failed',error:e.message}));
  return {description:engine.description(),generation};
 }
 if(type==='reset'){generation++;return{description:engine.reset(body.seed>>>0),generation};}
 if(type==='step'){
  if(!ready)throw Error('The full neural model is still loading');
  if(body.generation!==generation||body.after_tick!==engine.tick)return{stale:true,generation,frames:[]};
  const frames=[],start=performance.now(),version=cancellation;for(let i=0;i<Math.min(8,body.count)&&!engine.done&&version===cancellation;i++)frames.push(await engine.step());
  return{generation,frames,done:engine.done,msPerTick:(performance.now()-start)/Math.max(1,frames.length),gpuTiming:engine.gpu.timing};
 }
 throw Error('Unknown local inference request');
}
onmessage=({data:{id,type,body}})=>{if(type==='reset')cancellation++;chain=chain.then(async()=>{try{postMessage({id,result:await request(type,body)});}catch(e){postMessage({id,error:e.message});}});};
