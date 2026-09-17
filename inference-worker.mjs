import {FlyGPU} from './gpu-model.mjs';
import {Engine} from './engine.mjs';
let engine,meta,generation=0,chain=Promise.resolve(),ready=false,cancellation=0;
async function request(type,body){
 if(type==='init'){
  meta=await(await fetch('./model/model.json')).json();engine=new Engine(meta,null);
  FlyGPU.create(meta,p=>postMessage({event:'progress',...p})).then(gpu=>{engine.gpu=gpu;gpu.reset();ready=true;postMessage({event:'ready',adapter:gpu.adapterInfo,candidates:gpu.adapterCandidates,kernel:gpu.kernel});}).catch(e=>postMessage({event:'failed',error:e.message}));
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
