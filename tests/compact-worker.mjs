import {CompactModel} from '../compact-model.mjs';
import {Engine,clearance,observe} from '../engine.mjs';
import {measureBackend,chooseMeasuredBackend} from '../backend-selector.mjs';
let model,meta,engine,modelHash;
onmessage=async({data})=>{try{
 if(data.type==='init'){
  const start=performance.now();const modelFile=/^[a-f0-9]{12}$/.test(data.model||'')?`model-${data.model}.json`:'model.json';
  const [encoded,physics]=await Promise.all([fetch('../local-results/compact/'+modelFile).then(r=>r.arrayBuffer()),fetch('../model/model.json').then(r=>r.json())]);
  modelHash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',encoded)),x=>x.toString(16).padStart(2,'0')).join('');
  const config=JSON.parse(new TextDecoder().decode(encoded));meta=physics;model=new CompactModel(config);engine=new Engine(meta,model);engine.reset(91000);
  const obs=observe(engine.state,engine.layout,engine.stage,engine.context,meta),perf=await measureBackend(model,[obs],{warmup:5,samples:20});
  postMessage({type:'init',modelHash,seconds:(performance.now()-start)/1000,kind:model.kernel,fullConnectome:false,...perf});
 }else if(data.type==='run'){
  for(const seed of data.seeds){if(model.meta.train_seeds.includes(seed)||model.meta.validation_seeds.includes(seed))throw Error('Evaluation seed leakage');engine.reset(seed);const t=performance.now();let minClearance=Infinity;const timings=[];
   while(!engine.done){const start=performance.now();await engine.step();timings.push(performance.now()-start);minClearance=Math.min(minClearance,clearance(engine.state,engine.walls,meta));if(engine.tick%100===0){postMessage({status:`Seed ${seed}, tick ${engine.tick}, gate ${engine.stage}`});await new Promise(r=>setTimeout(r,0));}}
   timings.sort((a,b)=>a-b);postMessage({type:'rollout',modelHash,seed,success:engine.success,ticks:engine.tick,stage:engine.stage,minClearance,seconds:(performance.now()-t)/1000,controlMedianMs:timings[Math.floor(timings.length/2)],controlP95Ms:timings[Math.ceil(timings.length*.95)-1]});
  }
 }
}catch(e){postMessage({error:e.stack});}};
