import {FlyWASM} from '../wasm-model.mjs';
import {twistToWrench,Engine,clearance} from '../engine.mjs';
let model,meta;
const send=value=>postMessage(value),diff=(a,b)=>{const x=b.flat(Infinity);return Math.max(...a.flat(Infinity).map((v,i)=>Math.abs(v-x[i])));};
onmessage=async({data})=>{try{
 if(data.type==='init'){
  const t=performance.now();meta=await(await fetch('../model/model.json')).json();model=await FlyWASM.create(meta,p=>send({progress:p}),{threads:data.threads});send({type:'init',seconds:(performance.now()-t)/1000,threads:model.threads,isolated:crossOriginIsolated});
 }else if(data.type==='parity'){
  model.reset();const rows=(await(await fetch('../model/parity.json')).json()).rows;let features=0,means=0,precisions=0;const timings=[];
  for(let i=0;i<rows.length;i++){const row=rows[i],out=await model.step(row.obs);features=Math.max(features,diff(out.features,row.feature));means=Math.max(means,diff(twistToWrench(out.twist,row.obs,meta),row.mean));precisions=Math.max(precisions,diff(out.precision,row.precision));timings.push(model.timing.wall);if(i%10===0)send({status:`Parity ${i+1}/${rows.length}`});}
  send({type:'parity',features,means,precisions,timings,lastParts:model.timing.parts});
 }else if(data.type==='run'){
  const engine=new Engine(meta,model);for(const seed of data.seeds){engine.reset(seed);const t=performance.now();let minClearance=Infinity;while(!engine.done){await engine.step();minClearance=Math.min(minClearance,clearance(engine.state,engine.walls,meta));if(engine.tick%50===0)send({status:`Seed ${seed}, tick ${engine.tick}, gate ${engine.stage}`});}send({type:'rollout',seed,ticks:engine.tick,success:engine.success,minClearance,seconds:(performance.now()-t)/1000});}
 }
}catch(e){send({error:e.stack});}};
