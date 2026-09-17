import {FlyGPU} from '../gpu-model.mjs';
import {Engine,twistToWrench,clearance} from '../engine.mjs';

const percentile=(xs,p)=>[...xs].sort((a,b)=>a-b)[Math.max(0,Math.ceil(xs.length*p)-1)];
const stats=xs=>({median:percentile(xs,.5),p95:percentile(xs,.95),mean:xs.reduce((a,b)=>a+b,0)/xs.length});
const maxDiff=(a,b)=>{const y=b.flat(Infinity);return Math.max(...a.flat(Infinity).map((v,i)=>Math.abs(v-y[i])));};

// Both the native compute harness and the actual-browser Worker run this
// identical audit. Outputs say which runtime was used; no replay is a race.
export async function auditPrecision(options,emit=()=>{}){
 const meta=await (await fetch(new URL('../model/model.json',import.meta.url))).json();
 const {rows}=await (await fetch(new URL('../model/parity.json',import.meta.url))).json();
 const start=performance.now(),gpu=await FlyGPU.create(meta,progress=>emit({type:'progress',...progress}),options);
 const identity={kernel:gpu.kernel,precision:gpu.precision,memoryLayout:gpu.memoryLayout,subgroupSize:options.subgroupSize||null,subgroupRows:options.subgroupRows||null,adapter:gpu.adapterInfo,neuralStateBytes:gpu.state.reduce((n,b)=>n+b.size,0),neurons:meta.neurons,edges:25563197,neuralSteps:meta.neural_steps,weightsSha256:meta.weights_sha256};
 const report=row=>emit({...identity,...row});
 try{
  report({type:'precision-init',seconds:(performance.now()-start)/1000,features:[...gpu.device.features]});
  let feature=0,wrench=0,precision=0;const timings=[];
  for(let i=0;i<rows.length;i++){
   const row=rows[i],t=performance.now(),out=await gpu.step(row.obs);
   const total=performance.now()-t;if(i>=8)timings.push({total,...gpu.timing});
   feature=Math.max(feature,maxDiff(out.features,row.feature));
   wrench=Math.max(wrench,maxDiff(twistToWrench(out.twist,row.obs,meta),row.mean));
   precision=Math.max(precision,maxDiff(out.precision,row.precision));
  }
  report({type:'precision-parity',steps:rows.length,maxAbs:{feature,wrench,precision},neuralTotalMs:stats(timings.map(t=>t.total)),readbackWallMs:stats(timings.map(t=>t.wall)),gpuPassMeanMs:timings[0].passes?.map((_,i)=>stats(timings.map(t=>t.passes[i])).mean),budgetMs:80/3.3});
  const engine=new Engine(meta,gpu),outcomes=[];
  for(const seed of options.seeds||[]){
   engine.reset(seed);let minClearance=Infinity;const times=[],t=performance.now();
   while(!engine.done){const tick=performance.now();await engine.step();times.push(performance.now()-tick);minClearance=Math.min(minClearance,clearance(engine.state,engine.walls,meta));if(engine.tick%150===0)emit({type:'progress',seed,tick:engine.tick,stage:engine.stage});}
   const outcome={type:'precision-rollout',seed,success:engine.success,stage:engine.stage,ticks:engine.tick,contacts:engine.contacts,minClearance,wireBytes:engine.wireBytes,inferences:engine.inferences,seconds:(performance.now()-t)/1000,controlMs:stats(times),state:engine.state};
   outcomes.push(outcome);report(outcome);
  }
  report({type:'precision-complete',seeds:outcomes.length,successes:outcomes.filter(r=>r.success).length});
 }finally{gpu.device.destroy();}
}
