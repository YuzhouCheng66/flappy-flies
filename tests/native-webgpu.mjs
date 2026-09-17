// Compute-only Dawn harness, NOT browser automation. The production shaders
// and controller are imported unchanged; label results native, not browser.
import fs from 'node:fs';
import {FlyGPU} from '../gpu-model.mjs';
import {Engine,twistToWrench,clearance} from '../engine.mjs';
const {create,globals}=await import('webgpu').catch(()=>import('../local-deps/package/index.js'));
Object.assign(globalThis,globals);Object.defineProperty(globalThis,'navigator',{value:{gpu:create(process.env.DAWN_ADAPTER?['backend=d3d12','adapter='+process.env.DAWN_ADAPTER]:[])},configurable:true});
const realFetch=globalThis.fetch;globalThis.fetch=async(url,opts)=>String(url).startsWith('file:')?new Response(await fs.promises.readFile(new URL(url))):realFetch(url,opts);
const meta=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url))),gpu=await FlyGPU.create(meta,()=>{},{kernel:process.env.FLY_KERNEL||'tile16'});
const report=r=>{r={...r,kernel:gpu.kernel,decisionInterval:Number(process.env.FLY_INTERVAL||1)};console.log(JSON.stringify(r));fs.mkdirSync(new URL('../local-results/',import.meta.url),{recursive:true});fs.appendFileSync(new URL('../local-results/native.jsonl',import.meta.url),JSON.stringify({...r,adapter:gpu.adapterInfo,runtime:'Dawn 0.6.1 native D3D12'})+'\n');};
report({type:'init',adapter:gpu.adapterInfo});
if(process.argv.includes('--parity')){
 const rows=JSON.parse(fs.readFileSync(new URL('../model/parity.json',import.meta.url))).rows;let feature=0,mean=0,precision=0;const diff=(a,b)=>{const y=b.flat(Infinity);return Math.max(...a.flat(Infinity).map((v,i)=>Math.abs(v-y[i])));};
 for(const row of rows){const out=await gpu.step(row.obs);feature=Math.max(feature,diff(out.features,row.feature));mean=Math.max(mean,diff(twistToWrench(out.twist,row.obs,meta),row.mean));precision=Math.max(precision,diff(out.precision,row.precision));}
 report({type:'parity',ticks:rows.length,feature,mean,precision,timing:gpu.timing});
}else{
 const seeds=(process.argv[2]||'91000').split(',').map(Number),engine=new Engine(meta,gpu,{decisionInterval:Number(process.env.FLY_INTERVAL||1)});
 for(const seed of seeds){engine.reset(seed);let minClearance=Infinity,maxAngle=0;const start=performance.now();while(!engine.done){await engine.step();minClearance=Math.min(minClearance,clearance(engine.state,engine.walls,meta));maxAngle=Math.max(maxAngle,Math.abs(engine.state[2]));if(engine.tick%100===0)console.log(`seed=${seed} tick=${engine.tick} stage=${engine.stage}`);}
  report({type:'rollout',kernel:gpu.kernel,decisionInterval:engine.decisionInterval,inferences:engine.inferences,seed,ticks:engine.tick,success:engine.success,stage:engine.stage,contacts:engine.contacts,minClearance,maxAngle,seconds:(performance.now()-start)/1000,state:engine.state,timing:gpu.timing});
 }
}
gpu.device.destroy();delete globalThis.navigator;process.exit(0);
