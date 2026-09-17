// Node/V8 WASM test, NOT a browser performance claim.
import fs from 'node:fs';
import {FlyWASM} from '../wasm-model.mjs';
import {Engine,twistToWrench,clearance} from '../engine.mjs';
const realFetch=globalThis.fetch;globalThis.fetch=async(url,opts)=>String(url).startsWith('file:')?new Response(await fs.promises.readFile(new URL(url))):realFetch(url,opts);
const meta=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url))),t=performance.now();
const gpu=await FlyWASM.create(meta,()=>{},{threads:Number(process.env.FLY_THREADS||8)});
const report=r=>{r={...r,runtime:'Node V8 WASM SIMD',threads:gpu.threads};console.log(JSON.stringify(r));fs.appendFileSync(new URL('../local-results/wasm.jsonl',import.meta.url),JSON.stringify(r)+'\n');};
report({type:'init',seconds:(performance.now()-t)/1000});
if(process.argv.includes('--parity')){
 const rows=JSON.parse(fs.readFileSync(new URL('../model/parity.json',import.meta.url))).rows.slice(0,Number(process.env.FLY_SAMPLES||80));let features=0,means=0,precisions=0;const timings=[];
 const diff=(a,b)=>{const flat=b.flat(Infinity);return Math.max(...a.flat(Infinity).map((v,i)=>Math.abs(v-flat[i])));};
 for(const row of rows){const out=await gpu.step(row.obs);timings.push(gpu.timing.wall);features=Math.max(features,diff(out.features,row.feature));means=Math.max(means,diff(twistToWrench(out.twist,row.obs,meta),row.mean));precisions=Math.max(precisions,diff(out.precision,row.precision));}
 report({type:'parity',ticks:rows.length,features,means,precisions,timings,lastParts:gpu.timing.parts});
 if(features>3e-5||means>.003||precisions>.001)process.exitCode=1;
}else{
 const engine=new Engine(meta,gpu);
 for(const seed of (process.argv[2]||'91000').split(',').map(Number)){
  engine.reset(seed);let minClearance=Infinity;const start=performance.now();while(!engine.done){await engine.step();minClearance=Math.min(minClearance,clearance(engine.state,engine.walls,meta));}
  report({type:'rollout',seed,ticks:engine.tick,success:engine.success,minClearance,seconds:(performance.now()-start)/1000});
 }
}
gpu.device.destroy();process.exit(process.exitCode||0);
