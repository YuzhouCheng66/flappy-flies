// Genuine model-induced DAgger: labels queried on states reached by the compact
// policy, with teacher recurrent memory advanced on that same observation stream.
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {FlyGPU} from '../gpu-model.mjs';import {CompactModel} from '../compact-model.mjs';import {Engine} from '../engine.mjs';
const {create,globals}=await import('../local-deps/package/index.js');Object.assign(globalThis,globals);Object.defineProperty(globalThis,'navigator',{value:{gpu:create(['backend=d3d12','adapter=NVIDIA'])},configurable:true});
const realFetch=fetch;globalThis.fetch=async(url,opts)=>String(url).startsWith('file:')?new Response(await fs.promises.readFile(new URL(url))):realFetch(url,opts);
const meta=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url))),teacher=await FlyGPU.create(meta),student=new CompactModel(JSON.parse(fs.readFileSync(new URL('../local-results/compact/model.json',import.meta.url))));
const round=Number(process.env.DAGGER_ROUND||1),folder=new URL('../local-results/distillation/',import.meta.url);let rows=[];
const studentBytes=fs.readFileSync(new URL('../local-results/compact/model.json',import.meta.url)),studentHash=createHash('sha256').update(studentBytes).digest('hex');
fs.writeFileSync(new URL(`../local-results/compact/model-${studentHash.slice(0,12)}.json`,import.meta.url),studentBytes);
const policy={reset(){teacher.reset();student.reset();},async step(obs){const target=await teacher.step(obs),pred=await student.step(obs);rows.push({obs,twist:target.twist,precision:target.precision});return pred;}};
const engine=new Engine(meta,policy);
for(const seed of Array.from({length:32},(_,i)=>94000+i).filter(s=>s%6!==5)){
 rows=[];engine.reset(seed);const start=performance.now();while(!engine.done)await engine.step();
 fs.writeFileSync(new URL(`dagger-${round}-${seed}.json`,folder),JSON.stringify({seed,modelInduced:true,round,studentHash,teacherCheckpoint:meta.checkpoint_sha256,success:engine.success,ticks:engine.tick,rows}));console.log(JSON.stringify({round,seed,studentSuccess:engine.success,ticks:engine.tick,seconds:(performance.now()-start)/1000}));
}teacher.device.destroy();delete globalThis.navigator;process.exit(0);
