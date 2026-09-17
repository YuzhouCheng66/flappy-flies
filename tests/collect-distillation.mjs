// Recorded local observations + original model proposals for a labelled compact
// student. No global poses, stage IDs, maps, seeds, or future states are inputs.
import fs from 'node:fs';
import {FlyGPU} from '../gpu-model.mjs';
import {Engine,observe} from '../engine.mjs';
const {create,globals}=await import('../local-deps/package/index.js');Object.assign(globalThis,globals);
Object.defineProperty(globalThis,'navigator',{value:{gpu:create(['backend=d3d12','adapter=NVIDIA'])},configurable:true});
const realFetch=fetch;globalThis.fetch=async(url,opts)=>String(url).startsWith('file:')?new Response(await fs.promises.readFile(new URL(url))):realFetch(url,opts);
const meta=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url))),gpu=await FlyGPU.create(meta),engine=new Engine(meta,gpu);
const folder=new URL('../local-results/distillation/',import.meta.url);fs.mkdirSync(folder,{recursive:true});
const seeds=Array.from({length:Number(process.env.DATA_SEEDS||48)},(_,i)=>94000+i);
for(const seed of seeds){const target=new URL(`seed-${seed}.json`,folder);if(fs.existsSync(target))continue;
 engine.reset(seed);const rows=[],t=performance.now();while(!engine.done){const obs=observe(engine.state,engine.layout,engine.stage,engine.context,meta);await engine.step();rows.push({obs,twist:engine.neural.twist,precision:engine.neural.precision});}
 const result={seed,teacher:'full frozen connectome + GRU, every tick',checkpoint:meta.checkpoint_sha256,success:engine.success,ticks:engine.tick,rows};fs.writeFileSync(target,JSON.stringify(result));console.log(JSON.stringify({seed,success:engine.success,ticks:engine.tick,seconds:(performance.now()-t)/1000}));
}gpu.device.destroy();delete globalThis.navigator;process.exit(0);
