import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {CompactModel} from '../compact-model.mjs';
import {Engine,clearance} from '../engine.mjs';
const modelPath=process.argv[3]||new URL('../local-results/compact/model.json',import.meta.url);
const config=JSON.parse(fs.readFileSync(modelPath)),gpu=new CompactModel(config),meta=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url)));
const modelHash=createHash('sha256').update(fs.readFileSync(modelPath)).digest('hex');
const parityPath=process.argv[3]?new URL(`../local-results/compact/parity-${modelHash.slice(0,12)}.json`,import.meta.url):new URL('../local-results/compact/parity.json',import.meta.url);
const parity=JSON.parse(fs.readFileSync(parityPath));let error=0;for(let t=0;t<parity.normalized_obs.length;t++){const output=gpu.agent(parity.normalized_obs[t].map(x=>Math.max(-8,Math.min(8,x))),0);for(let j=0;j<6;j++)error=Math.max(error,Math.abs(output[j]-parity.normalized_output[t][j]));}console.log(JSON.stringify({type:'parity',modelHash,error}));if(error>1e-4)throw Error('Compact Python/JS parity failed');
const engine=new Engine(meta,gpu),seeds=process.argv[2]?process.argv[2].split(',').map(Number):Array.from({length:20},(_,i)=>97000+i);
for(const seed of seeds){if(config.train_seeds.includes(seed)||config.validation_seeds.includes(seed))throw Error('Test seed leakage');engine.reset(seed);const t=performance.now();let minClearance=Infinity;const timings=[];while(!engine.done){await engine.step();timings.push(gpu.timing.wall);minClearance=Math.min(minClearance,clearance(engine.state,engine.walls,meta));}
const r={modelHash,seed,success:engine.success,ticks:engine.tick,stage:engine.stage,minClearance,seconds:(performance.now()-t)/1000,inferenceMean:timings.reduce((a,b)=>a+b,0)/timings.length};console.log(JSON.stringify(r));fs.appendFileSync(new URL('../local-results/compact/rollouts.jsonl',import.meta.url),JSON.stringify(r)+'\n');}
