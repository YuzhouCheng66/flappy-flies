import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {FlyGPU} from '../gpu-model.mjs';
import {Engine,clearance} from '../engine.mjs';
import {campaignLayout,campaignMeta} from '../campaign.mjs';
const {create,globals}=await import('../local-deps/package/index.js');Object.assign(globalThis,globals);Object.defineProperty(globalThis,'navigator',{value:{gpu:create(['backend=d3d12','adapter=NVIDIA'])},configurable:true});const realFetch=fetch;globalThis.fetch=async(url,opts)=>String(url).startsWith('file:')?new Response(await fs.promises.readFile(new URL(url))):realFetch(url,opts);
const base=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url))),isOriginal=process.argv[2]==='original',parent=isOriginal?{weightsSha256:base.weights_sha256}:JSON.parse(fs.readFileSync(process.argv[2])),weightsURL=isOriginal?undefined:new URL(parent.weightsFile,pathToFileURL(process.argv[2])).href,meta={...base,weights_sha256:parent.weightsSha256},gpu=await FlyGPU.create(meta,()=>{},{kernel:'tile16',precision:'f32',profile:false,weightsURL});
const levels=(process.argv[3]||'1,21,41,60,61,63,71,81,83,91,99,100').split(',').map(Number),rows=[],attempt=process.argv[4]===undefined?null:Number(process.argv[4]);if(attempt!==null&&(!Number.isInteger(attempt)||attempt<0))throw Error('Invalid evaluation attempt');
try{for(const level of levels){
 let layout;const accepted=new URL(`../local-results/campaign/level-${String(level).padStart(3,'0')}.json`,import.meta.url);
 if(attempt===null&&level<=60&&fs.existsSync(accepted))layout=JSON.parse(fs.readFileSync(accepted)).layout;else layout=campaignLayout(level,attempt??0);
 const m=campaignMeta(meta,layout),e=new Engine(m,gpu);e.reset(layout.seed);let margin=Infinity,pathLength=0,rotation=0,backwardDistance=0,forceDeltaSquared=0,priorForces=null,stalledTicks=0;const start=performance.now();
 while(!e.done){const q=e.state;const f=await e.step();margin=Math.min(margin,clearance(e.state,e.walls,m));const dx=e.state[0]-q[0],dy=e.state[1]-q[1];pathLength+=Math.hypot(dx,dy);backwardDistance+=Math.max(0,-dx);rotation+=Math.abs(Math.atan2(Math.sin(e.state[2]-q[2]),Math.cos(e.state[2]-q[2])));stalledTicks+=+(Math.hypot(dx,dy)<1e-5);if(priorForces)for(let i=0;i<8;i++)for(let j=0;j<2;j++)forceDeltaSquared+=(f.forces[i][j]-priorForces[i][j])**2;priorForces=f.forces;}
 const r={level,seed:layout.seed,success:e.success,stage:e.stage,ticks:e.tick,contacts:e.contacts,minClearance:margin,pathLength,rotationRad:rotation,backwardDistance,stalledTicks,forceChangeRms:Math.sqrt(forceDeltaSquared/(16*Math.max(1,e.tick-1))),final:e.state,seconds:(performance.now()-start)/1000};rows.push(r);console.log(JSON.stringify(r));
 const progress=new URL('../local-results/full-campaign-training/evaluation-progress.json',import.meta.url);fs.writeFileSync(progress,JSON.stringify({model:parent.weightsSha256,requested:levels,complete:rows.length===levels.length,rows}));
}}finally{gpu.device.destroy();delete globalThis.navigator;}
const result={model:parent.weightsSha256,checkpoint:process.argv[2],attempt,fullBrain:true,runtime:'native RTX4080',successes:rows.filter(x=>x.success).length,total:rows.length,rows};const output=isOriginal?new URL('../local-results/full-campaign-training/original-eval.json',import.meta.url):process.argv[2].replace('.json',`${attempt===null?'':'.attempt-'+attempt}.eval.json`);fs.writeFileSync(output,JSON.stringify(result,null,2));
const archive=new URL('../local-results/full-campaign-training/evaluations/',import.meta.url);fs.mkdirSync(archive,{recursive:true});fs.writeFileSync(new URL(`${parent.weightsSha256.slice(0,12)}-${createHash('sha256').update(JSON.stringify({levels,attempt})).digest('hex').slice(0,8)}.json`,archive),JSON.stringify(result,null,2));
console.log(JSON.stringify({successes:result.successes,total:result.total}));process.exit(0);
