// DAgger data for the FULL connectome decoder. No compact policy substitution.
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {FlyGPU} from '../gpu-model.mjs';
import {Engine} from '../engine.mjs';
import {campaignLayout,campaignMeta} from '../campaign.mjs';
import {createRouteTeacher} from './campaign-route-teacher.mjs';
const require=createRequire(import.meta.url),R=require('../race.js');
const {create,globals}=await import('../local-deps/package/index.js');Object.assign(globalThis,globals);Object.defineProperty(globalThis,'navigator',{value:{gpu:create(['backend=d3d12','adapter=NVIDIA'])},configurable:true});
const nativeFetch=fetch;globalThis.fetch=async(url,opts)=>String(url).startsWith('file:')?new Response(await fs.promises.readFile(new URL(url))):nativeFetch(url,opts);
const args=Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('='))),base=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url))),dataRoot=new URL('../local-results/full-campaign-training/',import.meta.url);fs.mkdirSync(dataRoot,{recursive:true});
let modelMeta=base,weightsURL;
if(args.parent){const parent=JSON.parse(fs.readFileSync(args.parent));modelMeta={...base,weights_sha256:parent.weightsSha256};weightsURL=new URL(parent.weightsFile,pathToFileURL(args.parent)).href;}
const gpu=await FlyGPU.create(modelMeta,()=>{},{kernel:'tile16',precision:'f32',profile:false,weightsURL}),round=Number(args.round||0),levels=(args.levels||'1,10,21,30,41,50,60,61,65,69,73,77,81,85,89,93,97,63,71,83,91,99').split(',').map(Number),mode=args.mode||'guided';
const routeTeacher=Object.hasOwn(args,'route-teacher')?createRouteTeacher():null;
if(gpu.adapterInfo.vendor!=='nvidia')throw Error('Expected local NVIDIA');
try{for(const level of levels){
 const split=[63,71,83,91,99].includes(level)?'validation':'train',layout=campaignLayout(level,Number(args.attempt??round)),m=campaignMeta(base,layout),rows=[];let engine,teacherCount=0,routeCount=0,recoveryActive=false,everContact=false,interventions=0;
 const policy={reset(){gpu.reset();},async step(obs){
  const pred=await gpu.step(obs),q=engine.state,goal=engine.layout.goals[engine.stage],d=engine.description(),target=[...goal,0,0,0];
  // Privileged geometry is ONLY used to certify this training label. Student
  // input is unchanged: local observations -> full brain -> local readout.
  const direct=(level>60||Object.hasOwn(args,'label-all-safe'))&&R.safeMotion(q,target,d);if(engine.context.hit){recoveryActive=true;everContact=true;}if(direct)recoveryActive=false;
  const allowRoute=!Object.hasOwn(args,'recovery-only')||recoveryActive,waypoint=direct?goal:allowRoute?routeTeacher?.query(q,d,engine.stage):null,usable=!!waypoint;let labelled=pred.rawProposal,action=pred;
  if(usable){
   const dx=waypoint[0]-q[0],dy=waypoint[1]-q[1],theta=Math.atan2(Math.sin(waypoint[2]-q[2]),Math.cos(waypoint[2]-q[2])),c=Math.cos(q[2]),s=Math.sin(q[2]),gain=direct?.9:3;
   let vx=gain*(c*dx+s*dy),vy=gain*(-s*dx+c*dy),w=(direct?1.2:3)*theta;const scale=Math.max(1,Math.hypot(vx,vy)/.18,Math.abs(w)/.4);vx/=scale;vy/=scale;w/=scale;if(!direct)routeCount++;
   labelled=Array.from({length:8},()=>[vx*.08,vy*.08,w*.08,0,0,0]);teacherCount++;
   if(mode==='guided'||(mode==='recovery'&&everContact)){action={...pred,twist:Array.from({length:8},()=>[vx/.65,vy/.65,w/1.4]),precision:Array.from({length:8},()=>[1,1,1])};interventions++;}
  }
  const row=new Float32Array(8*525);for(let a=0;a<8;a++){row.set(pred.features[a],a*525);row.set(labelled[a],a*525+512);row.set(pred.rawProposal[a],a*525+518);row[a*525+524]=usable?(direct?1:2):0;}rows.push(row);return action;
 }};
 engine=new Engine(m,policy);engine.reset(layout.seed);const start=performance.now();while(!engine.done)await engine.step();
 const stem=`r${round}-${mode}-${level}`,bin=new Float32Array(rows.length*8*525);rows.forEach((r,i)=>bin.set(r,i*8*525));const bytes=Buffer.from(bin.buffer);fs.writeFileSync(new URL(stem+'.bin',dataRoot),bytes);
 const record={schema:'full-brain-decoder-dagger-v1',file:stem+'.bin',sha256:createHash('sha256').update(bytes).digest('hex'),round,level,split,mode,interventions,labelAllSafe:Object.hasOwn(args,'label-all-safe'),routeTeacherSources:routeTeacher?.sources,recoveryOnly:Object.hasOwn(args,'recovery-only'),routeCount,studentWeights:gpu.meta.weights_sha256,fullConnectome:true,columns:525,agents:8,ticks:engine.tick,success:engine.success,stage:engine.stage,teacherCount,contacts:engine.contacts,seconds:(performance.now()-start)/1000,layout};fs.writeFileSync(new URL(stem+'.json',dataRoot),JSON.stringify(record));console.log(JSON.stringify(record));
}}finally{gpu.device.destroy();delete globalThis.navigator;}process.exit(0);
