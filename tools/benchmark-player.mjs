// An executable keyboard-controller reference, NOT a proof of optimal play.
// Uses the same gravity, speed cap, key quantization, swept collision and docking.
import fs from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {createRequire} from 'node:module';
import {decodeReplay} from '../replay-codec.mjs';
import {campaignMeta,playerTimeLowerBound} from '../campaign.mjs';
import {Engine} from '../engine.mjs';
const require=createRequire(import.meta.url),R=require('../race.js');
const root=new URL('../local-results/campaign/',import.meta.url),base=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url)));
const angle=x=>Math.atan2(Math.sin(x),Math.cos(x));
const moves=[[],['KeyD'],['KeyA'],['KeyW'],['KeyS'],['KeyD','KeyW'],['KeyD','KeyS'],['KeyA','KeyW'],['KeyA','KeyS']].map(keys=>({keys,u:R.inputs(new Set(keys))}));
export function referenceRun(frames,d,cap=.4){
 const route=[frames[0].state];let i=0;
 while(i<frames.length-1){let next=i+1;for(let j=Math.min(frames.length-1,i+60);j>i+1;j--)if(R.safeMotion(frames[i].state,frames[j].state,d)){next=j;break;}route.push(frames[next].state);i=next;}
 const r=R.create(d),carry=[0,0,0],commands=[];let waypoint=1,steps=0,bestProgress=0,stuck=0;
 while(!r.winner&&r.time<Math.max(120,frames.length*d.dt*4)){
  const q=r.state,target=route[waypoint];
  if(waypoint<route.length-1&&Math.hypot(target[0]-q[0],target[1]-q[1])<.009&&Math.abs(angle(target[2]-q[2]))<.012){waypoint++;continue;}
  let vx=6*(target[0]-q[0]-q[3]/R.SETTINGS.drag),vy=6*(target[1]-q[1]-q[4]/R.SETTINGS.drag),w=8*angle(target[2]-q[2]-q[5]/R.SETTINGS.turnDrag);
  const scale=Math.max(1,Math.hypot(vx,vy)/cap);vx/=scale;vy/=scale;w=Math.max(-.4,Math.min(.4,w));
  const desired=[vx/R.SETTINGS.speed,(vy+R.SETTINGS.gravity/R.SETTINGS.drag)/R.SETTINGS.speed,w/R.SETTINGS.turnSpeed];
  const norm=Math.max(1,Math.hypot(desired[0],desired[1]));desired[0]/=norm;desired[1]/=norm;
  for(let j=0;j<3;j++)carry[j]+=desired[j];
  let best=moves[0],cost=Infinity;for(const move of moves){const score=(carry[0]-move.u[0])**2+(carry[1]-move.u[1])**2;if(score<cost){best=move;cost=score;}}
  const keys=new Set(best.keys);carry[0]-=best.u[0];carry[1]-=best.u[1];const turn=carry[2]>.5?1:carry[2]<-.5?-1:0;carry[2]-=turn;if(turn)keys.add(turn>0?'KeyQ':'KeyE');
  const before=R.KEYS,mask=[...before].reduce((n,key,j)=>n|(keys.has(key)?1<<j:0),0);const last=commands.at(-1);if(last?.mask===mask)last.steps++;else commands.push({mask,steps:1});
  R.advance(r,d,keys,R.SETTINGS.step);steps++;
  if(waypoint>bestProgress){bestProgress=waypoint;stuck=0;}else stuck++;
  if(stuck>3600)break;
 }
 return{success:r.winner==='you',seconds:r.time,contacts:r.contacts,waypoint,waypoints:route.length,cap,final:r.state,commands,steps};
}
const files=process.argv.slice(2).length?process.argv.slice(2):fs.readdirSync(root).filter(f=>/^level-\d{3}\.json$/.test(f));
for(const name of files){
 const record=JSON.parse(fs.readFileSync(new URL(name,root))),{frames}=decodeReplay(new Uint8Array(gunzipSync(fs.readFileSync(new URL(record.file,root))))),m=campaignMeta(base,record.layout),d=new Engine(m,null).reset(record.seed);
 const trials=[];for(const cap of [.4,.25,.16]){const result=referenceRun(frames,d,cap);trials.push(result);if(result.success)break;}
 const best=trials.filter(x=>x.success).sort((a,b)=>a.seconds-b.seconds)[0];
 const output={level:record.level,sourceReplay:record.sha256,lowerBoundSeconds:playerTimeLowerBound(d,R.SETTINGS),success:!!best,best,trials:trials.map(({commands,...r})=>r),optimal:false,physics:R.SETTINGS};
 fs.writeFileSync(new URL(name.replace('.json','.player.json'),root),JSON.stringify(output));console.log(JSON.stringify({...output,best:best?{...best,commands:best.commands.length}:null}));
}
