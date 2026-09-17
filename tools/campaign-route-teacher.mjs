// TRAINING ONLY: recovery waypoints from audited successful single-gate paths.
// No runtime policy may import this module.
import fs from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {createRequire} from 'node:module';
import {createHash} from 'node:crypto';
import {decodeReplay} from '../replay-codec.mjs';
const require=createRequire(import.meta.url),R=require('../race.js');
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
const distance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1],.4*wrap(a[2]-b[2]));

export function createRouteTeacher(){
 const root=new URL('../local-results/campaign/',import.meta.url),routes=[],sources=[];
 for(const level of [21,30,41,50,60]){
  const record=JSON.parse(fs.readFileSync(new URL(`level-${String(level).padStart(3,'0')}.json`,root)));
  const packed=fs.readFileSync(new URL(record.file,root));if(createHash('sha256').update(packed).digest('hex')!==record.sha256)throw Error('Recovery source hash mismatch');
  const {frames}=decodeReplay(new Uint8Array(gunzipSync(packed)));
  sources.push({level,sha256:record.sha256});
  for(let stage=0;stage<record.layout.wall_x.length;stage++)if(record.layout.apertures[stage]<1.2){
   const x=record.layout.wall_x[stage],y=record.layout.gap_y[stage];
   const path=frames.filter(f=>f.stage===stage).map(f=>[f.state[0]-x,f.state[1]-y,f.state[2]]);
   if(path.length>10)routes.push(path.filter((_,i)=>i%3===0));
  }
 }
 return{sources,query(q,description,stage){
  const x=description.layout.wall_x[stage],y=description.layout.gap_y[stage],local=[q[0]-x,q[1]-y,q[2]];
  const candidates=[];
  for(const path of routes){
   let nearest=0,score=Infinity;
   for(let i=0;i<path.length;i++){const d=distance(local,path[i]);if(d<score){score=d;nearest=i;}}
   // Stay near a real successful route; do not bridge an obstacle with a
   // fictitious interpolated teacher. A backward recovery target is allowed.
   if(score>.55)continue;
   let forward=nearest,travel=0;
   while(forward+1<path.length&&travel<.10){travel+=distance(path[forward],path[forward+1]);forward++;}
   for(const [index,penalty] of [[forward,0],[nearest,.03],[Math.max(0,nearest-4),.08]]){
    const p=path[index],target=[p[0]+x,p[1]+y,p[2],0,0,0];
    if(distance(q,target)<.015)continue;
    candidates.push({target,score:score+penalty});
   }
  }
  candidates.sort((a,b)=>a.score-b.score);
  for(const c of candidates)if(R.safeMotion(q,c.target,description))return c.target.slice(0,3);
  return null;
 }};
}
