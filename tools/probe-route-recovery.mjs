// CPU physics feasibility of the TRAINING oracle from actual failed states.
// Its successes must never be reported as learned model successes.
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {Engine} from '../engine.mjs';
import {campaignMeta,campaignLayout} from '../campaign.mjs';
import {createRouteTeacher} from './campaign-route-teacher.mjs';
const require=createRequire(import.meta.url),R=require('../race.js'),teacher=createRouteTeacher();
const base=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url))),audit=JSON.parse(fs.readFileSync(new URL('../local-results/full-campaign-training/evaluation-progress.json',import.meta.url))),rows=[];
for(const row of audit.rows.filter(r=>!r.success)){
 let layout=campaignLayout(row.level);const accepted=new URL(`../local-results/campaign/level-${String(row.level).padStart(3,'0')}.json`,import.meta.url);if(row.level<=60&&fs.existsSync(accepted))layout=JSON.parse(fs.readFileSync(accepted)).layout;
 let e;let missing=0;
 const policy={reset(){},async step(){
  const q=e.state,goal=e.layout.goals[e.stage],d=e.description(),direct=R.safeMotion(q,[...goal,0,0,0],d),target=direct?goal:teacher.query(q,d,e.stage);
  let v=[0,0,0];if(target){const c=Math.cos(q[2]),s=Math.sin(q[2]),dx=target[0]-q[0],dy=target[1]-q[1],gain=direct?.9:3;v=[gain*(c*dx+s*dy),gain*(-s*dx+c*dy),(direct?1.2:3)*Math.atan2(Math.sin(target[2]-q[2]),Math.cos(target[2]-q[2]))];const ratio=Math.max(1,Math.hypot(v[0],v[1])/.18,Math.abs(v[2])/.4);v=v.map(x=>x/ratio);}else missing++;
  return{twist:Array.from({length:8},()=>[v[0]/.65,v[1]/.65,v[2]/1.4]),precision:Array.from({length:8},()=>[1,1,1])};
 }};
 e=new Engine(campaignMeta(base,layout),policy);e.reset(layout.seed);e.state=row.final.slice();e.stage=row.stage;
 while(!e.done)await e.step();
 const result={level:row.level,sourceModel:audit.model,teacherOnly:true,success:e.success,ticks:e.tick,contacts:e.contacts,missingLabels:missing,final:e.state};rows.push(result);console.log(JSON.stringify(result));
}
fs.writeFileSync(new URL('../local-results/full-campaign-training/recovery-oracle-probe.json',import.meta.url),JSON.stringify({teacherOnly:true,rows},null,2));
