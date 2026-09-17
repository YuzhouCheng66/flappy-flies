import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {Engine,observe,coordinate,transition,clearance,layoutFor} from '../engine.mjs';
import {createRequire} from 'node:module';
import {RACE_SPEED,raceDuration,canStream,TICK_BUDGET_MS} from '../race-clock.mjs';
import {selectAdapter} from '../gpu-device.mjs';
const require=createRequire(import.meta.url),C=require('../contracts.js'),R=require('../race.js');
const meta=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url))),engine=new Engine(meta,null);
test('Race speed never depends on inference time or display frame rate',()=>{
 assert.equal(RACE_SPEED,3.3);assert.equal(raceDuration(10),33);
 assert.equal(raceDuration(.2),.66); // no 60 ms render-frame clamp
 assert.ok(Math.abs(TICK_BUDGET_MS-24.2424242424)<1e-8);
 assert.equal(canStream(5),true);assert.equal(canStream(130),false);assert.equal(canStream(0),false);
});
test('Request hardware WebGPU, prefer exposed discrete GPU, never claim one not returned',async()=>{
 const fake=vendor=>({info:{vendor},limits:{maxStorageBufferBindingSize:134217728,maxBufferSize:268435456}}),requests=[];
 const result=await selectAdapter({requestAdapter:async opts=>{requests.push(opts);return fake(opts.powerPreference?'intel':'nvidia');}});
 assert.equal(requests[0].powerPreference,'high-performance');assert.equal(requests[0].forceFallbackAdapter,false);assert.equal(result.info.vendor,'nvidia');
 const intel=await selectAdapter({requestAdapter:async()=>fake('intel')});assert.equal(intel.info.vendor,'intel');assert.equal(intel.candidates.length,1);
 await assert.rejects(()=>selectAdapter(undefined),/WebGPU unavailable/);
 await assert.rejects(()=>selectAdapter({requestAdapter:async()=>({...fake('software'),isFallbackAdapter:true})}),/No hardware/);
});
test('The exact frozen full-network identity is retained',()=>{
 assert.equal(meta.neurons,165122);assert.equal(meta.agents,8);assert.equal(meta.neural_steps,4);
 assert.equal(meta.checkpoint_sha256,'d6e98e76ba8f8d45406cd40934a7fa6f9049565a137cb9df35a4b44925f717c3');
 assert.equal(meta.arrays['core.projection'].shape[0],2048);assert.equal(meta.output_ids.length,512);
});
test('Seeds are deterministic, safe initial conditions, and four genuine gates',()=>{
 for(const seed of [91000,...Array.from({length:100},(_,i)=>93000+i)]){
  assert.deepEqual(layoutFor(seed,meta),layoutFor(seed,meta));const d=engine.reset(seed);
  assert.equal(d.colliders.length,8);assert.ok(!C.collides(d.state,d));assert.ok(clearance(d.state,d.colliders,meta)>0);
  assert.ok(d.layout.apertures.every(v=>v>=.9&&v<=.94));
 }
});
test('Actual bidirectional GBP messages follow the radius graph',()=>{
 const d=engine.reset(91000),obs=observe(d.state,d.layout,0,engine.context,meta),mu=obs.map(()=>[1,.2,-.3]),p=obs.map(()=>[1,2,3]);
 const g=coordinate(mu,p,obs,meta);
 for(let j=0;j<g.sender.length;j++){
  assert.equal(g.sender[j],g.receiver[j^1]);const a=d.handles[g.sender[j]],b=d.handles[g.receiver[j]];
  assert.ok(Math.hypot(a[0]-b[0],a[1]-b[1])<=.4+1e-6);
 }
 assert.equal(g.natural.length,4);assert.ok(g.forces.every(f=>Math.hypot(...f)<=4+1e-5));
});
test('Player and flies share conservative collision authority',()=>{
 const d=engine.reset(91000),a=[-.8,1.2,0,0,0,0],b=[.8,1.2,0,0,0,0];
 assert.equal(R.safeMotion(a,b,d),false);assert.equal(R.safeMotion(d.state,d.state,d),true);
 assert.equal(C.collides([0,1.2,0,0,0,0],d),true);
});
test('Race pauses, awards first arrival, and ignores future prefetched winners',()=>{
 const d=engine.reset(91000),race=R.create(d);R.advance(race,d,new Set(),0,10);assert.equal(race.time,0);assert.equal(race.winner,null);
 R.advance(race,d,new Set(),.1,10);assert.equal(race.winner,null);R.advance(race,d,new Set(),.1,.15);assert.equal(race.winner,'flies');
 const final=JSON.stringify(race);R.advance(race,d,new Set(['KeyD']),1);assert.equal(JSON.stringify(race),final);
});
test('Human docking accepts 15 degree alignment but rejects larger errors',()=>{
 const d=engine.reset(91000),goal=d.layout.goals.at(-1);
 for(const [degrees,expected] of [[12,'you'],[-12,'you'],[20,null],[-20,null]]){
  const r=R.create(d);r.state=[goal[0],goal[1],goal[2]+degrees*Math.PI/180,0,0,0];
  assert.equal(C.collides(r.state,d),false);R.advance(r,d,new Set(),.45);assert.equal(r.winner,expected);
 }
});
test('Human rotation is slower and stops promptly on release',()=>{
 const d=engine.reset(91000),r=R.create(d);R.advance(r,d,new Set(['KeyQ']),.3);
 assert.ok(r.state[5]>.39&&r.state[5]<=.4);const before=r.state[2];
 R.advance(r,d,new Set(),.15);assert.ok(Math.abs(r.state[5])<.012);assert.ok(Math.abs(r.state[2]-before)<.018);
});
const fixture=new URL('../model/parity.json',import.meta.url);
test('Independent Python fixtures: local sensors, directed GBP and physics',{skip:!fs.existsSync(fixture)},()=>{
 const rows=JSON.parse(fs.readFileSync(fixture)).rows;let stats=null;const errors=Array(27).fill(0);let force=0,state=0;
 engine.reset(91000);
 for(const r of rows){
  const obs=observe(r.state,engine.layout,r.stage_before,r.context,meta);
  obs.forEach((o,i)=>o.forEach((x,k)=>errors[k]=Math.max(errors[k],Math.abs(x-r.obs[i][k]))));
  const g=coordinate(r.mean,r.precision,r.obs,meta,stats);stats=g.stats;
  g.forces.flat().forEach((x,i)=>force=Math.max(force,Math.abs(x-r.next.forces.flat()[i])));
  const t=transition(r.state,r.next.forces,engine.walls,meta);t.state.forEach((x,i)=>state=Math.max(state,Math.abs(x-r.next.state[i])));
  assert.equal(t.hit,r.next.hit);
 }
 assert.ok(Math.max(...errors.slice(0,18))<2e-6);assert.ok(Math.max(...errors)<.002);assert.ok(force<2e-6);assert.ok(state<2e-6);
 console.log({observationMax:Math.max(...errors),forceMax:force,stateMax:state,ticks:rows.length});
});
