import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {campaignLayout,campaignMeta,playerTimeLowerBound} from '../campaign.mjs';
import {Engine,observe} from '../engine.mjs';
const base=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url))),require=createRequire(import.meta.url),R=require('../race.js'),C=require('../contracts.js');
test('100 ordered candidates have 1–5 gates, mixed widths and stricter docking',()=>{
 for(let l=1;l<=100;l++){
  const layout=campaignLayout(l),m=campaignMeta(base,layout),e=new Engine(m,null),d=e.reset(layout.seed);
  assert.deepEqual(layout,campaignLayout(l));assert.equal(layout.wall_x.length,Math.ceil(l/20));
  assert.equal(C.collides(d.state,d),false);assert.ok(d.layout.goals.at(-1)[0]<d.world_x);
  const obs=observe(e.state,e.layout,0,e.context,m);assert.ok(obs.every(o=>o.length===27&&o.every(Number.isFinite)));
  assert.ok(playerTimeLowerBound(d,R.SETTINGS)>0);
 }
 assert.equal(campaignLayout(1).player_dock_degrees,15);assert.equal(campaignLayout(100).player_dock_degrees,3.5);
});
test('Variable-gate engine ends at the last goal, and retains the final goal',async()=>{
 for(const level of [1,41,100]){
  const l=campaignLayout(level),m=campaignMeta(base,l),gpu={reset(){},async step(){return{twist:Array.from({length:8},()=>[0,0,0]),precision:Array.from({length:8},()=>[1,1,1])};}},e=new Engine(m,gpu);e.reset(l.seed);e.stage=l.goals.length-1;e.state=[...l.goals.at(-1),0,0,0];e.stable=4;
  await e.step();assert.equal(e.success,true);assert.deepEqual(e.snapshot().goal,l.goals.at(-1));
 }
});
test('Opt-in rear-clearance progression does not waive final docking',async()=>{
 const l=campaignLayout(21);l.intermediate_transition='rear-clearance';const m=campaignMeta(base,l),gpu={reset(){},async step(){return{twist:Array.from({length:8},()=>[0,0,0]),precision:Array.from({length:8},()=>[1,1,1])};}},e=new Engine(m,gpu);e.reset(l.seed);e.state=[.7,0,0,.1,0,0];
 await e.step();assert.equal(e.stage,1);assert.equal(e.success,false);
 await e.step();assert.equal(e.success,false);
});
