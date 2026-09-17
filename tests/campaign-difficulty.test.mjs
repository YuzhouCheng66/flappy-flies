import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {campaignDifficulty} from '../campaign-difficulty.mjs';
const R=createRequire(import.meta.url)('../race.js');
test('Difficulty tightens smoothly; only flies are slowed by the arcade factor',()=>{
 const start=campaignDifficulty(1),end=campaignDifficulty(30);
 assert.equal(start.player_dock_radius,.18);assert.equal(start.fly_speed_factor,.8);
 assert.ok(Math.abs(end.player_dock_radius-.03)<1e-12);assert.equal(end.fly_speed_factor,1);
 for(let level=2;level<=30;level++){
  const a=campaignDifficulty(level-1),b=campaignDifficulty(level);
  assert.ok(b.player_dock_angle<a.player_dock_angle);assert.ok(b.player_dock_radius<a.player_dock_radius);
  assert.ok(Math.abs(b.fly_speed_factor-a.fly_speed_factor-.2/29)<1e-12);
 }
});
test('Approximate early docking wins, the same offset fails late; collision checks remain',()=>{
 const base={state:[.12,0,20*Math.PI/180,0,0,0],snapshot:{tick:0},dt:.08,rectangles:[[[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]]],world_x:10,world_y:10,colliders:[],layout:{goals:[[0,0,0]],wall_x:[]}};
 const early={...base,...campaignDifficulty(1)},late={...base,...campaignDifficulty(30)};
 const a=R.create(early),b=R.create(late);R.advance(a,early,new Set(),.2);R.advance(b,late,new Set(),.2);
 assert.equal(a.winner,'you');assert.equal(b.winner,null);
 assert.equal(R.SETTINGS.speed,.40);assert.equal(R.SETTINGS.gravity,.24);
});
