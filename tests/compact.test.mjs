import test from 'node:test';
import assert from 'node:assert/strict';
import {CompactModel} from '../compact-model.mjs';

function fixture(){
 const sizes={'embed.0.weight':128*27,'embed.0.bias':128,'gru.weight_ih_l0':384*128,'gru.weight_hh_l0':384*128,'gru.bias_ih_l0':384,'gru.bias_hh_l0':384,'head.0.weight':128*128,'head.0.bias':128,'head.2.weight':6*128,'head.2.bias':6};
 return {hidden:128,xmean:Array(27).fill(0),xstd:Array(27).fill(1),ymean:Array(6).fill(0),ystd:Array(6).fill(1),parameters:Object.fromEntries(Object.entries(sizes).map(([k,n])=>[k,{data:Array(n).fill(k.includes('weight')?.001:0)}]))};
}
test('Compact memory is recurrent, local to each agent, and resettable',()=>{
 const model=new CompactModel(fixture()),obs=Array(27).fill(.4);
 model.agent(obs,0);const first=Array.from(model.hidden[0]);
 assert.ok(first.some(x=>x!==0));assert.ok(model.hidden[1].every(x=>x===0));
 model.agent(obs,0);assert.notDeepEqual(Array.from(model.hidden[0]),first);
 model.agent(obs,1);assert.deepEqual(Array.from(model.hidden[1]),first);
 model.reset();assert.ok(model.hidden.every(h=>h.every(x=>x===0)));
});
test('Compact diagnostics report actual GRU state, not fabricated full brain activity',async()=>{
 const model=new CompactModel(fixture()),out=await model.step(Array.from({length:8},()=>Array(27).fill(.4)));
 assert.equal(out.twist.length,8);assert.equal(out.precision.length,8);
 assert.equal(out.brain.length,128);assert.deepEqual(out.brain,Array.from(model.hidden[0]));
 assert.ok(out.precision.every(p=>p.every(x=>Number.isFinite(x)&&x>0)));
});
