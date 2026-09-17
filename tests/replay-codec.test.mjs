import test from 'node:test';
import assert from 'node:assert/strict';
import {encodeReplay,decodeReplay,packetStrengths} from '../replay-codec.mjs';
test('Replay preserves pose and real directed message timing; bounds brain-display error',()=>{
 const state=[-1,.25,.01,0,0,0].map(Math.fround),initial={state,stage:0},f={state:state.map(v=>Math.fround(v+.01)),stage:1,hit:false,done:true,success:true,wire_bytes:1024,forces:Array.from({length:8},()=>[.25,-.5]),brain_rms:Array(8).fill(.25),brain:[-.25,.003,1,-1],natural:[[[1,2,3],[3,2,1]],[[2,3,4],[4,3,2]]],precision:Array.from({length:2},()=>Array.from({length:2},()=>[1,1,1]))};
 const header={brainCount:4,sender:[0,1],receiver:[1,0],sweeps:2};const encoded=encodeReplay([initial,f],header),decoded=decodeReplay(encoded);
 assert.deepEqual(decoded.frames[1].state,f.state);assert.equal(decoded.frames[1].success,true);
 decoded.frames[1].brain.forEach((x,i)=>assert.ok(Math.abs(x-f.brain[i])<=1/65534));
 assert.deepEqual(decoded.frames[1].packets.map(p=>[p.at,p.u,p.v]),[[0,0,1],[0,1,0],[.5,0,1],[.5,1,0]]);
 decoded.frames[1].packets.forEach((p,i)=>assert.equal(p.strength,Math.fround(packetStrengths(f)[i])));
 assert.throws(()=>decodeReplay(encoded.subarray(0,encoded.length-1)),/Invalid replay/);
});
