import test from 'node:test';import assert from 'node:assert/strict';
import {chooseMeasuredBackend} from '../backend-selector.mjs';
test('Measured backend must meet the deadline and be independently validated',()=>{
 const candidates=[{id:'branded GPU',validated:true,medianMs:100,p95Ms:120},{id:'CPU',validated:true,medianMs:8,p95Ms:10},{id:'unvalidated compact',validated:false,medianMs:1,p95Ms:2}];
 assert.equal(chooseMeasuredBackend(candidates).id,'CPU');assert.equal(chooseMeasuredBackend(candidates.slice(0,1)),null);
 assert.equal(chooseMeasuredBackend([{id:'jitter',validated:true,medianMs:4,p95Ms:25}]),null);
 assert.equal(chooseMeasuredBackend([{id:'empty',validated:true,medianMs:0,p95Ms:0}]),null);
});
