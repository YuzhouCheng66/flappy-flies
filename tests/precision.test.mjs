import test from 'node:test';
import assert from 'node:assert/strict';
import {FlyGPU} from '../gpu-model.mjs';

test('Unknown neural storage formats fail closed before requesting a device',async()=>{
 await assert.rejects(()=>FlyGPU.create({},()=>{},{precision:'int4'}),/Unknown neural state precision/);
});
test('FP16 does not silently run as FP32 when shader-f16 is unavailable',async()=>{
 const previous=Object.getOwnPropertyDescriptor(globalThis,'navigator');let deviceRequested=false;
 const adapter={info:{vendor:'test'},features:new Set(),limits:{maxStorageBufferBindingSize:134217728,maxBufferSize:268435456},requestDevice:()=>{deviceRequested=true;throw Error('must not request');}};
 Object.defineProperty(globalThis,'navigator',{value:{gpu:{requestAdapter:async()=>adapter}},configurable:true});
 try{await assert.rejects(()=>FlyGPU.create({},()=>{},{precision:'f16'}),/requires shader-f16/);assert.equal(deviceRequested,false);}
 finally{if(previous)Object.defineProperty(globalThis,'navigator',previous);else delete globalThis.navigator;}
});
test('Odd recurrence counts cannot silently read the wrong ping-pong buffer',async()=>{
 await assert.rejects(()=>FlyGPU.prototype.init.call({meta:{neurons:1,neural_steps:3},options:{}}),/even number/);
});
