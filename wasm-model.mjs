import createModule from './wasm/full-model.mjs';
import {loadGraph,sha256} from './graph-loader.mjs';
const keys=['input_mean','input_std','target_mean','target_std','net.0.weight','net.0.bias','net.2.weight','net.2.bias','net.4.weight','net.4.bias','embed.0.weight','embed.0.bias','gru.weight_ih_l0','gru.weight_hh_l0','gru.bias_ih_l0','gru.bias_hh_l0','residual.weight','residual.bias','core.projection','core.bias','core.obs_mean','core.obs_std'];
export class FlyWASM{
 static async create(meta,onProgress=()=>{},options={}){
  if(typeof process==='undefined'&&globalThis.crossOriginIsolated!==true)throw Error('WASM threads require cross-origin isolation (COOP/COEP).');
  if(options.threads!==undefined&&(!Number.isInteger(options.threads)||options.threads<1||options.threads>8))throw Error('WASM Worker count must be 1–8 for this build.');
  const mod=await createModule(),graph=await loadGraph(new URL('./graph/manifest.json',import.meta.url).href,{onProgress});
  const weights=await(await fetch(new URL('./model/weights.bin',import.meta.url))).arrayBuffer();
  if(await sha256(weights)!==meta.weights_sha256)throw Error('Model weight checksum mismatch');
  const put=data=>{const p=mod._malloc(data.byteLength);if(!p)throw Error('WASM memory exhausted');mod.HEAPU8.set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength),p);return p;};
  const packed=new Uint32Array(graph.col.length),denom=new Float32Array(meta.neurons),inputs=new Uint32Array(meta.neurons).fill(0xffffffff);
  for(let r=0;r<meta.neurons;r++){let sum=0;for(let j=graph.crow[r];j<graph.crow[r+1];j++){if(graph.col[j]>=262144||graph.counts[j]>=16384)throw Error('Packing overflow');packed[j]=graph.col[j]|graph.counts[j]<<18;sum+=graph.counts[j];}denom[r]=Math.max(1,sum);}
  meta.input_ids.forEach((id,i)=>inputs[id]=i);
  const threads=options.threads||Math.min(8,Math.max(1,(navigator.hardwareConcurrency||4)-2));
  const rc=mod._setup(meta.neurons,put(graph.crow),put(packed),put(denom),put(inputs),put(new Uint32Array(meta.output_ids)),put(new Uint32Array(meta.display_ids)),put(new Float32Array(weights)),put(new Uint32Array(keys.map(k=>meta.arrays[k].offset))),threads);
  if(rc)throw Error('WASM thread initialization failed: '+rc);
  // Allow the runtime to dispatch pthread start messages before a barrier wait.
  await new Promise(resolve=>setTimeout(resolve,0));
  const self=new FlyWASM();self.mod=mod;self.obs=mod._malloc(8*27*4);self.threads=threads;self.kernel='wasm-simd-threads';self.adapterInfo={vendor:'CPU',description:`WASM SIMD · ${threads} threads`};self.device={features:[],destroy:()=>mod.PThread.terminateAllThreads()};return self;
 }
 reset(){this.mod._reset_model();}
 async step(observations){
  const t=performance.now();this.mod.HEAPF32.set(observations.flat(),this.obs/4);const ptr=this.mod._evaluate(this.obs)/4,data=this.mod.HEAPF32.slice(ptr,ptr+6200);
  if(!data.every(Number.isFinite))throw Error('Non-finite WASM inference');this.timing={wall:performance.now()-t,parts:Array.from(this.mod.HEAPF32.slice(ptr+6200,ptr+6203))};
  const twist=[],precision=[];for(let a=0;a<8;a++){const v=Array.from(data.slice(a*6,a*6+3),x=>Math.fround(x/.08)),ratio=Math.max(1,Math.hypot(v[0],v[1])/.18,Math.abs(v[2])/.4);twist.push(v.map((x,i)=>Math.fround(x/ratio/(i===2?1.4:.65))));precision.push(Array.from(data.slice(a*6+3,a*6+6),x=>Math.fround(Math.exp(Math.max(-3,Math.min(3,x))))));}
  return{twist,precision,brain:Array.from(data.slice(48,2096)),brain_rms:Array.from(data.slice(2096,2104)),features:Array.from({length:8},(_,i)=>Array.from(data.slice(2104+i*512,2104+(i+1)*512)))};
 }
}
