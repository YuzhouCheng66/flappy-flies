// Recorded full-model states. Quantization applies only to displayed activity.
// Poses, forces, communication strength and RMS retain float32 precision.
export const REPLAY_SCHEMA='flappy-full-brain-replay-v1';
export function packetStrengths(f){
 if(!f.natural)return [];
 const energy=f.natural.map((sweep,s)=>sweep.map((vec,j)=>Math.sqrt(vec.reduce((v,h,k)=>v+h*h/Math.max(f.precision[s][j][k],1e-8),0)/3)));
 const scale=Math.max(1e-8,...energy.flat());return energy.flat().map(x=>Math.sqrt(x/scale));
}
export function encodeReplay(frames,header){
 const B=header.brainCount,E=header.sender.length,S=header.sweeps,recordBytes=24+4+4+64+32+E*S*4+B*2;
 const metadata=new TextEncoder().encode(JSON.stringify({...header,schema:REPLAY_SCHEMA,frameCount:frames.length,recordBytes,brainEncoding:'signed-int16 / 32767; visualization only'}));
 const bytes=new Uint8Array(4+metadata.length+frames.length*recordBytes),v=new DataView(bytes.buffer);let o=0;v.setUint32(o,metadata.length,true);o+=4;bytes.set(metadata,o);o+=metadata.length;
 for(const f of frames){
  const put=x=>{if(!Number.isFinite(x))throw Error('Non-finite replay value');v.setFloat32(o,x,true);o+=4;};
  f.state.forEach(put);v.setUint32(o,(f.stage&255)|(+!!f.hit<<8)|(+!!f.done<<9)|(+!!f.success<<10),true);o+=4;v.setUint32(o,f.wire_bytes||0,true);o+=4;
  (f.forces?.flat()||Array(16).fill(0)).forEach(put);(f.brain_rms||Array(8).fill(0)).forEach(put);
  const strength=packetStrengths(f);for(let j=0;j<E*S;j++)put(strength[j]||0);
  for(let j=0;j<B;j++){const x=f.brain?.[j]||0;if(!Number.isFinite(x)||Math.abs(x)>1.00001)throw Error('Brain state outside leaky-tanh range');v.setInt16(o,Math.round(Math.max(-1,Math.min(1,x))*32767),true);o+=2;}
 }
 if(o!==bytes.length)throw Error('Replay record size mismatch');return bytes;
}
export function decodeReplay(bytes){
 const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);let o=0;const len=v.getUint32(o,true);o+=4;
 const h=JSON.parse(new TextDecoder().decode(bytes.subarray(o,o+len)));o+=len;
 if(h.schema!==REPLAY_SCHEMA||o+h.frameCount*h.recordBytes!==bytes.length)throw Error('Invalid replay envelope');
 const frames=[];
 for(let tick=0;tick<h.frameCount;tick++){
  const get=()=>{const x=v.getFloat32(o,true);o+=4;return x;};const state=Array.from({length:6},get),flags=v.getUint32(o,true);o+=4;const wire_bytes=v.getUint32(o,true);o+=4;
  const forces=Array.from({length:8},()=>Array.from({length:2},get)),brain_rms=Array.from({length:8},get),packets=[];
  for(let s=0;s<h.sweeps;s++)for(let j=0;j<h.sender.length;j++){const strength=get();if(tick>0)packets.push({at:tick-1+s/h.sweeps,u:h.sender[j],v:h.receiver[j],strength});}
  const brain=Array.from({length:h.brainCount},()=>{const x=v.getInt16(o,true)/32767;o+=2;return x;});
  frames.push({tick,state,stage:flags&255,hit:!!(flags&256),done:!!(flags&512),success:!!(flags&1024),wire_bytes,forces,brain_rms,brain,packets});
 }
 return{header:h,frames};
}
