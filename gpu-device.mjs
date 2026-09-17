// Standard WebGPU requests only. A website cannot enumerate arbitrary PCI
// devices or override the browser/OS GPU assignment.
export function adapterInfo(adapter){
 const info=adapter.info||{};
 return {...Object.fromEntries(['vendor','architecture','device','description'].map(k=>[k,info[k]||''])),fallback:!!(info.isFallbackAdapter??adapter.isFallbackAdapter)};
}
export async function selectAdapter(gpu){
 if(!gpu)throw Error('WebGPU unavailable. Use a browser with hardware WebGPU to race.');
 const candidates=[];
 for(const preference of ['high-performance',undefined]){
  const adapter=await gpu.requestAdapter({...preference?{powerPreference:preference}:{},forceFallbackAdapter:false});
  if(!adapter)continue;
  const info=adapterInfo(adapter),key=JSON.stringify(info);
  if(!candidates.some(c=>c.key===key))candidates.push({adapter,info,key,request:preference||'browser-default'});
 }
 const required=25563197*4;
 const usable=candidates.filter(c=>!c.info.fallback&&c.adapter.limits.maxStorageBufferBindingSize>=required&&c.adapter.limits.maxBufferSize>=required);
 // Prefer an exposed NVIDIA device; otherwise preserve high-performance
 // request order. We never label an integrated adapter as an NVIDIA device.
 const rank=c=>/nvidia/i.test(c.info.vendor)?2:1;
 usable.sort((a,b)=>rank(b)-rank(a));
 if(!usable.length)throw Error('No hardware WebGPU adapter with sufficient memory for the neural race.');
 return {adapter:usable[0].adapter,info:usable[0].info,candidates:candidates.map(({info,request})=>({...info,request}))};
}
