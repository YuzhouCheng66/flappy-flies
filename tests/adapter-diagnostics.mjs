// Read-only, standard-API diagnostic. No browser flags or OS settings changed.
export async function inspectAdapters(context){
 const result={type:'adapter-diagnostic',context,time:new Date().toISOString(),userAgent:navigator.userAgent,brands:navigator.userAgentData?.brands,secure:isSecureContext,isolated:crossOriginIsolated,hasWebGPU:!!navigator.gpu,languageFeatures:[...(navigator.gpu?.wgslLanguageFeatures||[])],requests:[]};
 if(!navigator.gpu)return result;
 const info=a=>Object.fromEntries(['vendor','architecture','device','description','subgroupMinSize','subgroupMaxSize','isFallbackAdapter'].map(k=>[k,a?.[k]??null]));
 // Keep every response, even if privacy-filtered labels are identical.
 for(const options of [{featureLevel:'core',powerPreference:'high-performance'},{featureLevel:'core'},{featureLevel:'core',powerPreference:'low-power'},{featureLevel:'compatibility',powerPreference:'high-performance'}]){
  let device;const row={options};result.requests.push(row);
  try{
   const adapter=await navigator.gpu.requestAdapter({...options,forceFallbackAdapter:false});if(!adapter){row.adapter=null;continue;}
   row.adapter=info(adapter.info);row.features=[...adapter.features];row.limits=Object.fromEntries(['maxBufferSize','maxStorageBufferBindingSize','maxComputeWorkgroupStorageSize','maxComputeInvocationsPerWorkgroup'].map(k=>[k,adapter.limits[k]]));
   row.fullModelMemorySuitable=adapter.limits.maxBufferSize>=102252788&&adapter.limits.maxStorageBufferBindingSize>=102252788;
   device=await adapter.requestDevice();row.deviceAdapter=info(device.adapterInfo);
   const buffer=device.createBuffer({size:256,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const shader=device.createShaderModule({code:'@group(0) @binding(0) var<storage,read_write> out:array<u32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) i:vec3<u32>){out[i.x]=i.x*3u+7u;}'});
   const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:shader,entryPoint:'main'}}),bindings=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer}}]});
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bindings);pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(buffer,0,read,0,256);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);row.computePassed=Array.from(new Uint32Array(read.getMappedRange())).every((v,i)=>v===i*3+7);read.unmap();buffer.destroy();read.destroy();
  }catch(error){row.error=error.message;}finally{device?.destroy();}
 }
 return result;
}
