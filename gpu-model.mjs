import {loadGraph,sha256} from './graph-loader.mjs';

export class FlyGPU {
 static async create(meta,onProgress=()=>{}){
  if(!navigator.gpu)throw Error('This browser has no WebGPU. Try current desktop Chrome or Edge.');
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  if(!adapter)throw Error('No compatible WebGPU adapter. Enable browser hardware acceleration.');
  const required=25563197*4;
  if(adapter.limits.maxStorageBufferBindingSize<required||adapter.limits.maxBufferSize<required)throw Error('GPU storage limit is too small for the full fly network.');
  const device=await adapter.requestDevice({requiredFeatures:adapter.features.has('timestamp-query')?['timestamp-query']:[],requiredLimits:{maxStorageBufferBindingSize:required,maxBufferSize:required}});
  const graph=await loadGraph(new URL('./graph/manifest.json',import.meta.url).href,{onProgress});
  const response=await fetch(new URL('./model/weights.bin',import.meta.url)),weights=await response.arrayBuffer();
  if(await sha256(weights)!==meta.weights_sha256)throw Error('Model weight checksum mismatch');
  const self=new FlyGPU(device,meta,graph,new Float32Array(weights));
  self.adapterInfo=Object.fromEntries(['vendor','architecture','device','description'].map(k=>[k,adapter.info[k]]));
  await self.init();return self;
 }
 constructor(device,meta,graph,weights){this.device=device;this.meta=meta;this.graph=graph;this.weights=weights;this.ping=0;this.dead=false;device.lost.then(info=>{this.dead=true;this.lostMessage=info.message;});device.addEventListener('uncapturederror',e=>{this.dead=true;this.lostMessage=e.error.message;});}
 buffer(dataOrSize,label){const data=typeof dataOrSize==='number'?null:dataOrSize,size=data?data.byteLength:dataOrSize,b=this.device.createBuffer({label,size:Math.max(4,Math.ceil(size/4)*4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});if(data)this.device.queue.writeBuffer(b,0,data);return b;}
 async pipeline(code,buffers,label){const module=this.device.createShaderModule({code,label}),info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw Error(label+': '+errors.map(m=>m.message).join('; '));const p=await this.device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'},label});return{p,group:this.device.createBindGroup({layout:p.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))})};}
 async init(){
  const m=this.meta,N=m.neurons;this.N=N;this.w=this.buffer(this.weights,'readout weights');this.obs=this.buffer(8*27*4,'local observations');this.drive=this.buffer(2048*8*4,'sensory drive');this.state=[this.buffer(N*8*4,'neural A'),this.buffer(N*8*4,'neural B')];
  const inputMap=new Uint32Array(N).fill(0xffffffff);m.input_ids.forEach((id,i)=>inputMap[id]=i);
  this.inputMap=this.buffer(inputMap,'sensory neuron IDs');this.outputIds=this.buffer(new Uint32Array(m.output_ids),'readout neurons');this.displayIds=this.buffer(new Uint32Array(m.display_ids),'display neurons');
  const crow=this.buffer(this.graph.crow,'CSR row pointers'),col=this.buffer(this.graph.col,'CSR columns'),value=this.buffer(this.graph.weights,'CSR weights');this.graph=null;
  this.z=this.buffer(8*512*4,'normalized readout');this.l1=this.buffer(8*256*4,'hidden 1');this.l2=this.buffer(8*256*4,'hidden 2');this.ff=this.buffer(8*6*4,'feedforward');this.emb=this.buffer(8*128*4,'GRU embedding');this.hidden=[this.buffer(8*128*4,'GRU A'),this.buffer(8*128*4,'GRU B')];this.residual=this.buffer(8*6*4,'GRU residual');
  this.output=this.buffer((8*6+2048+8+8*512)*4,'readback features, neural activity, proposals');this.readback=this.device.createBuffer({size:(8*6+2048+8+8*512)*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  if(this.device.features.has('timestamp-query')){this.query=this.device.createQuerySet({type:'timestamp',count:32});this.queryBuffer=this.device.createBuffer({size:256,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});this.queryRead=this.device.createBuffer({size:256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});}
  const offset=k=>m.arrays[k].offset+'u';
  const declarations='@group(0) @binding(0) var<storage,read> w:array<f32>;';
  this.drivePipe=await this.pipeline(`${declarations}
@group(0) @binding(1) var<storage,read> obs:array<f32>;
@group(0) @binding(2) var<storage,read_write> drive:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let k=g.x;if(k>=16384u){return;}let neuron=k/8u;let agent=k%8u;var sum=w[${offset('core.bias')}+neuron];for(var j=0u;j<27u;j++){let z=clamp((obs[agent*27u+j]-w[${offset('core.obs_mean')}+j])/w[${offset('core.obs_std')}+j],-8.0,8.0);sum+=z*w[${offset('core.projection')}+neuron*27u+j];}drive[k]=sum;}`,[this.w,this.obs,this.drive],'sensory injection');
  // Eight adjacent lanes are eight flies: source state loads are coalesced.
  // Every original row and edge is evaluated; there is no graph pruning.
  const spmm=`@group(0) @binding(0) var<storage,read> row:array<u32>;
@group(0) @binding(1) var<storage,read> col:array<u32>;
@group(0) @binding(2) var<storage,read> value:array<f32>;
@group(0) @binding(3) var<storage,read> h:array<f32>;
@group(0) @binding(4) var<storage,read_write> next:array<f32>;
@group(0) @binding(5) var<storage,read> drive:array<f32>;
@group(0) @binding(6) var<storage,read> inputMap:array<u32>;
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) g:vec3<u32>){
let r=g.x/8u;let fly=g.x%8u;if(r>=${N}u){return;}var signal=0.0;
for(var j=row[r];j<row[r+1u];j++){signal+=value[j]*h[col[j]*8u+fly];}
let input=inputMap[r];if(input!=0xffffffffu){signal+=drive[input*8u+fly];}
next[r*8u+fly]=0.5*h[r*8u+fly]+0.5*tanh(signal);}`;
  this.spmm=[];for(let k=0;k<2;k++)this.spmm.push(await this.pipeline(spmm,[crow,col,value,this.state[k],this.state[1-k],this.drive,this.inputMap],'full CSR recurrence'));
  this.normalize=await this.pipeline(`${declarations}
@group(0) @binding(1) var<storage,read> h:array<f32>;@group(0) @binding(2) var<storage,read> ids:array<u32>;@group(0) @binding(3) var<storage,read_write> z:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let k=g.x;if(k>=4096u){return;}let j=k%512u;let a=k/512u;z[k]=clamp((h[ids[j]*8u+a]-w[${offset('input_mean')}+j])/w[${offset('input_std')}+j],-10.0,10.0);}`,[this.w,this.state[0],this.outputIds,this.z],'normalize features');
  const dense=async(name,input,output,ins,outs,activation=false)=>this.pipeline(`${declarations}
@group(0) @binding(1) var<storage,read> x:array<f32>;@group(0) @binding(2) var<storage,read_write> y:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let k=g.x;if(k>=${8*outs}u){return;}let a=k/${outs}u;let row=k%${outs}u;var sum=w[${offset(name+'.bias')}+row];for(var j=0u;j<${ins}u;j++){sum+=w[${offset(name+'.weight')}+row*${ins}u+j]*x[a*${ins}u+j];}y[k]=${activation?'sum/(1.0+exp(-sum))':'sum'};}`,[this.w,input,output],name);
  this.densePipes=[await dense('net.0',this.z,this.l1,512,256,true),await dense('net.2',this.l1,this.l2,256,256,true),await dense('net.4',this.l2,this.ff,256,6),await dense('embed.0',this.z,this.emb,512,128,true)];
  this.gru=[];this.res=[];
  for(let k=0;k<2;k++){
   this.gru.push(await this.pipeline(`${declarations}
@group(0) @binding(1) var<storage,read> x:array<f32>;@group(0) @binding(2) var<storage,read> h:array<f32>;@group(0) @binding(3) var<storage,read_write> out:array<f32>;
fn sigmoid(x:f32)->f32{return 1.0/(1.0+exp(-x));}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let k=g.x;if(k>=1024u){return;}let a=k/128u;let unit=k%128u;var iv:array<f32,3>;var hv:array<f32,3>;for(var gate=0u;gate<3u;gate++){let r=unit+gate*128u;var si=w[${offset('gru.bias_ih_l0')}+r];var sh=w[${offset('gru.bias_hh_l0')}+r];for(var j=0u;j<128u;j++){si+=w[${offset('gru.weight_ih_l0')}+r*128u+j]*x[a*128u+j];sh+=w[${offset('gru.weight_hh_l0')}+r*128u+j]*h[a*128u+j];}iv[gate]=si;hv[gate]=sh;}let reset=sigmoid(iv[0]+hv[0]);let update=sigmoid(iv[1]+hv[1]);let candidate=tanh(iv[2]+reset*hv[2]);out[k]=(1.0-update)*candidate+update*h[k];}`,[this.w,this.emb,this.hidden[k],this.hidden[1-k]],'local GRU'));
   this.res.push(await dense('residual',this.hidden[1-k],this.residual,128,6));
  }
  this.rmsPartial=this.buffer(Math.ceil(N/256)*8*4,'parallel RMS partials');
  this.rmsPipe=await this.pipeline(`@group(0) @binding(0) var<storage,read> h:array<f32>;@group(0) @binding(1) var<storage,read_write> out:array<f32>;
var<workgroup> lo:array<vec4<f32>,256>;var<workgroup> hi:array<vec4<f32>,256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) tid:u32){let r=group.x*256u+tid;var a=vec4<f32>(0.0);var b=vec4<f32>(0.0);if(r<${N}u){let k=r*8u;a=vec4<f32>(h[k],h[k+1u],h[k+2u],h[k+3u]);b=vec4<f32>(h[k+4u],h[k+5u],h[k+6u],h[k+7u]);}lo[tid]=a*a;hi[tid]=b*b;workgroupBarrier();for(var stride=128u;stride>0u;stride/=2u){if(tid<stride){lo[tid]+=lo[tid+stride];hi[tid]+=hi[tid+stride];}workgroupBarrier();}if(tid<8u){out[group.x*8u+tid]=select(hi[0][tid%4u],lo[0][tid%4u],tid<4u);}}`,[this.state[0],this.rmsPartial],'parallel neural RMS');
  this.finalPipe=await this.pipeline(`${declarations}
@group(0) @binding(1) var<storage,read> ff:array<f32>;@group(0) @binding(2) var<storage,read> residual:array<f32>;@group(0) @binding(3) var<storage,read> h:array<f32>;@group(0) @binding(4) var<storage,read> display:array<u32>;@group(0) @binding(5) var<storage,read> ids:array<u32>;@group(0) @binding(6) var<storage,read_write> out:array<f32>;
@group(0) @binding(7) var<storage,read> partial:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let k=g.x;
if(k<48u){let j=k%6u;out[k]=(ff[k]+residual[k])*w[${offset('target_std')}+j]+w[${offset('target_mean')}+j];}
if(k<2048u){out[48u+k]=h[display[k]*8u];}
if(k<8u){var sum=0.0;for(var r=0u;r<${Math.ceil(N/256)}u;r++){sum+=partial[r*8u+k];}out[2096u+k]=sqrt(sum/${N}.0);}
if(k<4096u){out[2104u+k]=h[ids[k%512u]*8u+k/512u];}}`,[this.w,this.ff,this.residual,this.state[0],this.displayIds,this.outputIds,this.output,this.rmsPartial],'decode and neural readback');
  this.weights=null;this.reset();
 }
 reset(){for(const b of [...this.state,...this.hidden])this.device.queue.writeBuffer(b,0,new Uint8Array(b.size));this.ping=0;}
 async step(observations){
  if(this.dead)throw Error('GPU stopped: '+this.lostMessage);
  this.device.queue.writeBuffer(this.obs,0,new Float32Array(observations.flat()));const enc=this.device.createCommandEncoder();
  const start=performance.now();let queryIndex=0;const dispatch=(pipe,n)=>{const pass=enc.beginComputePass(this.query?{timestampWrites:{querySet:this.query,beginningOfPassWriteIndex:queryIndex++,endOfPassWriteIndex:queryIndex++}}:{});pass.setPipeline(pipe.p);pass.setBindGroup(0,pipe.group);pass.dispatchWorkgroups(n);pass.end();};
  dispatch(this.drivePipe,256);for(let i=0;i<this.meta.neural_steps;i++)dispatch(this.spmm[i%2],Math.ceil(this.N/16));
  dispatch(this.normalize,64);[32,32,1,16].forEach((n,i)=>dispatch(this.densePipes[i],n));dispatch(this.gru[this.ping],16);dispatch(this.res[this.ping],1);this.ping=1-this.ping;
  dispatch(this.rmsPipe,Math.ceil(this.N/256));dispatch(this.finalPipe,64);enc.copyBufferToBuffer(this.output,0,this.readback,0,this.readback.size);if(this.query){enc.resolveQuerySet(this.query,0,queryIndex,this.queryBuffer,0);enc.copyBufferToBuffer(this.queryBuffer,0,this.queryRead,0,queryIndex*8);}this.device.queue.submit([enc.finish()]);
  await this.readback.mapAsync(GPUMapMode.READ);const data=new Float32Array(this.readback.getMappedRange()).slice();this.readback.unmap();
  this.timing={wall:performance.now()-start};if(this.query){await this.queryRead.mapAsync(GPUMapMode.READ);const stamps=new BigUint64Array(this.queryRead.getMappedRange());this.timing.passes=Array.from({length:queryIndex/2},(_,i)=>Number(stamps[2*i+1]-stamps[2*i])/1e6);this.queryRead.unmap();}
  if(!data.every(Number.isFinite))throw Error('Non-finite GPU inference');
  const twist=[],precision=[];
  for(let a=0;a<8;a++){const v=Array.from(data.slice(a*6,a*6+3),x=>Math.fround(x/.08)),ratio=Math.max(1,Math.hypot(v[0],v[1])/.18,Math.abs(v[2])/.4);twist.push(v.map((x,i)=>Math.fround(x/ratio/(i===2?1.4:.65))));precision.push(Array.from(data.slice(a*6+3,a*6+6),x=>Math.fround(Math.exp(Math.max(-3,Math.min(3,x))))));}
  return{twist,precision,brain:Array.from(data.slice(48,2096)),brain_rms:Array.from(data.slice(2096,2104)),features:Array.from({length:8},(_,i)=>Array.from(data.slice(2104+i*512,2104+(i+1)*512)))};
 }
}
