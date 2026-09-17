import {loadGraph,sha256} from './graph-loader.mjs';
import {selectAdapter} from './gpu-device.mjs';
import {reorderBySourceFrequency} from './graph-layout.mjs';

export class FlyGPU {
 static async create(meta,onProgress=()=>{},options={}){
  if(!['f32','f16','packed16'].includes(options.precision||'f32'))throw Error('Unknown neural state precision');
  const selected=await selectAdapter(navigator.gpu),adapter=selected.adapter;
  onProgress({phase:'adapter',adapter:selected.info,candidates:selected.candidates});
  const required=25563197*4;
  if(adapter.limits.maxStorageBufferBindingSize<required||adapter.limits.maxBufferSize<required)throw Error('GPU storage limit is too small for the full fly network.');
  if(options.precision==='f16'&&!adapter.features.has('shader-f16'))throw Error('FP16 neural storage requires shader-f16; use the FP32 model on this adapter.');
  if(options.subgroupSize&&(!adapter.features.has('subgroup-size-control')||options.kernel!=='subgroup'||![8,16,32,64,128].includes(options.subgroupSize)||options.subgroupSize<adapter.info.subgroupMinSize||options.subgroupSize>adapter.info.subgroupMaxSize))throw Error('Requested subgroup size is not supported');
  const device=await adapter.requestDevice({requiredFeatures:[...(options.profile===false?[]:['timestamp-query']),...(options.precision==='f16'?['shader-f16']:[]),...(options.kernel==='subgroup'?['subgroups']:[]),...(options.subgroupSize?['subgroup-size-control']:[])].filter(f=>adapter.features.has(f)),requiredLimits:{maxStorageBufferBindingSize:required,maxBufferSize:required}});
  try{
  const graph=await loadGraph(new URL('./graph/manifest.json',import.meta.url).href,{onProgress});
  const response=await fetch(new URL(options.weightsURL||'./model/weights.bin',import.meta.url)),weights=await response.arrayBuffer();
  if(await sha256(weights)!==meta.weights_sha256)throw Error('Model weight checksum mismatch');
  const self=new FlyGPU(device,meta,graph,new Float32Array(weights));self.options=options;
  self.adapterInfo=selected.info;self.adapterCandidates=selected.candidates;
  await self.init();return self;
  }catch(error){device.destroy();throw error;}
 }
 constructor(device,meta,graph,weights){this.device=device;this.meta=meta;this.graph=graph;this.weights=weights;this.ping=0;this.dead=false;device.lost.then(info=>{this.dead=true;this.lostMessage=info.message;});device.addEventListener('uncapturederror',e=>{this.dead=true;this.lostMessage=e.error.message;});}
 buffer(dataOrSize,label){const data=typeof dataOrSize==='number'?null:dataOrSize,size=data?data.byteLength:dataOrSize,b=this.device.createBuffer({label,size:Math.max(4,Math.ceil(size/4)*4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});if(data)this.device.queue.writeBuffer(b,0,data);return b;}
 async pipeline(code,buffers,label){const module=this.device.createShaderModule({code,label}),info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw Error(label+': '+errors.map(m=>m.message).join('; '));const p=await this.device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'},label});return{p,group:this.device.createBindGroup({layout:p.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))})};}
 async init(){
  const m=this.meta,N=m.neurons;this.N=N;this.precision=this.options?.precision||'f32';
  if(m.neural_steps%2!==0)throw Error('Neural readout requires an even number of recurrence steps');
  const half=this.precision==='f16',packedHalf=this.precision==='packed16',stateBytes=N*8*(this.precision==='f32'?4:2);
  // Only persistent neural states are rounded. Graph counts/indices, sensory
  // drive, products, reductions, tanh, GRU and readout remain FP32.
  const neuralHeader=(vector=false)=>packedHalf
   ?`alias Neural=u32;\nalias Neural4=vec2<u32>;\nfn loadH(i:u32)->${vector?'vec4<f32>{let v=h[i];return vec4<f32>(unpack2x16float(v.x),unpack2x16float(v.y));':'f32{return unpack2x16float(h[i/2u])[i%2u];'}}\nfn storeH(v:vec4<f32>)->Neural4{return vec2<u32>(pack2x16float(v.xy),pack2x16float(v.zw));}\n`
   :`${half?'enable f16;\n':''}alias Neural=${half?'f16':'f32'};\nalias Neural4=vec4<Neural>;\nfn loadH(i:u32)->${vector?'vec4<f32>':'f32'}{return ${vector?'vec4<f32>':'f32'}(h[i]);}\nfn storeH(v:vec4<f32>)->Neural4{return Neural4(v);}\n`;
  this.w=this.buffer(this.weights,'readout weights');this.obs=this.buffer(8*27*4,'local observations');this.drive=this.buffer(2048*8*4,'sensory drive');this.state=[this.buffer(stateBytes,'neural A'),this.buffer(stateBytes,'neural B')];
  this.memoryLayout=this.options?.memoryLayout||'original';
  if(!['original','source-frequency'].includes(this.memoryLayout))throw Error('Unknown neural memory layout');
  let remap;
  if(this.memoryLayout==='source-frequency'){const reordered=reorderBySourceFrequency(this.graph);this.graph=reordered.graph;remap=reordered.originalToStorage;}
  const storageId=id=>remap?remap[id]:id;
  const inputMap=new Uint32Array(N).fill(0xffffffff);m.input_ids.forEach((id,i)=>inputMap[storageId(id)]=i);
  this.inputMap=this.buffer(inputMap,'sensory neuron IDs');this.outputIds=this.buffer(new Uint32Array(m.output_ids.map(storageId)),'readout neurons');this.displayIds=this.buffer(new Uint32Array(m.display_ids.map(storageId)),'display neurons');
  const graph=this.graph,order=Uint32Array.from({length:N},(_,i)=>i);
  order.sort((a,b)=>(graph.crow[a+1]-graph.crow[a])-(graph.crow[b+1]-graph.crow[b]));
  const rowOrder=this.buffer(order,'degree-sorted neuron worklist');
  const packed=new Uint32Array(graph.col.length),denominator=new Float32Array(N);
  for(let r=0;r<N;r++){let sum=0;for(let j=graph.crow[r];j<graph.crow[r+1];j++){
   if(graph.col[j]>=262144||graph.counts[j]>=16384)throw Error('Edge exceeds lossless packing range');
   packed[j]=graph.col[j]|(graph.counts[j]<<18);sum+=graph.counts[j];
  }denominator[r]=Math.max(1,sum);}
  this.kernel=this.options?.kernel||'tile16';
  if(packedHalf&&!this.kernel.startsWith('tile')&&!this.kernel.startsWith('pair')&&!['row8','sell','subgroup','segmented'].includes(this.kernel))throw Error('Packed FP16 requires a vector neural kernel');
  let pointers=graph.crow,edgeData=this.kernel==='legacy'?graph.col:packed,worklist=order;
  if(this.kernel==='sell'){
   const width=32,blocks=Math.ceil(N/width);let length=0;pointers=new Uint32Array(N);worklist=new Uint32Array(N*2);
   for(let b=0;b<blocks;b++){const end=Math.min(N,(b+1)*width),last=order[end-1],degree=graph.crow[last+1]-graph.crow[last];for(let i=b*width;i<end;i++){pointers[i]=length+i%width;worklist[i*2]=order[i];worklist[i*2+1]=graph.crow[order[i]+1]-graph.crow[order[i]];}length+=degree*width;}
   edgeData=new Uint32Array(length);for(let i=0;i<N;i++){const r=order[i];for(let j=graph.crow[r];j<graph.crow[r+1];j++)edgeData[pointers[i]+(j-graph.crow[r])*width]=packed[j];}
  }
  const crow=this.buffer(pointers,'sparse row pointers'),col=this.buffer(edgeData,'lossless sparse edges'),value=this.buffer(this.kernel==='legacy'?graph.weights:denominator,'CSR normalization'),sortedRows=this.kernel==='sell'?this.buffer(worklist,'SELL row identities and degrees'):rowOrder;this.graph=null;
  this.z=this.buffer(8*512*4,'normalized readout');this.l1=this.buffer(8*256*4,'hidden 1');this.l2=this.buffer(8*256*4,'hidden 2');this.ff=this.buffer(8*6*4,'feedforward');this.emb=this.buffer(8*128*4,'GRU embedding');this.hidden=[this.buffer(8*128*4,'GRU A'),this.buffer(8*128*4,'GRU B')];this.residual=this.buffer(8*6*4,'GRU residual');
  this.readbackFloats=6200+(this.options.captureAllBrains?7*2048:0);
  this.output=this.buffer(this.readbackFloats*4,'readback features, neural activity, proposals');this.readback=this.device.createBuffer({size:this.readbackFloats*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  if(this.device.features.has('timestamp-query')){this.query=this.device.createQuerySet({type:'timestamp',count:64});this.queryBuffer=this.device.createBuffer({size:512,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});this.queryRead=this.device.createBuffer({size:512,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});}
  const offset=k=>m.arrays[k].offset+'u';
  const declarations='@group(0) @binding(0) var<storage,read> w:array<f32>;';
  this.drivePipe=await this.pipeline(`${declarations}
@group(0) @binding(1) var<storage,read> obs:array<f32>;
@group(0) @binding(2) var<storage,read_write> drive:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let k=g.x;if(k>=16384u){return;}let neuron=k/8u;let agent=k%8u;var sum=w[${offset('core.bias')}+neuron];for(var j=0u;j<27u;j++){let z=clamp((obs[agent*27u+j]-w[${offset('core.obs_mean')}+j])/w[${offset('core.obs_std')}+j],-8.0,8.0);sum+=z*w[${offset('core.projection')}+neuron*27u+j];}drive[k]=sum;}`,[this.w,this.obs,this.drive],'sensory injection');
  // Eight adjacent lanes are eight flies: source state loads are coalesced.
  // Every original row and edge is evaluated; there is no graph pruning.
  const bindings=`@group(0) @binding(0) var<storage,read> row:array<u32>;
@group(0) @binding(1) var<storage,read> col:array<u32>;
@group(0) @binding(2) var<storage,read> value:array<f32>;
@group(0) @binding(3) var<storage,read> h:array<Neural>;
@group(0) @binding(4) var<storage,read_write> next:array<Neural>;
@group(0) @binding(5) var<storage,read> drive:array<f32>;
@group(0) @binding(6) var<storage,read> inputMap:array<u32>;
@group(0) @binding(7) var<storage,read> order:array<u32>;`;
  let spmm=bindings+`@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) g:vec3<u32>){
let index=g.x/8u;let fly=g.x%8u;if(index>=${N}u){return;}let r=order[index];var signal=0.0;
for(var j=row[r];j<row[r+1u];j++){${this.kernel==='legacy'?'signal+=value[j]*loadH(col[j]*8u+fly);':'let edge=col[j];signal+=(0.5*(f32(edge>>18u)/value[r]))*loadH((edge&262143u)*8u+fly);'}}
let input=inputMap[r];if(input!=0xffffffffu){signal+=drive[input*8u+fly];}
next[r*8u+fly]=Neural(0.5*loadH(r*8u+fly)+0.5*tanh(signal));}`;
  this.spmmGroups=Math.ceil(N/16);
  if(this.kernel.startsWith('tile')){
   const lanes=Number(this.kernel.slice(4))||4,rows=64/(lanes*2);
   spmm=bindings.replace('h:array<Neural>','h:array<Neural4>').replace('next:array<Neural>','next:array<Neural4>')+`
var<workgroup> partial:array<vec4<f32>,64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) tid:u32){
let index=(group.x+group.y*65535u)*${rows}u+tid/${lanes*2}u;let fly=tid%2u;let part=(tid/2u)%${lanes}u;let r=order[min(index,${N-1}u)];var signal=vec4<f32>(0.0);
if(index<${N}u){for(var j=row[r]+part;j<row[r+1u];j+=${lanes}u){let edge=col[j];signal+=f32(edge>>18u)*loadH((edge&262143u)*2u+fly);}}
partial[tid]=signal;workgroupBarrier();
if(index<${N}u&&part==0u){for(var i=1u;i<${lanes}u;i++){signal+=partial[tid+i*2u];}signal=0.5*(signal/value[r]);let input=inputMap[r];if(input!=0xffffffffu){let k=input*8u+fly*4u;signal+=vec4<f32>(drive[k],drive[k+1u],drive[k+2u],drive[k+3u]);}next[r*2u+fly]=storeH(0.5*loadH(r*2u+fly)+0.5*tanh(signal));}}`;
   this.spmmGroups=Math.ceil(N/rows);
  }
  if(this.kernel==='sell'){
   spmm=bindings.replace('h:array<Neural>','h:array<Neural4>').replace('next:array<Neural>','next:array<Neural4>')+`
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) g:vec3<u32>){
let index=g.x/2u;let fly=g.x%2u;if(index>=${N}u){return;}let r=order[index*2u];let degree=order[index*2u+1u];var signal=vec4<f32>(0.0);
for(var j=0u;j<degree;j++){let edge=col[row[index]+j*32u];signal+=f32(edge>>18u)*loadH((edge&262143u)*2u+fly);}
signal=0.5*(signal/value[r]);let input=inputMap[r];if(input!=0xffffffffu){let k=input*8u+fly*4u;signal+=vec4<f32>(drive[k],drive[k+1u],drive[k+2u],drive[k+3u]);}
next[r*2u+fly]=storeH(0.5*loadH(r*2u+fly)+0.5*tanh(signal));}`;
   this.spmmGroups=Math.ceil(N/64);
  }
  if(this.kernel==='row8'||this.kernel.startsWith('pair')){
   // Each lane loads an edge once and advances all eight independent agents.
   // This changes scheduling, not the graph, states or recurrence count.
   const lanes=this.kernel==='row8'?1:Number(this.kernel.slice(4)),rows=64/lanes;
   if(![1,2,4,8,16,32].includes(lanes))throw Error('Invalid paired row width');
   spmm=bindings.replace('h:array<Neural>','h:array<Neural4>').replace('next:array<Neural>','next:array<Neural4>')+`
var<workgroup> low:array<vec4<f32>,64>;var<workgroup> high:array<vec4<f32>,64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) tid:u32){
let index=(group.x+group.y*65535u)*${rows}u+tid/${lanes}u;let part=tid%${lanes}u;let r=order[min(index,${N-1}u)];var a=vec4<f32>(0.0);var b=vec4<f32>(0.0);
if(index<${N}u){for(var j=row[r]+part;j<row[r+1u];j+=${lanes}u){let edge=col[j];let source=(edge&262143u)*2u;let count=f32(edge>>18u);a+=count*loadH(source);b+=count*loadH(source+1u);}}
${lanes>1?'low[tid]=a;high[tid]=b;workgroupBarrier();':''}
if(index<${N}u&&part==0u){${lanes>1?`for(var i=1u;i<${lanes}u;i++){a+=low[tid+i];b+=high[tid+i];}`:''}
a=0.5*(a/value[r]);b=0.5*(b/value[r]);let input=inputMap[r];if(input!=0xffffffffu){let k=input*8u;a+=vec4<f32>(drive[k],drive[k+1u],drive[k+2u],drive[k+3u]);b+=vec4<f32>(drive[k+4u],drive[k+5u],drive[k+6u],drive[k+7u]);}
next[r*2u]=storeH(0.5*loadH(r*2u)+0.5*tanh(a));next[r*2u+1u]=storeH(0.5*loadH(r*2u+1u)+0.5*tanh(b));}}`;
   this.spmmGroups=Math.ceil(N/rows);
  }
  if(this.kernel==='subgroup'){
   if(!this.device.features.has('subgroups'))throw Error('Subgroups unavailable');
   const size=this.options.subgroupSize,groupRows=size?128/size:(this.options.subgroupRows||4);
   if(![1,2,4,8,16].includes(groupRows))throw Error('Invalid subgroup row count');
   spmm=bindings.replace('h:array<Neural>','h:array<Neural4>').replace('next:array<Neural>','next:array<Neural4>')+`
${size?`@subgroup_size(${size})`:''}
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) tid:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) size:u32){
let fly=lane%2u;let part=lane/2u;let groupIndex=group.x+group.y*65535u;
for(var round=0u;round<(${groupRows}u+128u/size-1u)/(128u/size);round++){
 let index=groupIndex*${groupRows}u+tid/size+round*(128u/size);let valid=index<min(${N}u,groupIndex*${groupRows}u+${groupRows}u);
 let r=order[min(index,${N-1}u)];var signal=vec4<f32>(0.0);
 if(valid){for(var j=row[r]+part;j<row[r+1u];j+=size/2u){let edge=col[j];signal+=f32(edge>>18u)*loadH((edge&262143u)*2u+fly);}}
 for(var offset=2u;offset<size;offset*=2u){signal+=subgroupShuffleXor(signal,offset);}
 if(valid&&lane<2u){signal=0.5*(signal/value[r]);let input=inputMap[r];if(input!=0xffffffffu){let k=input*8u+fly*4u;signal+=vec4<f32>(drive[k],drive[k+1u],drive[k+2u],drive[k+3u]);}next[r*2u+fly]=storeH(0.5*loadH(r*2u+fly)+0.5*tanh(signal));}
}}`;
   this.spmmGroups=Math.ceil(N/groupRows);
  }
  spmm=(this.kernel==='subgroup'?'enable subgroups;\n'+(this.options.subgroupSize?'enable subgroup_size_control;\n':''):'')+neuralHeader(this.kernel.startsWith('tile')||this.kernel.startsWith('pair')||['row8','sell','subgroup'].includes(this.kernel))+spmm;
  this.spmm=[];if(this.kernel!=='segmented')for(let k=0;k<2;k++)this.spmm.push(await this.pipeline(spmm,[crow,col,value,this.state[k],this.state[1-k],this.drive,this.inputMap,sortedRows],'full CSR recurrence'));
  if(this.kernel==='segmented'){
   // Bound work per group even for high-degree rows; sum every original edge.
   const segments=[],starts=new Uint32Array(N+1),chunk=256;
   for(let r=0;r<N;r++){starts[r]=segments.length/4;for(let j=graph.crow[r];j<graph.crow[r+1];j+=chunk)segments.push(r,j,Math.min(j+chunk,graph.crow[r+1]),0);}starts[N]=segments.length/4;
   const descriptors=this.buffer(new Uint32Array(segments),'bounded CSR segments'),ptr=this.buffer(starts,'row segment pointers'),partial=this.buffer(starts[N]*8*4,'segment sums');
   const segmentCode=neuralHeader(true)+`@group(0) @binding(0) var<storage,read> segments:array<vec4<u32>>;
@group(0) @binding(1) var<storage,read> edges:array<u32>;
@group(0) @binding(2) var<storage,read> h:array<Neural4>;
@group(0) @binding(3) var<storage,read_write> out:array<vec4<f32>>;
var<workgroup> sums:array<vec4<f32>,64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) t:u32){
let s=g.x+g.y*65535u;let valid=s<${starts[N]}u;let fly=t%2u;let lane=t/2u;var v=vec4<f32>(0.0);
if(valid){let d=segments[s];for(var j=d.y+lane;j<d.z;j+=32u){let edge=edges[j];v+=f32(edge>>18u)*loadH((edge&262143u)*2u+fly);}}
sums[t]=v;workgroupBarrier();for(var stride=16u;stride>0u;stride/=2u){if(lane<stride){sums[t]+=sums[t+stride*2u];}workgroupBarrier();}
if(valid&&lane==0u){out[s*2u+fly]=sums[t];}}`;
   const finishCode=neuralHeader(true)+`@group(0) @binding(0) var<storage,read> ptr:array<u32>;
@group(0) @binding(1) var<storage,read> partial:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> denom:array<f32>;
@group(0) @binding(3) var<storage,read> h:array<Neural4>;
@group(0) @binding(4) var<storage,read_write> next:array<Neural4>;
@group(0) @binding(5) var<storage,read> drive:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read> ids:array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let r=g.x/2u;let fly=g.x%2u;if(r>=${N}u){return;}var v=vec4<f32>(0.0);for(var j=ptr[r];j<ptr[r+1u];j++){v+=partial[j*2u+fly];}v=0.5*v/denom[r];let id=ids[r];if(id!=0xffffffffu){v+=drive[id*2u+fly];}next[r*2u+fly]=storeH(0.5*loadH(r*2u+fly)+0.5*tanh(v));}`;
   this.spmm=[];this.segmentFinalize=[];this.spmmGroups=starts[N];
   for(let k=0;k<2;k++){this.spmm.push(await this.pipeline(segmentCode,[descriptors,col,this.state[k],partial],'segmented full CSR'));this.segmentFinalize.push(await this.pipeline(finishCode,[ptr,partial,value,this.state[k],this.state[1-k],this.drive,this.inputMap],'segment reduction and recurrence'));}
  }
  this.normalize=await this.pipeline(`${neuralHeader()}${declarations}
@group(0) @binding(1) var<storage,read> h:array<Neural>;@group(0) @binding(2) var<storage,read> ids:array<u32>;@group(0) @binding(3) var<storage,read_write> z:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let k=g.x;if(k>=4096u){return;}let j=k%512u;let a=k/512u;z[k]=clamp((loadH(ids[j]*8u+a)-w[${offset('input_mean')}+j])/w[${offset('input_std')}+j],-10.0,10.0);}`,[this.w,this.state[0],this.outputIds,this.z],'normalize features');
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
  this.rmsPipe=await this.pipeline(`${neuralHeader()}@group(0) @binding(0) var<storage,read> h:array<Neural>;@group(0) @binding(1) var<storage,read_write> out:array<f32>;
var<workgroup> lo:array<vec4<f32>,256>;var<workgroup> hi:array<vec4<f32>,256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) tid:u32){let r=group.x*256u+tid;var a=vec4<f32>(0.0);var b=vec4<f32>(0.0);if(r<${N}u){let k=r*8u;a=vec4<f32>(loadH(k),loadH(k+1u),loadH(k+2u),loadH(k+3u));b=vec4<f32>(loadH(k+4u),loadH(k+5u),loadH(k+6u),loadH(k+7u));}lo[tid]=a*a;hi[tid]=b*b;workgroupBarrier();for(var stride=128u;stride>0u;stride/=2u){if(tid<stride){lo[tid]+=lo[tid+stride];hi[tid]+=hi[tid+stride];}workgroupBarrier();}if(tid<8u){out[group.x*8u+tid]=select(hi[0][tid%4u],lo[0][tid%4u],tid<4u);}}`,[this.state[0],this.rmsPartial],'parallel neural RMS');
  this.finalPipe=await this.pipeline(`${neuralHeader()}${declarations}
@group(0) @binding(1) var<storage,read> ff:array<f32>;@group(0) @binding(2) var<storage,read> residual:array<f32>;@group(0) @binding(3) var<storage,read> h:array<Neural>;@group(0) @binding(4) var<storage,read> display:array<u32>;@group(0) @binding(5) var<storage,read> ids:array<u32>;@group(0) @binding(6) var<storage,read_write> out:array<f32>;
@group(0) @binding(7) var<storage,read> partial:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let k=g.x;
if(k<48u){let j=k%6u;out[k]=(ff[k]+residual[k])*w[${offset('target_std')}+j]+w[${offset('target_mean')}+j];}
if(k<2048u){out[48u+k]=loadH(display[k]*8u);}
if(k<8u){var sum=0.0;for(var r=0u;r<${Math.ceil(N/256)}u;r++){sum+=partial[r*8u+k];}out[2096u+k]=sqrt(sum/${N}.0);}
if(k<4096u){out[2104u+k]=loadH(ids[k%512u]*8u+k/512u);}
${this.options.captureAllBrains?'if(k<14336u){out[6200u+k]=loadH(display[k%2048u]*8u+1u+k/2048u);}':''}}`,[this.w,this.ff,this.residual,this.state[0],this.displayIds,this.outputIds,this.output,this.rmsPartial],'decode and neural readback');
  this.weights=null;this.reset();
 }
 reset(){for(const b of [...this.state,...this.hidden])this.device.queue.writeBuffer(b,0,new Uint8Array(b.size));this.ping=0;}
 async step(observations){
  if(this.dead)throw Error('GPU stopped: '+this.lostMessage);
  this.device.queue.writeBuffer(this.obs,0,new Float32Array(observations.flat()));const enc=this.device.createCommandEncoder();
  const start=performance.now();let queryIndex=0;const dispatch=(pipe,n)=>{const pass=enc.beginComputePass(this.query?{timestampWrites:{querySet:this.query,beginningOfPassWriteIndex:queryIndex++,endOfPassWriteIndex:queryIndex++}}:{});pass.setPipeline(pipe.p);pass.setBindGroup(0,pipe.group);pass.dispatchWorkgroups(Math.min(n,65535),Math.ceil(n/65535));pass.end();};
  dispatch(this.drivePipe,256);for(let i=0;i<this.meta.neural_steps;i++){dispatch(this.spmm[i%2],this.spmmGroups);if(this.segmentFinalize)dispatch(this.segmentFinalize[i%2],Math.ceil(this.N*2/64));}
  dispatch(this.normalize,64);[32,32,1,16].forEach((n,i)=>dispatch(this.densePipes[i],n));dispatch(this.gru[this.ping],16);dispatch(this.res[this.ping],1);this.ping=1-this.ping;
  dispatch(this.rmsPipe,Math.ceil(this.N/256));dispatch(this.finalPipe,this.options.captureAllBrains?224:64);enc.copyBufferToBuffer(this.output,0,this.readback,0,this.readback.size);if(this.query){enc.resolveQuerySet(this.query,0,queryIndex,this.queryBuffer,0);enc.copyBufferToBuffer(this.queryBuffer,0,this.queryRead,0,queryIndex*8);}this.device.queue.submit([enc.finish()]);
  await this.readback.mapAsync(GPUMapMode.READ);const data=new Float32Array(this.readback.getMappedRange()).slice();this.readback.unmap();
  this.timing={wall:performance.now()-start};if(this.query){await this.queryRead.mapAsync(GPUMapMode.READ);const stamps=new BigUint64Array(this.queryRead.getMappedRange());this.timing.passes=Array.from({length:queryIndex/2},(_,i)=>Number(stamps[2*i+1]-stamps[2*i])/1e6);this.queryRead.unmap();}
  if(this.dead)throw Error('GPU stopped: '+this.lostMessage);
  if(!data.every(Number.isFinite))throw Error('Non-finite GPU inference');
  const twist=[],precision=[];
  for(let a=0;a<8;a++){const v=Array.from(data.slice(a*6,a*6+3),x=>Math.fround(x/.08)),ratio=Math.max(1,Math.hypot(v[0],v[1])/.18,Math.abs(v[2])/.4);twist.push(v.map((x,i)=>Math.fround(x/ratio/(i===2?1.4:.65))));precision.push(Array.from(data.slice(a*6+3,a*6+6),x=>Math.fround(Math.exp(Math.max(-3,Math.min(3,x))))));}
  return{twist,precision,rawProposal:Array.from({length:8},(_,i)=>Array.from(data.slice(i*6,i*6+6))),brain:Array.from(data.slice(48,2096)),brain_all:this.options.captureAllBrains?Array.from({length:8},(_,i)=>Array.from(i?data.slice(6200+(i-1)*2048,6200+i*2048):data.slice(48,2096))):undefined,brain_rms:Array.from(data.slice(2096,2104)),features:Array.from({length:8},(_,i)=>Array.from(data.slice(2104+i*512,2104+(i+1)*512)))};
 }
}
