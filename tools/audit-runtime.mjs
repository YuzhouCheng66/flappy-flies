// Native compute-only profiling. Does not automate or configure a browser.
import fs from 'node:fs';
import {FlyGPU} from '../gpu-model.mjs';
import {Engine,observe,coordinate,transition} from '../engine.mjs';
const {create,globals}=await import('../local-deps/package/index.js');
Object.assign(globalThis,globals);
Object.defineProperty(globalThis,'navigator',{value:{gpu:create(process.env.DAWN_ADAPTER?['backend=d3d12','adapter='+process.env.DAWN_ADAPTER]:[])},configurable:true});
const originalFetch=globalThis.fetch;
globalThis.fetch=async(url,options)=>String(url).startsWith('file:')?new Response(await fs.promises.readFile(new URL(url))):originalFetch(url,options);
const meta=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url)));
const rows=JSON.parse(fs.readFileSync(new URL('../model/parity.json',import.meta.url))).rows;
const gpu=await FlyGPU.create(meta,()=>{},{kernel:'tile16'});
const samples=[];
for(let i=0;i<34;i++){await gpu.step(rows[i%rows.length].obs);if(i>=4)samples.push({...gpu.timing});}
const avg=xs=>xs.reduce((a,b)=>a+b,0)/xs.length;
const percentile=(xs,p)=>[...xs].sort((a,b)=>a-b)[Math.ceil(xs.length*p)-1];
const labels=['sensory','graph_1','graph_2','graph_3','graph_4','normalize','dense_1','dense_2','dense_out','embedding','GRU','residual','RMS','decode'];
const engine=new Engine(meta,null);engine.reset(91000);
const sums={observation:0,coordination:0,physics:0},count=800;
for(let i=0;i<count;i++){
 const row=rows[i%rows.length];let start=performance.now();
 observe(row.state,engine.layout,row.stage_before,row.context,meta);sums.observation+=performance.now()-start;
 start=performance.now();coordinate(row.mean,row.precision,row.obs,meta);sums.coordination+=performance.now()-start;
 start=performance.now();transition(row.state,row.next.forces,engine.walls,meta);sums.physics+=performance.now()-start;
}
const report={runtime:'Native Dawn 0.6.1 D3D12, NOT browser',adapter:gpu.adapterInfo,kernel:gpu.kernel,samples:30,
 neural_wall_ms:{median:percentile(samples.map(s=>s.wall),.5),p95:percentile(samples.map(s=>s.wall),.95),mean:avg(samples.map(s=>s.wall))},
 gpu_pass_mean_ms:Object.fromEntries(labels.map((label,i)=>[label,avg(samples.map(s=>s.passes?.[i]??0))])),
 cpu_components_mean_ms:Object.fromEntries(Object.entries(sums).map(([k,v])=>[k,v/count])),
 notes:'CPU stages measured independently using recorded inputs, not whole-loop or browser timing. GPU timestamp resolution may round tiny passes to zero.',budget_ms:80/3.3};
fs.mkdirSync(new URL('../local-results/',import.meta.url),{recursive:true});
fs.writeFileSync(new URL('../local-results/runtime-feasibility.json',import.meta.url),JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));gpu.device.destroy();delete globalThis.navigator;process.exit(0);
