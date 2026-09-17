// Compute only: no browser automation or browser configuration.
import fs from 'node:fs';
import {FlyGPU} from '../gpu-model.mjs';
const {create,globals}=await import('../local-deps/package/index.js');
Object.assign(globalThis,globals);
Object.defineProperty(globalThis,'navigator',{value:{gpu:create(process.env.DAWN_ADAPTER?['backend=d3d12','adapter='+process.env.DAWN_ADAPTER]:[])},configurable:true});
const realFetch=globalThis.fetch;globalThis.fetch=async(url,opts)=>String(url).startsWith('file:')?new Response(await fs.promises.readFile(new URL(url))):realFetch(url,opts);
const meta=JSON.parse(fs.readFileSync(new URL('../model/model.json',import.meta.url)));
const gpu=await FlyGPU.create(meta,()=>{},{kernel:process.argv[2]||'tile16'});
console.log(JSON.stringify({adapter:gpu.adapterInfo,kernel:gpu.kernel,features:[...gpu.device.features]}));
const rows=JSON.parse(fs.readFileSync(new URL('../model/parity.json',import.meta.url))).rows;
for(let i=0;i<10;i++){await gpu.step(rows[i].obs);console.log(JSON.stringify({i,...gpu.timing}));}
gpu.device.destroy();delete globalThis.navigator;process.exit(0);
