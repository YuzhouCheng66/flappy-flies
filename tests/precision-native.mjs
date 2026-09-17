// Native compute only. Does not launch, control or configure any browser.
import fs from 'node:fs';
import {auditPrecision} from './precision-audit.mjs';
const {create,globals}=await import('../local-deps/package/index.js');
Object.assign(globalThis,globals);
Object.defineProperty(globalThis,'navigator',{value:{gpu:create(process.env.DAWN_ADAPTER?['backend=d3d12','adapter='+process.env.DAWN_ADAPTER]:[])},configurable:true});
const originalFetch=globalThis.fetch;
globalThis.fetch=async(url,opts)=>String(url).startsWith('file:')?new Response(await fs.promises.readFile(new URL(url))):originalFetch(url,opts);
const options={kernel:process.argv[2]||'tile16',precision:process.argv[3]||'f32',seeds:(process.argv[4]||'').split(',').filter(Boolean).map(Number),memoryLayout:process.argv[5]||'original'};
const path=new URL('../local-results/precision-native.jsonl',import.meta.url);fs.mkdirSync(new URL('../local-results/',import.meta.url),{recursive:true});
await auditPrecision(options,row=>{if(row.type==='progress')return;const record={...row,runtime:'native Dawn D3D12; NOT browser',time:new Date().toISOString()};console.log(JSON.stringify(record));fs.appendFileSync(path,JSON.stringify(record)+'\n');});
delete globalThis.navigator;process.exit(0);
