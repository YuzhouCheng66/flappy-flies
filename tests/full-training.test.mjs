import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';

const root=new URL('../',import.meta.url),folder=new URL('local-results/full-campaign-training/checkpoints/',root);
const meta=JSON.parse(fs.readFileSync(new URL('model/model.json',root))),original=fs.readFileSync(new URL('model/weights.bin',root));
const checkpoints=fs.existsSync(folder)?fs.readdirSync(folder).filter(n=>n.endsWith('.json')).map(n=>({name:n,info:JSON.parse(fs.readFileSync(new URL(n,folder)))})).filter(x=>x.info.schema==='full-connectome-decoder-finetune-v1'):[];

test('Local full-brain candidates preserve every declared frozen parameter and data split',{skip:!checkpoints.length},()=>{
 for(const {name,info} of checkpoints){
  assert.equal(info.fullConnectome,true,name);
  const bytes=fs.readFileSync(new URL(info.weightsFile,folder));
  assert.equal(createHash('sha256').update(bytes).digest('hex'),info.weightsSha256,name);
  assert.equal(bytes.length,original.length,name);
  const allowed=new Uint8Array(bytes.length);
  for(const [key,a] of Object.entries(meta.arrays)){
   if(key.startsWith('net.')||(info.trainable.includes('residual.*')&&key.startsWith('residual.')))allowed.fill(1,a.offset*4,(a.offset+a.length)*4);
  }
  for(let i=0;i<bytes.length;i++)if(!allowed[i])assert.equal(bytes[i],original[i],`${name} byte ${i}`);
  const train=new Set(info.data.filter(r=>r.split==='train').map(r=>r.level));
  for(const r of info.data.filter(r=>r.split==='validation'))assert.equal(train.has(r.level),false,`${name} leaks ${r.level}`);
  if(info.maxNativeParityError!==undefined)assert.ok(info.maxNativeParityError<2e-4,name);
 }
});
