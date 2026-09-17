import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {COURSE_IDS} from '../campaign-selection.mjs';
const folder=new URL('../local-results/campaign/',import.meta.url),file=new URL('brains.json',folder);
test('All 30 courses have distinct, authentic CNS 02–08 aligned to unchanged poses',{skip:!fs.existsSync(file)},()=>{
 const index=JSON.parse(fs.readFileSync(file)),manifest=JSON.parse(fs.readFileSync(new URL('manifest.json',folder)));
 for(const source of COURSE_IDS){
  const row=index.levels[source],entry=manifest.levels.find(e=>e.id===source);assert.ok(row,`Missing ${source}`);
  assert.equal(row.recordingSha256,entry.sha256);assert.equal(row.weightsSha256,entry.weightsSha256);
  assert.equal(row.poseMaxError,0);assert.ok(row.agent01MaxDisplayError<1/32767);
  assert.deepEqual(row.agents.map(a=>a.agent),[2,3,4,5,6,7,8]);
  const hashes=[];
  for(const a of row.agents){const packed=fs.readFileSync(new URL(a.file,folder)),raw=gunzipSync(packed);assert.equal(createHash('sha256').update(packed).digest('hex'),a.sha256);assert.equal(raw.length,row.frames*2048*2);assert.ok(raw.subarray(0,4096).every(x=>x===0));hashes.push(createHash('sha256').update(raw).digest('hex'));}
  assert.equal(new Set(hashes).size,7,`Duplicated CNS traces ${source}`);
 }
});
