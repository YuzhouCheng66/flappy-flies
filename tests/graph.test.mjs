import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {verifiedChunk,decodeGraphChunk,sha256} from '../graph-loader.mjs';
test('Malformed graph blocks fail closed',()=>{
 assert.throws(()=>decodeGraphChunk(new ArrayBuffer(8)),/Truncated/);
 assert.throws(()=>decodeGraphChunk(new ArrayBuffer(32)),/Unsupported/);
});
test('Every graph column, synapse count, and float32 weight matches Python',async()=>{
 const root=new URL('../graph/',import.meta.url),m=JSON.parse(await fs.readFile(new URL('manifest.json',root)));let rows=0,edges=0,bytes=0;
 for(const entry of m.chunks){const buf=await fs.readFile(new URL(entry.file,root)),data=buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength),r=await verifiedChunk(data,entry);assert.equal(r.rowStart,rows);assert.equal(r.neurons,m.neurons);rows+=r.rows;edges+=r.edges;bytes+=entry.bytes;}
 assert.equal(rows,165122);assert.equal(edges,25563197);assert.equal(bytes,62760837);
});
test('Frozen weights match release SHA-256',async()=>{
 const m=JSON.parse(await fs.readFile(new URL('../model/model.json',import.meta.url))),w=await fs.readFile(new URL('../model/weights.bin',import.meta.url));assert.equal(await sha256(w),m.weights_sha256);
});
