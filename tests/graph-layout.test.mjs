import test from 'node:test';
import assert from 'node:assert/strict';
import {reorderBySourceFrequency} from '../graph-layout.mjs';
test('Source-frequency layout is a bijection preserving every edge, count, weight and row order',()=>{
 const graph={crow:new Uint32Array([0,3,3,5,7]),col:new Uint32Array([3,2,0,3,2,3,2]),counts:new Uint16Array([1,2591,3,4,5,6,7]),weights:new Float32Array([.1,.2,.3,.4,.5,.6,.7])};
 const original=structuredClone(graph),{graph:g,originalToStorage:p,storageToOriginal:inv}=reorderBySourceFrequency(graph);
 assert.deepEqual(graph,original);assert.deepEqual([...inv],[2,3,0,1]);assert.equal(new Set(p).size,4);
 for(let old=0;old<4;old++){
  assert.equal(inv[p[old]],old);assert.equal(g.crow[p[old]+1]-g.crow[p[old]],graph.crow[old+1]-graph.crow[old]);
  for(let j=graph.crow[old];j<graph.crow[old+1];j++){
   const k=g.crow[p[old]]+j-graph.crow[old];assert.equal(inv[g.col[k]],graph.col[j]);assert.equal(g.counts[k],graph.counts[j]);assert.equal(g.weights[k],graph.weights[j]);
  }
 }
});
