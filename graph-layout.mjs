// Lossless memory permutation: frequently read source neurons are adjacent.
// No edge removal or summation-order change within a row. Semantic neuron IDs
// are mapped back at sensory injection, readout and visualization boundaries.
export function reorderBySourceFrequency(graph){
 const n=graph.crow.length-1,frequency=new Uint32Array(n);
 for(const column of graph.col)frequency[column]++;
 const storageToOriginal=Uint32Array.from({length:n},(_,i)=>i);
 storageToOriginal.sort((a,b)=>frequency[b]-frequency[a]||a-b);
 const originalToStorage=new Uint32Array(n);
 storageToOriginal.forEach((original,storage)=>originalToStorage[original]=storage);
 const crow=new Uint32Array(n+1),col=new Uint32Array(graph.col.length),counts=new Uint16Array(graph.counts.length),weights=graph.weights?new Float32Array(graph.weights.length):undefined;
 let cursor=0;
 for(let storage=0;storage<n;storage++){
  const original=storageToOriginal[storage];crow[storage]=cursor;
  for(let j=graph.crow[original];j<graph.crow[original+1];j++,cursor++){
   col[cursor]=originalToStorage[graph.col[j]];counts[cursor]=graph.counts[j];if(weights)weights[cursor]=graph.weights[j];
  }
 }
 crow[n]=cursor;
 return {graph:{...graph,crow,col,counts,weights},originalToStorage,storageToOriginal};
}
