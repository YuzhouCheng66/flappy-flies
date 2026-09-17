/* Browser-native lossless loader. Run in a Web Worker, not the render thread.
 * No API keys, Python server, remote inference, edge pruning or quantization.
 * gpu-model.mjs evaluates the complete graph in visitor-side WebGPU.
 */
const SCHEMA='flappy-flies-graph-gzip-delta-u32-count-u16-v1';
export async function sha256(bytes){
 const hash=await crypto.subtle.digest('SHA-256',bytes);
 return Array.from(new Uint8Array(hash),v=>v.toString(16).padStart(2,'0')).join('');
}
export async function gunzip(bytes){
 return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
export function decodeGraphChunk(buffer){
 if(buffer.byteLength<32)throw Error('Truncated graph header');
 const v=new DataView(buffer),magic=new TextDecoder().decode(new Uint8Array(buffer,0,8));
 const rowStart=v.getUint32(8,true),rows=v.getUint32(12,true),edges=v.getUint32(16,true),neurons=v.getUint32(20,true);
 if(magic!=='FFGRAPH1'||v.getUint32(24,true)!==1||v.getUint32(28,true)!==0)throw Error('Unsupported graph format');
 if(!neurons||rowStart+rows>neurons||buffer.byteLength!==32+4*rows+6*edges)throw Error('Invalid graph sizes');
 const degrees=new Uint32Array(buffer,32,rows),delta=new Uint32Array(buffer,32+4*rows,edges),counts=new Uint16Array(buffer,32+4*rows+4*edges,edges);
 const crow=new Uint32Array(rows+1),col=new Uint32Array(edges),weights=new Float32Array(edges);
 let cursor=0;
 for(let r=0;r<rows;r++){
  const end=cursor+degrees[r];if(end>edges)throw Error('Invalid row degree');
  let column=0,total=0;
  for(let i=cursor;i<end;i++){
   column+=delta[i];if(column>=neurons)throw Error('Column outside graph');
   col[i]=column;total+=counts[i];
  }
  const denominator=Math.max(1,Math.fround(total));
  for(let i=cursor;i<end;i++)weights[i]=Math.fround(Math.fround(counts[i]/denominator)*.5);
  cursor=end;crow[r+1]=cursor;
 }
 if(cursor!==edges)throw Error('Row count mismatch');
 return{rowStart,rows,edges,neurons,crow,col,counts,weights};
}
export async function verifiedChunk(bytes,entry){
 if(bytes.byteLength!==entry.bytes||await sha256(bytes)!==entry.sha256)throw Error('Graph download integrity check failed');
 const decoded=decodeGraphChunk(await gunzip(bytes));
 if(decoded.rowStart!==entry.row_start||decoded.rows!==entry.rows||decoded.edges!==entry.edges)throw Error('Graph manifest mismatch');
 for(const [array,key] of [[decoded.col,'columns_sha256'],[decoded.counts,'counts_sha256'],[decoded.weights,'weights_sha256']]){
  const data=new Uint8Array(array.buffer,array.byteOffset,array.byteLength);
  if(await sha256(data)!==entry[key])throw Error('Decoded graph differs from Python reference: '+key);
 }
 return decoded;
}
export async function loadGraph(manifestUrl,{onProgress=()=>{},signal,useCache=true}={}){
 const response=await fetch(manifestUrl,{signal});if(!response.ok)throw Error('Graph manifest unavailable');
 const manifest=await response.json();
 if(manifest.schema!==SCHEMA||manifest.pruned_edges!==0||!manifest.lossless||!Number.isInteger(manifest.neurons)||!Number.isInteger(manifest.edges)||manifest.neurons<1||manifest.edges<0||manifest.neurons>1000000||manifest.edges>40000000)throw Error('Unsupported graph manifest');
 const crow=new Uint32Array(manifest.neurons+1),col=new Uint32Array(manifest.edges),weights=new Float32Array(manifest.edges);
 let cache=null,row=0,edge=0,received=0,networkBytes=0;
 if(useCache&&typeof caches!=='undefined'){try{cache=await caches.open('flappy-flies-graph-v1');}catch{/* Cache denial must not break inference. */}}
 for(const entry of manifest.chunks){
  if(signal?.aborted)throw new DOMException('Aborted','AbortError');
  if(!/^graph-\d{6}-[a-f0-9]{12}\.bin\.gz$/.test(entry.file)||entry.row_start!==row)throw Error('Invalid graph chunk sequence');
  const url=new URL(entry.file,manifestUrl).href;
  let cached=null;try{cached=await cache?.match(url);}catch{}
  let bytes,decoded;
  if(cached){try{bytes=await cached.arrayBuffer();decoded=await verifiedChunk(bytes,entry);}catch{try{await cache?.delete(url);}catch{}}}
  if(!decoded){
   const download=await fetch(url,{signal});if(!download.ok)throw Error('Graph download failed');
   bytes=await download.arrayBuffer();networkBytes+=bytes.byteLength;decoded=await verifiedChunk(bytes,entry);
   try{await cache?.put(url,new Response(bytes,{headers:{'Content-Type':'application/octet-stream'}}));}catch{/* Quota eviction does not change model data. */}
  }
  if(decoded.neurons!==manifest.neurons||edge+decoded.edges>manifest.edges)throw Error('Graph shape mismatch');
  for(let i=0;i<=decoded.rows;i++)crow[row+i]=edge+decoded.crow[i];
  col.set(decoded.col,edge);weights.set(decoded.weights,edge);row+=decoded.rows;edge+=decoded.edges;
  received+=bytes.byteLength;onProgress({received,total:manifest.graph_download_bytes,networkBytes,rowsReady:row});
 }
 if(row!==manifest.neurons||edge!==manifest.edges||received!==manifest.graph_download_bytes)throw Error('Incomplete graph');
 return{crow,col,weights,manifest,networkBytes};
}
